import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import models, pre_tokenizers, trainers, Tokenizer, decoders
from transformers import PreTrainedTokenizerFast
from torch.cuda.amp import autocast, GradScaler
import gc


from config.config import CONFIG



# ==========================================
# 2. Model Architecture
# ==========================================
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * 8)
        self.w2 = nn.Linear(d_model * 4, d_model)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]
            expert_weight = weights[:, :, k].unsqueeze(-1)
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i).unsqueeze(-1)
                if mask.any():
                    output += mask * expert(x) * expert_weight
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, num_experts, top_k):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, top_k)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x = residual + self.moe(x)
        return x

class SharedMoETransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = CONFIG['d_model']
        self.token_emb = nn.Embedding(CONFIG['vocab_size'], self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, CONFIG['block_size'], self.d_model))
        self.drop = nn.Dropout(0.1)

        self.physical_layers = nn.ModuleList([
            TransformerBlock(self.d_model, CONFIG['n_head'], CONFIG['num_experts'], CONFIG['top_k']) 
            for _ in range(CONFIG['physical_layers'])
        ])
        
        self.n_layers = CONFIG['n_layers']
        self.norm = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, CONFIG['vocab_size'])

    def forward(self, x):
        b, t = x.size()
        x = self.token_emb(x) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)

        for i in range(self.n_layers):
            layer_idx = i % CONFIG['physical_layers']
            x = self.physical_layers[layer_idx](x, mask)

        x = self.norm(x)
        return self.fc_out(x)

# ==========================================
# 3. Tokenizer Builder & Dataset
# ==========================================
def build_tokenizer(dataset):
    print("🔨 Building custom tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    from tokenizers.pre_tokenizers import ByteLevel
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=CONFIG['vocab_size'], 
        special_tokens=["<|endoftext|>", "<|pad|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    def batch_iterator():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]['output']

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>"
    )
    fast_tokenizer.save_pretrained(CONFIG['tokenizer_path'])
    return fast_tokenizer

class PythonCodeDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}<|endoftext|>"
        enc = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0)

# ==========================================
# 4. Training Main Logic
# ==========================================
def main():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🔥 Training Device: {CONFIG['device']}")

    print("📚 Downloading dataset (flytech/python-codes-25k)...")
    dataset = load_dataset("flytech/python-codes-25k")
    
    if not os.path.exists(CONFIG['tokenizer_path']):
        tokenizer = build_tokenizer(dataset['train'])
    else:
        print("✅ Loading existing custom tokenizer")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(CONFIG['tokenizer_path'])

    split_data = dataset['train'].train_test_split(test_size=0.05)
    train_dataset = PythonCodeDataset(split_data['train'], tokenizer, CONFIG['block_size'])
    val_dataset = PythonCodeDataset(split_data['test'], tokenizer, CONFIG['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    model = SharedMoETransformer().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(CONFIG['save_path']):
        print(f"🔄 Resuming training: {CONFIG['save_path']}")
        model.load_state_dict(torch.load(CONFIG['save_path'], map_location=CONFIG['device']))

    print(f"\n🚀 Training started! ({len(train_dataset)} samples)")
    accumulation_steps = CONFIG.get('accumulation_steps', 4)

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(CONFIG['device'])
            targets = inputs.clone()

            # autocast handles mixed precision
            with autocast():
                logits = model(inputs)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            
            if i % 100 == 0:
                print(f"   [Epoch {epoch+1}] Step {i}/{len(train_loader)} | Loss: {loss.item() * accumulation_steps:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        
        print("📝 Validating...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(CONFIG['device'])
                targets = inputs.clone()
                # autocast for validation
                with autocast():
                    logits = model(inputs)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"✨ Epoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"💾 New record! Saving model ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
        
        torch.save(model.state_dict(), "last_checkpoint.pth")

if __name__ == "__main__":
    main()