# Run: streamlit run app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
import os
import time

st.set_page_config(
    page_title="aiaiai",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("aiaiai")
from config.config import CONFIG



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

@st.cache_resource
def load_model():
    device = CONFIG['device']
    
    if not os.path.exists(CONFIG['tokenizer_path']):
        st.error("❌ Tokenizer folder not found.")
        return None, None
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CONFIG['tokenizer_path'])
    
    model = SharedMoETransformer().to(device)
    
    if os.path.exists(CONFIG['save_path']):
        try:
            model.load_state_dict(torch.load(CONFIG['save_path'], map_location=device))
            model.eval()
            return model, tokenizer
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, None
    else:
        st.error(f"❌ Model file ({CONFIG['save_path']}) not found.")
        return None, None

model, tokenizer = load_model()

with st.sidebar:
    st.header("Generation Options")
    mode = st.selectbox("Select Mode", ["CoT (Logical Reasoning)", "Basic (Default)"])
    temp = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.3)
    max_tokens = st.slider("Max Length", 100, 1000, 500)
    
    st.divider()
    st.info(f"Running on: {CONFIG['device']}")
    if model:
        st.success("Model loaded successfully")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if "CoT" in mode:
            input_text = f"### Instruction:\n{prompt}\nThink step-by-step:\n1. Define function.\n2. Implement logic.\n3. Return result.\n\n### Response:\n"
        else:
            input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"

        ids = tokenizer.encode(input_text, return_tensors='pt').to(CONFIG['device'])
        
        with torch.no_grad():
            for _ in range(max_tokens):
                cond = ids[:, -CONFIG['block_size']:]
                logits = model(cond)
                next_token_logits = logits[:, -1, :]
                
                for token_id in set(ids[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= 1.2
                    else:
                        next_token_logits[0, token_id] *= 1.2

                next_token_logits = next_token_logits / temp
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                ids = torch.cat((ids, next_token), dim=1)
                word = tokenizer.decode(next_token[0])
                
                full_response += word
                message_placeholder.markdown(full_response + "▌")
                
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})