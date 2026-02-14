import torch

CONFIG = {
    # Model Architecture
    "vocab_size": 32000,
    "d_model": 512,
    "n_head": 8,
    "n_layers": 12,         
    "physical_layers": 2,  
    "block_size": 512,
    "num_experts": 4,
    "top_k": 4,

    # Device & Paths
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "model/best_moe_model.pth",       
    "tokenizer_path": "tokenizer", 

    # Training Hyperparameters
    "batch_size": 8,
    "epochs": 3,
    "lr": 3e-4,
    "accumulation_steps": 4,
}
```
