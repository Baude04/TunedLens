import torch
import torch.nn as nn

class SmolLMTunedLens(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.num_layers = len(base_model.model.layers)
        self.hidden_size = base_model.config.hidden_size
        
        self.translators = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) 
            for _ in range(self.num_layers - 1)
        ])
        
        for translator in self.translators:
            nn.init.eye_(translator.weight)
            nn.init.zeros_(translator.bias)
            
        self.final_norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        
        self.final_norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def forward(self, hidden_state, layer_idx):
        """
        Translates a single hidden state from a specific layer.
        """
        translated_state = self.translators[layer_idx](hidden_state)
        normed_state = self.final_norm(translated_state)
        logits = self.lm_head(normed_state)
        return logits