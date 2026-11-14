import torch
import torch.nn as nn

class SmolLMTunedLens(nn.Module):
    """Tuned Lens for SmolLM2-1.7B"""
    def __init__(self, base_model):
        super().__init__()
        
        # Retrieve dimensions from the base model's configuration
        self.num_layers = len(base_model.model.layers)
        self.hidden_size = base_model.config.hidden_size
        
        # Create the translators (Tuned Lens)
        # We create an affine transformation (Linear layer) for each intermediate layer.
        # No need to train a translator for the very last layer.
        self.translators = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) 
            for _ in range(self.num_layers - 1)
        ])
        
        # Initially, W = Identity Matrix and Bias = 0.
        # This way, at the start of training, the Tuned Lens behaves like a standard Logit Lens.
        for translator in self.translators:
            nn.init.eye_(translator.weight)
            nn.init.zeros_(translator.bias)
            
        # Retrieve final modules from the base model and freeze them
        self.final_norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.final_norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def forward(self, all_hidden_states):
        """
        Decodes all layers simultaneously.
        
        Args:
            all_hidden_states: Tuple of Tensors (from outputs.hidden_states).
                               Length should be (num_layers + 1).
                               
        Returns:
            Tensor of shape (num_layers - 1, batch_size, seq_len, vocab_size)
        """
        translated_states = []
        for idx, translator in enumerate(self.translators):
            # The output of 'idx' is at index 'idx + 1' in the tuple
            h_state = all_hidden_states[idx + 1]
            translated = translator(h_state)
            translated_states.append(translated)
            
        # Stack into shape: (num_layers - 1, batch_size, seq_len, hidden)
        stacked_translated_states = torch.stack(translated_states)
        
        # Pass through norm and head simultaneously via broadcasting
        normed_states = self.final_norm(stacked_translated_states)
        all_layer_logits = self.lm_head(normed_states)
        
        return all_layer_logits