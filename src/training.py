import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from src.tuned_lens_model import SmolLMTunedLens

def save_lens(tuned_lens: SmolLMTunedLens, save_path = "smollm2_tuned_lens_fineweb.pt"):
  torch.save(tuned_lens.state_dict(), save_path)
  print(f"Tuned Lens successfully saved to {save_path}")


def train_tuned_lens(
    tuned_lens, 
    base_model, 
    dataloader, 
    muon_lr=0.02, 
    adamw_lr=3e-4, 
    epochs=1,
    accumulation_steps=4,
    device="cuda",
    saving_rate=1000
):
    """
    Trains the Tuned Lens on a frozen base model.
    
    Args:
        tuned_lens: The SmolLMTunedLens module.
        base_model: The frozen LLM (e.g., SmolLM2).
        dataloader: PyTorch DataLoader yielding batches with "input_ids".
        muon_lr: Learning rate for the Muon optimizer (2D weights).
        adamw_lr: Learning rate for the AdamW optimizer (1D biases).
        epochs: Number of training epochs (1 is usually enough).
        device: "cuda" or "cpu".
    """
    # Muon strictly requires 2D parameters (weights). AdamW handles 1D (biases).
    muon_params = []
    adamw_params = []
    
    for name, param in tuned_lens.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
            
    # Initialize optimizers
    optimizer_muon = torch.optim.Muon(
        muon_params, 
        lr=muon_lr
    )
    optimizer_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr)
    
    # Initialize lr schedulers
    total_steps = (len(dataloader) * epochs) // accumulation_steps
    warmup_steps = int(0.001 * total_steps) # 0.1% of training used for warmup
    
    scheduler_muon = get_cosine_schedule_with_warmup(
        optimizer_muon, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scheduler_adamw = get_cosine_schedule_with_warmup(
        optimizer_adamw, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    # Ensure correct modes
    tuned_lens.train()
    base_model.eval() # Base model must remain completely frozen
    
    print("Starting Tuned Lens Training...")
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                outputs = base_model(input_ids, output_hidden_states=True)
                
            all_hidden_states = outputs.hidden_states
            
            # Causal Shift labels: shared across all layers
            shift_labels = input_ids[:, 1:].contiguous()
            shift_labels_flat = shift_labels.view(-1)
            
            # We track the unscaled display loss just for logging
            total_loss = 0
            
            # Calculate the loss layer by layer
            for layer_idx in range(tuned_lens.num_layers - 1):
                
                h_state = all_hidden_states[layer_idx + 1]
                
                # Single layer forward pass
                logits = tuned_lens(h_state, layer_idx)
                
                # Shift and Flatten for loss calculation
                shift_logits = logits[:, :-1, :].contiguous()
                shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                
                # Compute raw loss
                raw_layer_loss = loss_fn(shift_logits_flat, shift_labels_flat)
                
                # Accumulate for display (we use the unscaled value so logs make sense)
                total_loss += raw_layer_loss.item()
                
                # Scale loss for gradient accumulation before backward
                layer_loss = raw_layer_loss / accumulation_steps
                
                # Instant backward pass frees the graph for this layer
                layer_loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer_muon.step()
                optimizer_adamw.step()

                scheduler_muon.step()
                scheduler_adamw.step()

                optimizer_muon.zero_grad()
                optimizer_adamw.zero_grad()
            
            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx:04d} | Loss: {total_loss:.4f}")

            # Saving
            if (batch_idx % saving_rate == 0 and batch_idx !=0 ) or (batch_idx + 1) == len(dataloader):
                save_lens(tuned_lens, save_path=f"smollm2_tuned_lens_{batch_idx}.pt")

    print("Training Complete!")
    return tuned_lens