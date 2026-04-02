
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataset import GWExplanation
from functools import partial
from qwen_vl_utils import process_vision_info
import datetime
import os


def train(args, model, processor, optimizer, train_dataloader, device):
    scaler = GradScaler()
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_grad_norm = args.max_grad_norm

    for epoch in range(args.num_epochs):
        model.train()
        accumulated_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch
            pixel_values = batch[0]["pixel_values"].to(device)
            input_ids = batch[0]["input_ids"].to(device)
            attention_mask = batch[0]["attention_mask"].to(device)
            labels = batch[1].to(device)
            image_grid_thw = batch[0]["image_grid_thw"].to(device)
            if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                print(f"NaN/Inf detected in pixel_values at step {step}")
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                print(f"NaN/Inf detected in input_ids at step {step}")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print(f"NaN/Inf detected in labels at step {step}")
            if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
                print(f"NaN/Inf detected in attention_mask at step {step}")
            if torch.isnan(image_grid_thw).any() or torch.isinf(image_grid_thw).any():
                print(f"NaN/Inf detected in image_grid_thw at step {step}")
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
            print(loss)

            scaler.scale(loss).backward()
            
            accumulated_loss = accumulated_loss + loss.item()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")
                optimizer.zero_grad()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"⚠️ Step {step}: NaN found in gradient of {name}!")
                    if torch.isinf(param.grad).any():
                        print(f"⚠️ Step {step}: Inf found in gradient of {name}!")
            if torch.isnan(loss).any():
                print(f"⚠️ Step {step}: NaN found in loss!")
                return
            if torch.isinf(loss).any():
                print(f"⚠️ Step {step}: Inf found in loss!")
                return
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {accumulated_loss:.4f}")
    save_model_dir = os.path.join(args.checkpoints_dir, args.save_ckpt_id)
    model.save_pretrained(save_model_dir, exist_ok=True)
    processor.save_pretrained(save_model_dir, exist_ok=True)
    print('Save checkpoints successfully!')