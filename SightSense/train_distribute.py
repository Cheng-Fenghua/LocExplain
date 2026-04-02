import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from functools import partial
from torch.cuda.amp import GradScaler, autocast
from qwen_vl_utils import process_vision_info
import datetime
import os
import torch.distributed as dist


def train_distribute(args, model, processor, optimizer, train_dataloader, device, rank):
    scaler = GradScaler()
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_grad_norm = args.max_grad_norm

    dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
    # local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    for epoch in range(args.num_epochs):
        model.train()
        accumulated_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch
            if inputs == None:
                print("Error")
                continue
            pixel_values = batch[0]["pixel_values"].cuda(rank)
            input_ids = batch[0]["input_ids"].cuda(rank)
            attention_mask = batch[0]["attention_mask"].cuda(rank)
            labels = batch[1].cuda(rank)
            image_grid_thw = batch[0]["image_grid_thw"].cuda(rank)
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
                
            accumulated_loss = accumulated_loss + loss.item()
            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
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
        break
    save_model_dir = os.path.join(args.checkpoints_dir, args.save_ckpt_id)
    if dist.get_rank() == 0:
        torch.save({'model_state_dict': model.module.state_dict()}, save_model_dir + '/model.pth')
    # model.save_pretrained(save_model_dir, exist_ok=True)
    processor.save_pretrained(save_model_dir, exist_ok=True)
    print('Save checkpoints successfully!')


if __name__=='__main__':
    print("successful!")