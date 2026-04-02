import torch
import numpy as np
import random
import argparse
import os
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AdamW
from functools import partial
import transformers
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR

from dataset import GWExplanation
from utils import collate_fn
from train import train
from train_distribute import train_distribute

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    # print('We will use the GPU:', torch.cuda.get_device_name())

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--resume_ckpt_id', type=str, default='GW_2')
    parser.add_argument('--save_ckpt_id', type=str, default='GW_2')

    parser.add_argument('--checkpoints_dir', type=str, default='/scratch/user/xxxx/checkpoints')
    parser.add_argument('--imageset_dir', type=str, default='/scratch/user/xxxx/GuessWhere')
    parser.add_argument('--ve_dir', type=str, default='/scratch/user/xxxx/GuessWhereVE')
    parser.add_argument('--explanation_list_dir', type=str, default='/scratch/user/xxxx/explanation_dic.json')
    parser.add_argument('--knowledge_dir', type=str, default='/scratch/user/xxxx/GuessWhereKnowledge/knowledge.json')
    parser.add_argument('--date_round_knowledge_list_dir', type=str, default='/scratch/user/xxxx/GuessWhereKnowledge/date_round_knowledge_map.json')
    parser.add_argument('--text_location_list_dir', type=str, default='/scratch/user/xxxx/lat_lng_2location.json')
    parser.add_argument('--train_test_split_path', type=str, default='/scratch/user/xxxx/GeoExplain_train_test_split.json')

    parser.add_argument('--sequence_length', type=int, default=128)
    parser.add_argument('--train_distribute', type=bool, default=False)
    parser.add_argument('--using_knowledge', type=int, default=1)
    parser.add_argument('--using_visual_clues', type=int, default=1)

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--train', type=str, default='normal', help='train the model')
    parser.add_argument('--train_mode', type=str, default='lora', help='lora or not')

    parser.add_argument('--num_epochs', type=int, default=3, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=int, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args
if __name__=='__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN Enabled:", torch.backends.cudnn.enabled)

    # model, processor
    if args.resume == 1:
        resume_model_path = os.path.join(args.checkpoints_dir, args.resume_ckpt_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(resume_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(resume_model_path, padding_side='right')
    else:
        if args.train == 'normal':
            model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            model = model.to(device)
            processor = AutoProcessor.from_pretrained(args.model_name, padding_side='right')
        elif args.train == 'distributed':
            model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
            processor = AutoProcessor.from_pretrained(args.model_name, padding_side='right')
            print(f"Available GPUs: {torch.cuda.device_count()}")
            rank = int(os.environ['RANK'])
            print("Rank:", rank)
    # peft
    if args.train_mode == 'lora':
        print("Train in lora mode")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            # target_modules=['down_proj', 'v_proj', 'lm_head', 'o_proj', 'q_proj', 'gate_proj', 'up_proj', 'proj', 'k_proj', 'fc1', 'qkv', 'fc2'],
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if args.train == 'normal':
            model = model.to(device)
        if args.train == 'distributed':
            model = model.cuda(rank)

    # dataloader
    gw_dataset = GWExplanation(imageset_path=args.imageset_dir,
                            ve_path=args.ve_dir,
                            knowledge_path=args.knowledge_dir,
                            explanation_list_path=args.explanation_list_dir,
                            date_round_knowledge_list_path=args.date_round_knowledge_list_dir,
                            text_location_list_path=args.text_location_list_dir,
                            train_test_split_path = args.train_test_split_path,
                            using_knowledge = args.using_knowledge,
                            using_visual_clues=args.using_visual_clues)
    if args.train == 'normal': 
        dataloader = DataLoader(
            gw_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=processor, device=device))
    elif args.train == 'distributed':
        sampler = DistributedSampler(gw_dataset, num_replicas=args.world_size, rank=rank)
        dataloader = DataLoader(
            gw_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=partial(collate_fn, processor=processor, device=device))


    # optimizer
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      betas=(0.9,0.98),
                      eps=args.eps,  # args.adam_epsilon  - default is 1e-8.
                      weight_decay=0.0
                      )
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

    # train
    if args.train == 'normal':
        train(args, model, processor, optimizer, dataloader, device)
    elif args.train == 'distributed':
        print('Train Distributed!')
        train_distribute(args, model, processor, optimizer, dataloader, device, rank)
