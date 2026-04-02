import torch
import os
import argparse
import random
import json
import numpy as np
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from functools import partial
import transformers
from torch.utils.data import DataLoader

from dataset import GWExplanation
from utils import collate_fn, evaluate_collate_fn
from train import train

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name())

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--resume_ckpt_id', type=str, default='GW_test')

    parser.add_argument('--checkpoints_dir', type=str, default='/scratch/user/xxxx/checkpoints')
    parser.add_argument('--imageset_dir', type=str, default='/scratch/user/xxxx/GuessWhere')
    parser.add_argument('--ve_dir', type=str, default='/scratch/user/xxxx/GuessWhereVE')
    parser.add_argument('--explanation_list_dir', type=str, default='/scratch/user/xxxx/explanation_dic.json')
    parser.add_argument('--knowledge_dir', type=str, default='/scratch/user/xxxx/GuessWhereKnowledge/knowledge.json')
    parser.add_argument('--date_round_knowledge_list_dir', type=str, default='/scratch/user/xxxx/GuessWhereKnowledge/date_round_knowledge_map.json')
    parser.add_argument('--text_location_list_dir', type=str, default='/scratch/user/xxxx/lat_lng_2location.json')
    parser.add_argument('--train_test_split_path', type=str, default='/scratch/user/xxxx/GeoExplain_train_test_split.json')

    parser.add_argument('--prediction_save_folder', type=str, default='/scratch/user/xxxx/GeoGuessResults')

    parser.add_argument('--sequence_length', type=int, default=128)
    parser.add_argument('--using_knowledge', type=int, default=1)
    parser.add_argument('--using_visual_clues', type=int, default=1)

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')

    parser.add_argument('--batch_size', type=int, default=1)

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

    # model, processor
    if args.resume == 1:
        resume_model_path = os.path.join(args.checkpoints_dir, args.resume_ckpt_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(resume_model_path, torch_dtype="auto", device_map="auto")
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(resume_model_path, padding_side='right')
        prediction_save_path = os.path.join(args.prediction_save_folder, f'{args.resume_ckpt_id}_results.json')
        print(f'We Are Using: {args.resume_ckpt_id}')
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(args.model_name, padding_side='right')
        prediction_save_path = os.path.join(args.prediction_save_folder, f'Qwen2-VL-7B-Instruct_K{str(args.using_knowledge)}_VE{str(args.using_visual_clues)}_results.json')
        print(prediction_save_path)
        print(f'We Are Using: {args.model_name}')

    # dataloader
    gw_dataset = GWExplanation(imageset_path=args.imageset_dir,
                            ve_path=args.ve_dir,
                            knowledge_path=args.knowledge_dir,
                            explanation_list_path=args.explanation_list_dir,
                            date_round_knowledge_list_path=args.date_round_knowledge_list_dir,
                            text_location_list_path=args.text_location_list_dir,
                            train_test_split_path = args.train_test_split_path,
                            isTrain=False,
                            using_knowledge = args.using_knowledge,
                            using_visual_clues = args.using_visual_clues)
    dataloader = DataLoader(
        gw_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(evaluate_collate_fn, processor=processor, device=device))

    predictions = {}
    for step, batch in enumerate(dataloader):
        inputs, date_round = batch
        inputs = inputs.to(device)
        if inputs == None:
            print("Error")
            continue
            
            
        generated_ids = model.generate(**inputs, max_new_tokens=2800)
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        input_text = processor.batch_decode(
            inputs['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        predictions[date_round[0]] = output_text
    with open(prediction_save_path, 'w') as f:
        json.dump(predictions, f)