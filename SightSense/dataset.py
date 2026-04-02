import os
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoProcessor
import pprint
from qwen_vl_utils import process_vision_info
from functools import partial
from utils import collate_fn, evaluate_collate_fn
import torch

class GWExplanation(Dataset):
    def __init__(self, imageset_path, ve_path, knowledge_path, explanation_list_path, date_round_knowledge_list_path, text_location_list_path, train_test_split_path, isTrain=True, using_knowledge=True, using_visual_clues=True, **kwargs):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.isTrain = isTrain
        self.using_knowledge = using_knowledge
        self.using_visual_clues = using_visual_clues
        self.imageset_path = imageset_path
        self.ve_path = ve_path
        self.explanation_list_path = explanation_list_path
        self.train_test_split_path = train_test_split_path
        with open(self.explanation_list_path, 'r') as f:
            self.explanation_list = json.load(f)
        self.knowledge_path = knowledge_path
        with open(self.knowledge_path, 'r') as f:
            self.knowledge_list = json.load(f)
        self.date_round_knowledge_list_path = date_round_knowledge_list_path
        with open(self.date_round_knowledge_list_path, 'r') as f:
            self.date_round_knowledge_list = json.load(f)
        self.text_location_list_path = text_location_list_path
        with open(self.text_location_list_path, 'r') as f:
            self.text_location_list = json.load(f)
        with open(self.train_test_split_path, 'r') as f:
            self.train_test_split = json.load(f)

        if self.isTrain == True:
            use_samples = self.train_test_split['Train']
        else:
            use_samples = self.train_test_split['Test']

        self.dataset_list = []
        for sample_name in use_samples:
            year = sample_name.split('_')[1]
            date = sample_name.split('_')[3]
            round_number = sample_name.split('_')[5]
            lat = sample_name.split('_')[7]
            lng = sample_name.split('_')[9]
            explanation_date = date[:3] + ' ' + date[3:]
            explanation_round = str(int(round_number) + 1)
            explanations = self.explanation_list[year][explanation_date][explanation_round]
            date_round = f'{year} {explanation_date} round {explanation_round}'
            knowledge_keys = self.date_round_knowledge_list[date_round]
            sample_path = os.path.join(self.imageset_path, sample_name)
            # sample_ve_path = os.path.join(self.ve_path, sample_name)
            for explanation in explanations:
                item_dic = {'date-round': date_round,
                            'images_path': sample_path,
                            # 've_path': sample_ve_path,
                            'explanation': explanation,
                            'text_location': self.text_location_list[year][explanation_date][int(round_number)],
                            'lat': lat,
                            'lng': lng,
                            'knowledge_keys': knowledge_keys}
                self.dataset_list.append(item_dic)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        sample = self.dataset_list[idx]
        knowledge = []
        visual_clues = []
        for knowledge_key_dic in sample['knowledge_keys']:
            knowledge.append(self.knowledge_list[knowledge_key_dic['knowledge_key']])
            knowledge_patch_name = knowledge_key_dic['image_name']
            if knowledge_patch_name.split('_')[0] == 'thumbnail':
                visual_clue_name = 'thumbnail.png'
            else:
                visual_clue_name = knowledge_patch_name.split('_')[0] + '_' + knowledge_patch_name.split('_')[1] + "_" + knowledge_patch_name.split('_')[2] + '.png'
            visual_clue_path = os.path.join(sample['images_path'], visual_clue_name)
            visual_clues.append(visual_clue_path)
        
        thumbnail_path = os.path.join(sample['images_path'], 'thumbnail.png')
        queries = [
            {
                "type": "image",
                "image": thumbnail_path
            }
        ]
        if self.using_visual_clues == True:
            for visual_clue in visual_clues:
                visual_clue_query = {
                    "type": "image",
                    "image": visual_clue
                }
                queries.append(visual_clue_query)

        # append text into query
        if self.using_knowledge == True and self.using_visual_clues == True:
            text_prompt = f"According to the given image and knowledge, guess where the photo was taken and explain why in the format: 'PLACE (COUNTRY, STATE, CITY, STREET). EXPLANATION'. The first image is the given image. Following {len(knowledge)} images are detected visual clues. We provide {len(knowledge)} visual clues and corresponding external knowledge pieces. "
            for (idx, knowledge_text) in enumerate(knowledge):
                text_prompt = text_prompt + f"Knowledge pieces {idx + 1} (corresponding to image{idx + 2}): {knowledge_text}"
                if text_prompt[-1] != '.':
                    text_prompt = text_prompt + '.'            
        elif self.using_knowledge == False and self.using_visual_clues == True:
            text_prompt = f"According to the given image and knowledge, guess where the photo was taken and explain why in the format: 'PLACE (COUNTRY, STATE, CITY, STREET). EXPLANATION.'. The first image is the given image. Following {len(knowledge)} images are detected visual clues. "
        else:
            text_prompt = f"According to the given image and knowledge, guess where the photo was taken and explain why in the format: 'PLACE (COUNTRY, STATE, CITY, STREET). EXPLANATION.'."
        
        text_query = {
            "type": "text",
            "text": text_prompt
        }
        queries.append(text_query)

        if self.isTrain == True:
            messages = [
                {
                    "role": 'user',
                    "content": queries,
                },
                {
                    "role": 'assistant',
                    "content": "PLACE: " + sample['text_location'] + ' EXPLANATION: ' + sample['explanation']['text']
                }
            ]
        else:
            messages = [
                {
                    "role": 'user',
                    "content": queries,
                }
            ]
            return messages, sample['date-round']

        return messages

if __name__ == '__main__':
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    imageset_path = "/scratch/user/xxxx/GuessWhere"
    print(len(os.listdir(imageset_path)))
    explanation_list_path = "/scratch/user/xxxx/explanation_dic.json"
    knowledge_path = "/scratch/user/xxxx/GuessWhereKnowledge/knowledge.json"
    date_round_knowledge_list_path = "/scratch/user/xxxx/GuessWhereKnowledge/date_round_knowledge_map.json"
    text_location_list_path = "/scratch/user/xxxx/lat_lng_2location.json"
    train_test_split_path = '/scratch/user/xxxx/GeoExplain_train_test_split.json'
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, padding_side='right')
    dataset = GWExplanation(imageset_path=imageset_path,
                            ve_path=None,
                            knowledge_path=knowledge_path,
                            explanation_list_path=explanation_list_path,
                            date_round_knowledge_list_path=date_round_knowledge_list_path,
                            text_location_list_path=text_location_list_path,
                            train_test_split_path = train_test_split_path,
                            isTrain=True,
                            using_knowledge=1,
                            using_visual_clues=1
                            )
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor, device=device))
    test_dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=partial(evaluate_collate_fn, processor=processor, device=device))
    print(len(dataset))
    print(dataset[0])
    print("Start Checking Nan:")
    for step, batch in enumerate(train_dataloader):
        inputs = batch
        print(inputs)
        # if torch.isnan(inputs['pixel_values']).any() or torch.isinf(inputs['pixel_values']).any():
        #     print(f"NaN/Inf detected in pixel_values at step {step}")
        # if torch.isnan(inputs['input_ids']).any() or torch.isinf(inputs['input_ids']).any():
        #     print(f"NaN/Inf detected in input_ids at step {step}")
        # if torch.isnan(labels).any() or torch.isinf(labels).any():
        #     print(f"NaN/Inf detected in labels at step {step}")
        # if torch.isnan(inputs['attention_mask']).any() or torch.isinf(inputs['attention_mask']).any():
        #     print(f"NaN/Inf detected in attention_mask at step {step}")
        # if torch.isnan(inputs['image_grid_thw']).any() or torch.isinf(inputs['image_grid_thw']).any():
        #     print(f"NaN/Inf detected in image_grid_thw at step {step}")
        break