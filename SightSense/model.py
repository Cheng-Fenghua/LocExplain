import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

# encoder unit
class Qwen2ForGuessWhere(nn.Module):
    def __init__(self, qwen2_base):
        super(Qwen2ForGuessWhere, self).__init__()
        self.qwen2_base

    def forward(self, query, image, image_details, knowledge_list, ):
