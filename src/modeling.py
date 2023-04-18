from torch import Tensor
from typing import Optional, Tuple

import numpy as np
import math

from dataclasses import dataclass
import torch 
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, BertForPreTraining , BertConfig , Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Config, Wav2Vec2Model , BertModel

from datasets import load_dataset
import logging


logging.basicConfig(
    filename='logs.log',
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


 


class CustomDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "patrickvonplaten/librispeech_asr_dummy",
        feature_extractor_name: str = "facebook/wav2vec2-base-960h",
        dataset_split: str = "validation",
        dataset_subset: str = "clean",
    ):
        self.dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            feature_extractor_name
        )
        self.dataset = self.dataset.remove_columns(
            ["file","speaker_id", "chapter_id", "id"]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        audio = item["audio"]["array"]
        return text, audio



class CustomDataLoader(DataLoader):
    def __init__(
        self,
        dataset_name: str = "patrickvonplaten/librispeech_asr_dummy",
        feature_extractor_name: str = "facebook/wav2vec2-base-960h",
        dataset_split: str = "validation",
        dataset_subset: str = "clean",
        batch_size: int = 4,
        num_workers: int = 4,
        tokenizer_name :str = "bert-base-uncased"
    ):
        self.dataset = CustomDataset(
            dataset_name, feature_extractor_name, dataset_split, dataset_subset
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            feature_extractor_name
        )
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def collate_fn(self, batch):
        text, audio = zip(*batch)
        input_values = self.feature_extractor(
            audio, sampling_rate=16_000 , padding=True, return_tensors="pt" 
        ).input_values

        text = self.tokenizer(text, padding=True, return_tensors="pt", truncation=True)

        return {
            "text_input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
            "audio_input_values": input_values,
            "audio_attention_mask": input_values.ne(0).float(),

        }
    


class BertTeacherModel(nn.Module):
    def __init__(self , model_name ):
        super(BertTeacherModel,self).__init__()
        self.model = BertModel.from_pretrained(model_name)
 

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

class Wav2Vec2StudentModel(nn.Module):
    def __init__(self ,  config : Wav2Vec2Config):
        super(Wav2Vec2StudentModel,self).__init__()
        self.model = Wav2Vec2Model(config)

    def forward(self, input_values, attention_mask):
        return self.model(input_values, attention_mask)
    


class Model(nn.Module):
    def __init__(self,  model_name="bert-base-uncased" , config = Wav2Vec2Config()):
        super(Model ,self).__init__()
        self.bert_teacher_model = BertTeacherModel(model_name)
        self.wav2vec2_student_model = Wav2Vec2StudentModel(config)
        # set the output of the model [batch_size , seq_len , hidden_size] of the teacher model to the input of the student model



    def forward(self, text_input_ids, text_attention_mask, audio_input_values, audio_attention_mask):
        teacher_output = self.bert_teacher_model(text_input_ids, text_attention_mask).last_hidden_state
        student_output = self.wav2vec2_student_model(audio_input_values, audio_attention_mask).last_hidden_state
       
        return teacher_output, student_output
    



if __name__ == "__main__":
    data_loader = CustomDataLoader(batch_size=8)
    # bert_teacher_model = BertTeacherModel("bert-base-uncased")
    # wav2vec2_student_model = Wav2Vec2StudentModel(Wav2Vec2Config())
    model = Model()
    for batch in data_loader:
        teacher_output, student_output = model(**batch)  
        print(f"teacher_output: {teacher_output.shape}")
        print(f"student_output: {student_output.shape}")
        break



    