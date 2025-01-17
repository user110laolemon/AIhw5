import json
import math
import chardet
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
from PIL import Image

from config import *
config = Config()

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model)
        self.label_dict_number = config.label_dict_number
        self.label_dict_str = config.label_dict_str

    def __getitem__(self, index):
        return self.tokenize(self.data[index])

    def __len__(self):
        return len(self.data)

    def tokenize(self, item):
        item_id = item['id']
        text = item['text']
        image_path = item['image_path']
        label = item['label']

        text_token = self.tokenizer(text, return_tensors="pt", max_length=self.config.text_size,
                                    padding='max_length', truncation=True)
        text_token['input_ids'] = text_token['input_ids'].squeeze()
        text_token['attention_mask'] = text_token['attention_mask'].squeeze()

        img_token = self.transform(image_path) if self.transform else torch.tensor(image_path)

        label_token = self.label_dict_number[label] if label in self.label_dict_number else -1
        return item_id, text_token, img_token, label_token


def load_json(file):
    data_list = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for line in lines:
            item = {
                'image_path': np.array(Image.open(line['image_path'])),
                'text': line['text'],
                'label': line['label'],
                'id': line['guid']
            }
            data_list.append(item)
    return data_list


def load_data(config):
    img_size = (config.img_size, config.img_size)
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(img_size),
         transforms.Normalize([0.5], [0.5])]
    )
    data_list = {
        'train': load_json(config.train_file),
        'dev': load_json(config.dev_file),
        'test': config.test_file and load_json(config.test_file),
    }
    data_set = {
        'train': MyDataset(data_list['train'], transform=data_transform),
        'dev': MyDataset(data_list['dev'], transform=data_transform),
        'test': config.test_file and MyDataset(data_list['test'], transform=data_transform),
    }

    return data_set[config.mode], data_set['dev']


def save_data(file, predict_list):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for pred in predict_list:
            f.write(f"{pred['guid']},{pred['tag']}\n")


def read_text_file(file, encoding):
    text = ''
    with open(file, encoding=encoding) as fp:
        for line in fp.readlines():
            line = line.strip('\n')
            text += line
    return text


def transform_data(data_values, data_path):
    dataset = []
    for i in range(len(data_values)):
        guid = str(int(data_values[i][0]))
        label = data_values[i][1]
        if type(label) != str and math.isnan(label):
            label = None

        file_path = data_path + guid + '.txt'
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            if encoding == "GB2312":
                encoding = "GBK"

        text = ''
        try:
            text = read_text_file(file_path, encoding)
        except UnicodeDecodeError:
            try:
                text = read_text_file(file_path, 'ANSI')
            except UnicodeDecodeError:
                print('UnicodeDecodeError')
        dataset.append({
            'guid': guid,
            'text': text,
            'label': label,
            'image_path': data_path + guid + '.jpg',
        })
    return dataset


if __name__ == '__main__':
    train_dev_df = pd.read_csv(config.train_txt_path)
    test_df = pd.read_csv(config.test_txt_path)
    train_df, dev_df = train_test_split(train_dev_df, test_size=config.dev_size)

    train_data = transform_data(train_df.values, config.data_path)
    dev_data = transform_data(dev_df.values, config.data_path)
    test_data = transform_data(test_df.values, config.data_path)

    with open(config.train_file, 'w', encoding="utf-8") as f:
        json.dump(train_data, f)

    with open(config.dev_file, 'w', encoding="utf-8") as f:
        json.dump(dev_data, f)

    with open(config.test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)