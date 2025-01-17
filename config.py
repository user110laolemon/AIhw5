import torch


class Config:
    def __init__(self):
        self.do_train = True
        self.do_test = True

        self.train_txt_path = './P5data/train.txt'
        self.test_txt_path = './P5data/test_without_label.txt'
        self.data_path = './P5data/data/'

        self.train_output_file = './train_result.txt'
        self.test_output_file = './test_result.txt'

        self.train_file = './P5data/train.json'
        self.dev_file = './P5data/dev.json'
        self.test_file = './P5data/test.json'
        self.dev_size = 0.2
        
        self.pretrained_model = 'roberta-base'
        self.lr = 1e-6
        self.dropout = 0.1
        self.epochs = 10
        self.batch_size = 4
        self.img_size = 384
        self.text_size = 64


        self.mode = 'train'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_dict_str = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label_dict_number = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.save_model_path = './model.pth'
        self.load_model_path = './model.pth'
        self.train_output_file = './train_result.txt'
        self.dev_output_file = './dev_result.txt'
        self.img_path = './P5data/images'
        