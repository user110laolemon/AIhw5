import torch


class Config:
    def __init__(self):
        self.do_train = True
        self.do_test = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = 1e-6
        self.dropout = 0.1
        self.epochs = 10
        self.batch_size = 4
        self.img_size = 384
        self.text_size = 64
        self.patience = 3

        self.train_txt_path = './P5data/train.txt'
        self.test_txt_path = './P5data/test_without_label.txt'
        self.data_path = './P5data/data/'

        self.train_file = './P5data/train.json'
        self.dev_file = './P5data/dev.json'
        self.test_file = './P5data/test.json'
        self.dev_size = 0.2
        self.label_dict_str = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label_dict_number = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.pretrained_model = 'roberta-base'

        self.mode = 'train'
        self.test_output_file = f'./test_result_{self.lr}_{self.dropout}'
        self.save_model_path = f'./model_{self.lr}_{self.dropout}.pth'
        self.load_model_path = f'./model_{self.lr}_{self.dropout}.pth'
        self.save_plt_path = f'./loss_{self.lr}_{self.dropout}.png'