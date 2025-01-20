import torch


class Config:
    def __init__(self):
        # 训练模式or预测模式
        self.do_train = True
        self.do_test = False
        # CPU or GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 调整这些超参数
        self.lr = 1e-8
        self.dropout = 0.1
        self.epochs = 300
        self.batch_size = 4
        self.img_size = 384
        self.text_size = 64
        self.patience = 3
        # 数据路径
        self.train_txt_path = './P5data/train.txt'
        self.test_txt_path = './P5data/test_without_label.txt'
        self.data_path = './P5data/data/'
        # 转换后数据集路径
        self.train_file = './P5data/train.json'
        self.dev_file = './P5data/dev.json'
        self.test_file = './P5data/test.json'
        # 
        self.dev_size = 0.2
        self.label_dict_str = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label_dict_number = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.pretrained_model = 'roberta-base'
        # 使用数据集，选择train or test
        self.mode = 'train'
        self.test_output_file = './test_results'
        self.model_path = './pretrained_models'
        self.plt_path = './loss_curves'