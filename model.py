import torch
from torch import nn
from torchvision.models import convnext
from transformers import RobertaModel

from dataprocess import *
from config import *
config = Config()

class TEXTRobertaModel(nn.Module):
    '''文本模型'''
    def __init__(self, args):
        '''初始化文本模型
            设置RoBERTa模型和全连接层
        '''
        super(TEXTRobertaModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.pretrained_model)
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.transform = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
        )

    def forward(self, encoded_input):
        '''前向传播'''
        encoder_output = self.encoder(**encoded_input)
        hidden_state = encoder_output['last_hidden_state']
        pooler_output = encoder_output['pooler_output']
        output = self.transform(pooler_output)
        return hidden_state, output


class IMGConvnextModel(nn.Module):
    '''图像模型'''
    def __init__(self, args):
        '''初始化图像模型
            设置ConvNeXt模型
        '''
        super(IMGConvnextModel, self).__init__()
        self.encoder = convnext.convnext_base(weights=convnext.ConvNeXt_Base_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        '''前向传播'''
        x = self.encoder(x)
        return x


class MultiModalModel(nn.Module):
    '''多模态模型'''
    def __init__(self, args):
        '''初始化多模态模型
            设置文本模型、图像模型、多头注意力机制、Transformer编码器和分类器
        '''
        super(MultiModalModel, self).__init__()
        self.TextModule_ = TEXTRobertaModel(args)
        self.ImgModule_ = IMGConvnextModel(args)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=1000, num_heads=2, batch_first=True)
        self.linear_text_k1 = nn.Linear(1000, 1000)
        self.linear_text_v1 = nn.Linear(1000, 1000)
        self.linear_img_k2 = nn.Linear(1000, 1000)
        self.linear_img_v2 = nn.Linear(1000, 1000)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.classifier_img = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        self.classifier_text = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )

    def forward(self, text=None, image=None):
        '''前向传播
            判断输入类型，转发文本和图像输出
            返回文本、图像或多模态的输出
        '''
        if text is not None and image is None:
            _, text_out = self.TextModule_(text)
            text_out = self.classifier_text(text_out)
            return text_out

        if text is None and image is not None:
            img_out = self.ImgModule_(image)
            img_out = self.classifier_img(img_out)
            return img_out

        _, text_out = self.TextModule_(text)
        img_out = self.ImgModule_(image)
        multi_out = torch.cat((text_out, img_out), 1)
        text_out = self.classifier_text(text_out)
        img_out = self.classifier_img(img_out)
        multi_out = self.classifier_multi(multi_out)
        return text_out if text is not None else img_out if image is not None else multi_out


    def Multihead_self_attention(self, text_out, img_out):
        '''多头自注意力机制
            转发文本和图像输出
            返回注意力输出
        '''
        text_k1 = self.linear_text_k1(text_out)
        text_v1 = self.linear_text_v1(text_out)
        img_k2 = self.linear_img_k2(img_out)
        img_v2 = self.linear_img_v2(img_out)
        key = torch.stack((text_k1, img_k2), dim=1)
        value = torch.stack((text_v1, img_v2), dim=1)
        query = torch.stack((text_out, img_out), dim=1)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output

    def Transformer_Encoder(self, text_out, img_out):
        '''Transformer编码器
            转发文本和图像输出
            返回编码输出
        '''
        multimodal_sequence = torch.stack((text_out, img_out), dim=1)
        return self.transformer_encoder(multimodal_sequence)