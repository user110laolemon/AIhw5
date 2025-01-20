# 多模态融合模型
这是第五次实验的仓库，设计一个多模态融合模型：自行从训练集中划分验证集，预测测试集上的情感标签。


## Setup

该实验基于Python3.8。要运行代码，你需要以下依赖项：

&#9726;chardet==5.2.0\
&#9726;charset-normalizer==3.3.2\
&#9726;jsonpatch==1.33\
&#9726;jsonpointer==3.0.0\
&#9726;jsonschema==4.23.0\
&#9726;jsonschema-specifications==2023.12.1\
&#9726;matplotlib==3.9.0\
&#9726;matplotlib-inline==0.1.7\
&#9726;numpy==1.26.4\
&#9726;pandas==2.2.2\
&#9726;scikit-learn==1.5.2\
&#9726;torch==2.5.1\
&#9726;torchdata==0.7.1\
&#9726;torchtext==0.16.2\
&#9726;torchvision==0.20.1\
&#9726;transformers==4.47.0

在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository Structure 

本仓库的文件结构如下：

```
|-- model.py    # 本次实验模型代码
|-- dataprocess.py  # 本次实验数据处理代码
|-- config.py   # 本次实验超参、配置代码
|-- train_evaluate.py   # 本次实验训练、评估代码
|-- README.md   # 对仓库的解释
|-- requirements.txt    # 本次实验所需环境配置
|-- 10214602404_李芳_实验五.pdf # 实验报告
|-- test_result.txt # 预测结果
|-- P5data/  # 实验数据集文件夹
|-- loss_curves/ # 训练loss曲线图文件夹
|-- pretrained_models/   # 保存的训练好的模型文件夹
|-- test_results/    # 所有get_test的结果文件夹
```


## Usage

进入项目目录并运行以下命令，首先进行数据预处理：

```shell
python dataprocess.py
```

然后运行下面语句进行模型训练与预测：

```shell
python train_evaluate.py
```

**注：**实验的所有代码以及数据、新建文件夹都需要放于同一目录下，运行时间比较长，建议使用云GPU服务器！另：ROBERTA模型需要下载Huggingface中的资源，注意联网需求。


## Reference

[1] https://github.com/RecklessRonan/GloGNN/blob/master/readme.md

[2] https://github.com/zdou0830/METER

## Attribution

部分代码基于以下仓库：

&#9726;Hugging Face Transformers\
&#9726;PyTorch