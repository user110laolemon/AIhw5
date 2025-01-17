# 多模态融合模型
第五次实验：
设计一个多模态融合模型。 自行从训练集中划分验证集，调整超参数。 预测测试集（test_without_label.txt）上的情感标签。


## Setup
 
在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository Structure 

本仓库的文件结构如下：

```
|-- model.py # 本次实验模型代码
|-- dataprocess.py # 本次实验数据处理代码
|-- config.py # 本次实验超参、配置代码
|-- train_evaluate.py # 本次实验训练、评估代码
|-- README.md   # 对仓库的解释
|-- requirements.txt    # 本次实验的环境
|-- 10214602404_李芳_实验五.pdf # 实验报告
|-- test_result.csv  # 预测结果
|-- P5data    # 实验数据集
```


## Usage

在终端中输入

```shell
python dataprocess.py
python train_evaluate.py
```
实验的所有内容都需要放于同一目录下，运行时间比较长，建议使用云GPU服务器！

注：ROBERTA模型需要下载Huggingface中的资源，注意联网需求。


## Reference

[1] https://github.com/RecklessRonan/GloGNN/blob/master/readme.md

[2] https://github.com/zdou0830/METER

