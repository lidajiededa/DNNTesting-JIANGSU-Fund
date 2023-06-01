## 实现

我们复现并实现了以下的神经元覆盖率指标，包括  

- [x] Neuron Coverage (**NC**) 
- [x] K-Multisection Neuron Coverage (**KMNC**) 
- [x] Neuron Boundary Coverage (**NBC**) 
- [x] Strong Neuron Activation Coverage (**SNAC**) 
- [x] Top-K Neuron Coverage (**TKNC**) 
- [x] Top-K Neuron Patterns (**TKNP**) 
- [x] Cluster-based Coverage (**CC**) 
- [x] Likelihood Surprise Coverage (**LSC**) 
- [x] Distance-ratio Surprise Coverage (**DSC**) 
- [x]  Mahalanobis Distance Surprise Coverage (**MDSC**) 


## 安装

- 根据源代码安装

    ```setup
    git clone https://github.com/lidajiededa/DNNTesting-JIANGSU-Fund
    cd NeuraL-Coverage
    pip install -r requirements.txt
    ```

## 模型和数据集

- 预训练模型: 请参考 [模型](https://github.com/lidajiededa/DNNTesting-JIANGSU-Fund/pretrained_models).
- 数据集: 请参考 [数据集](https://github.com/lidajiededa/DNNTesting-JIANGSU-Fund/datasets).

下载 `pretrained_models`, `datasets`以及 `adversarial_examples` 文件夹 [此处](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/EhO-hLQ6SRVItt-ZBkrD-8YBAZTqGAdxOsnMOvHIXeKS9A?e=DjdDsK).

## 如何开始使用

```python
import torch
# Implemented using Pytorch

import tool
import coverage

# 0. Get layer size in model
input_size = (1, image_channel, image_size, image_size)
random_input = torch.randn(input_size).to(device)
layer_size_dict = tool.get_layer_output_sizes(model, random_input)

# 1. Initialization
# `hyper` denotes the hyper-paramter of a criterion;
# set `hyper` as None if a criterion is hyper-paramter free (e.g., NLC).
criterion = coverage.NLC(model, layer_size_dict, hyper=None)
# KMNC/NBC/SNAC/LSC/DSC/MDSC requires training data statistics of the tested model,
# which is implemented in `build`. `train_loader` can be a DataLoader object in Pytorch or a list of data samples.
# For other criteria, `build` function is empty.
criterion.build(train_loader)

# 2. Calculation
# `test_loader` stores all test inputs; it can be a DataLoader object in Pytorch or a list of data samples.
criterion.assess(test_loader)
# If test inputs are gradually given from a data stream (e.g., in fuzzing), then calculate the coverage as the following way.
for data in data_stream:
    criterion.step(data)

# 3. Result
# The following instruction assigns the current coverage value to `cov`.
cov = criterion.current
```

## 实验

在准备玩模型和数据集后, 首先在`constants.py`中设置好路径 .  

### 测试套件的多样性

#### 判别式（图像）模型

```bash
python eval_diversity_image.py --model resnet50 --dataset CIFAR10 --criterion NC --hyper 0.75
```

- `--model` - 被测试的DNN 
chocies = [`resnet50`, `vgg16_bn`, `mobilenet_v2`]

- `--dataset` - 被测试的DNN的训练集，测试套件是使用此数据集的测试拆分生成的
choices = [`CIFAR10`, `ImageNet`]

- `--criterion` - 使用的覆盖率指标.  
choices = [`NC`, `KMNC`, `NBC`, `SNAC`, `TKNC`, `TKNP`, `CC`, `LSC`, `DSC`, `MDSC`, `NLC`]

- `--hyper` - 所用指标的超参数。 `None` 如果指标没有超参数 (也就是NLC, SNAC, NBC)。

#### 生成式(文本)模型

```bash
python eval_diversity_text.py --criterion NC --hyper 0.75
```

- `--criterion` - 使用的覆盖率指标  
choices = [`NC`, `KMNC`, `NBC`, `SNAC`, `TKNC`, `TKNP`, `CC`, `LSC`, `DSC`, `MDSC`, `NLC`]

- `--hyper` - 所用指标的超参数。 `None` 如果指标没有超参数 (也就是NLC, SNAC, NBC)。



### 测试套件的故障揭示能力

```bash
python eval_fault_revealing.py --dataset CIFAR10 --model resnet50 --criterion NC --hyper 0.75 --AE PGD --split test
```

- `--AE` - AE生成算法.  
choices = [`PGD`,  `CW`]

- `--split` - 生成AE所使用的部分数据集.  
choices = [`train`, `test`]


### DNN测试中的指导输入变异

```bash
python fuzz.py --dataset CIFAR10 --model resnet50 --criterion NC
```

对于随机的变异（也就是没有任何客观标准的），运行

```bash
python fuzz_rand.py --dataset CIFAR10 --model resnet50
```
