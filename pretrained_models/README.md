我们在以下DNN模型上评估了以前的覆盖标准和NLC


| Model        | Training dataset | Remark                       |
| ------------ | ---------------- | ---------------------------- |
| ResNet50     | CIFAR10          | Image & Discriminative model |
| ResNet50     | ImageNet         | Image & Discriminative model |
| VGG16_bn     | CIFAR10          | Image & Discriminative model |
| VGG16_bn     | ImageNet         | Image & Discriminative model |
| MobileNet_V2 | CIFAR10          | Image & Discriminative model |
| MobileNet_V2 | ImageNet         | Image & Discriminative model |
| BigGAN       | CIFAR10          | Image & Generative model     |
| BigGAN       | ImageNet         | Image & Generative model     |
| LSTM         | IMDB             | Text & Discriminative model  |

它们的实现见 `models`.

下载预训练权重 [此处](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/EhO-hLQ6SRVItt-ZBkrD-8YBAZTqGAdxOsnMOvHIXeKS9A?e=DjdDsK).

