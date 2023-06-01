在我们的评估中使用了以下的数据集
- ImageNet

  可以从此处下载 [official release](https://www.image-net.org/download.php) 或者 [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)。

- CIFAR10:

  可以从此处下载 [official release](https://www.cs.toronto.edu/~kriz/cifar.html).

  请注意，在我们的评估中，我们首先将CIFAR-10数据转换为图像，然后将这些图像用作测试套件。我们在此处提供了转换后的图像 [here]()。

- IMDB:

  此数据集将在运行 `./data` 文件夹里的 `eval_diversity_text.py` 后自动下载。