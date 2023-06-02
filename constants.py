PAD_LENGTH = 32
# 为文本模型展开层

PRETRAINED_MODELS = './pretrained_models'

AE_DIR = './adversarial_examples/'

CIFAR10_JPEG_DIR = './datasets/CIFAR10/'
IMAGENET_JPEG_DIR = '/data/yyuanaq/data/IMAGE-NET/ILSVRC/Data/CLS-LOC/'
IMAGENET_LABEL_TO_INDEX = './datasets/ImageNet/ImageNetLabel2Index.json'
# 我们使用的是pytorch提供的预训练权重,
# 我们应当使用相同的 `label_to_index` 映射.

BIGGAN_IMAGENET_PROJECT_DIR = './BigGAN-projects/ImageNet'
BIGGAN_CIFAR10_PROJECT_DIR = './BigGAN-projects/CIFAR10'

BIGGAN_CIFAR10_LATENT_DIM = 128
BIGGAN_IMAGENET_LATENT_DIM = 120

STYLE_IMAGE_DIR = './datasets/painting'
STYLE_MODEL_DIR = './pretrained_models/Style'