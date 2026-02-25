本测试项目基于：https://github.com/LEFTeyex/U-DECN

conda activate oceancls  <cr>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 <cr>
pip install pyyaml tqdm opencv-python tensorboard <cr>

离线测试:
cd ./ODID_OceanDetectionwithImprovedDeNoising <cr>
python train.py --config configs/configs/default.yaml --fake-data --model simple_cnn --device cpu --epochs 1 --batch-size 16 --workers 0 --img-size 32 --exp-name smoke_test <cr>

## Dataset 数据集

下载 CIFAR-10:

https://www.cs.toronto.edu/~kriz/cifar.html

于下方路径解压:

./data/cifar-10-batches-py/

或 联网状态下：

cd ./ODID_OceanDetectionwithImprovedDeNoising
python train.py --config configs/configs/default.yaml --dataset cifar10 --model resnet18 --epochs 5 --batch-size 32 --workers 2 --exp-name cifar10_resnet18


## 训练PT
下载地址：
链接：https://pan.quark.cn/s/854319d63447
提取码：fh8M

于路径下使用：
./weights/model.pt

