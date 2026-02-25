# 测试项目说明

本测试项目基于开源仓库：

https://github.com/LEFTeyex/U-DECN

## 环境配置

建议使用 Conda 环境：

```bash
conda activate oceancls

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml tqdm opencv-python tensorboard
```

## 离线测试（Smoke Test）

进入项目目录：
```bash
cd ./ODID_OceanDetectionwithImprovedDeNoising
```
运行测试示例：
```bash
python train.py \
--config configs/configs/default.yaml \
--fake-data \
--model simple_cnn \
--device cpu \
--epochs 1 \
--batch-size 16 \
--workers 0 \
--img-size 32 \
--exp-name smoke_test
```
该测试用于验证环境和程序是否能够正常运行，不依赖真实数据集。

# 数据集（Dataset）
## CIFAR-10 数据集下载
下载地址：  

https://www.cs.toronto.edu/~kriz/cifar.html  

下载完成后，将数据解压至以下目录：  
```bash
./weights/model.pt
```
