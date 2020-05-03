# BP神经网络识别手写数字
### 开发环境
- 系统：macos
- ide：pycharm

```
.
├── README.md
├── config                  // 存放配置文件和数据集图片处理后数据文件
│   ├── config.txt          // 配置文件
│   ├── data_process.py     // 数据集图片处理脚本
│   ├── test.txt            // 测试集图片路径及标签
│   ├── test_dataset.txt    // 测试集矩阵
│   ├── train.txt           // 训练集图片路径及标签
│   └── train_dataset.txt   // 训练集矩阵
├── data
│   └── train_data.txt      // 保存训练完成后的权值矩阵
├── demo
│   ├── demo.py             // 训练并测试，保存权值矩阵
│   ├── detector_demo.py    // 读取保存的权值矩阵进行识别
│   └── detector_image.py   // 识别自己手写数字
├── image                   // 数据集
│   ├── num_test
│   │   ├── create_link_linux.sh    // linux版本创建测试集图片路径及标签脚本
│   │   └── create_link_mac.sh      // mac版本创建测试集图片路径及标签脚本
│   └── num_train
│       ├── create_link_linux.sh
│       └── create_link_mac.sh
├── mnist_dataset           // mnist数据集
│   ├── mnist_readme.txt
│   ├── mnist_test.csv
│   ├── mnist_test_10.csv
│   ├── mnist_train.csv
│   └── mnist_train_100.csv
├── net_work
    ├── __init__.py
    ├── config.py           // 读取配置文件参数
    ├── image_process.py    // 处理自己手写数字图像并识别
    └── net_work.py         // BP神经网络

```
### config路径配置
```
[datapath]
train_path = ../config/train.txt    // 训练图片路径
test_path = ../config/test.txt      // 测试图片路径
train_dataset_path = ../config/train_dataset.txt    // 训练标签特征值保存路径
test_dataset_path = ../config/test_dataset.txt      // 测试标签特征值保存路径
save_path = ../data/train_data.txt                  // 训练得出的权值矩阵保存路径
mnist_train_path = ../mnist_dataset/mnist_train.csv // mnist数据集
mnist_test_path = ../mnist_dataset/mnist_test.csv   // mnist数据集
```
### 相关库安装
```
pip3 install -r requirements.txt
```
### net_work API
```
- def process(self, train_path) // 训练，train_path：训练集路径
- def detector(self, test_path) // 测试，test_path：测试集路径
```
### 图片数据集处理
```
- 修改训练集和测试集路径
- cd config
- python3 data_process.py
- cd demo
- python demo.py        // 训练并识别测试集，保存权值矩阵
- python3 detector.py   // 读取图片识别
```
### minst数据集
```
- 修改训练集和测试集路径
- cd demo
- python demo.py            // 训练并识别测试集，保存权值矩阵
- python3 detector_image.py // 读取自己手写数字图片识别
```

