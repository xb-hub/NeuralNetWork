# BP神经网络识别手写数字
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
│   ├── demo.py             // 训练并识别
│   └── detector_demo.py    // 读取保存的权值矩阵进行识别
├── image                   // 数据集
│   ├── num_test
│   │   ├── create_link_linux.sh
│   │   └── create_link_mac.sh
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
    └── net_work.py         // BP神经网络

```
## 运行步骤
```
- cd demo
- python demo.py    // 训练并识别
- python3 detector.py // 读取训练数据识别
```

