import numpy as np
import scipy.special
import imageio

import  os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from net_work.config import Config

class NeuralNetWork:
    # 初始化BP神经网络
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.inputnodes = self.config.input_dim
        self.hiddennodes = self.config.hidden_dim
        self.outputnodes = self.config.output_dim
        self.learnrate = self.config.learn_rate

        self.activation_function = lambda x: scipy.special.expit(x)

        self.wih = np.random.normal(0.0, pow(self.inputnodes, -0.5), (self.hiddennodes, self.inputnodes))
        self.who = np.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.outputnodes, self.hiddennodes))

    # 训练图片
    def train(self, train_data, target_data):
        inputs = np.array(train_data, ndmin=2).T
        targets = np.array(target_data, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.learnrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.wih += self.learnrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))
        pass

    # 训练数据集
    def process(self, train_path):
        print("trainning data...")
        file = open(train_path)
        num_data = file.readlines()
        for e in range(self.config.iterator):
            for num in num_data:
                all_values = num.split(',')
                train_data = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                target_data = np.zeros(self.config.output_dim) + 0.01
                target_data[int(all_values[0])] = 0.99
                self.train(train_data, target_data)
            print("迭代次数：", e)
        self.save_train_data()
        file.close()
        pass

    # 分类
    def query(self, test_data):
        inputs = np.array(test_data, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        label = np.argmax(final_outputs)
        return label
        pass

    # 识别测试集
    def detector(self, test_path):
        print("detectoring data...")
        file = open(test_path)
        num_data = file.readlines()
        scorecard = []
        for num in num_data:
            all_values = num.split(',')
            test_data = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            label = int(all_values[0])
            detector_label = self.query(test_data)
            print("识别结果：" + str(detector_label)+ "    测试标签：" + str(label))
            if label == detector_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
        scorecard_array = np.asarray(scorecard)
        print("准确率: ", scorecard_array.sum() / scorecard_array.size)
        file.close()
        pass

    # 识别单个图片
    def detector_image(self, image_path):
        print("测试图片：", image_path)
        image = imageio.imread(image_path, as_gray = True)
        image_data = image.reshape(self.config.image_height * self.config.image_width)
        detector_label = self.query(image_data)
        print("识别结果：", detector_label)
        return detector_label

    # 保存训练出的权值矩阵
    def save_train_data(self):
        print("Saving data...")
        file = open(self.config.save_path, 'w')
        file .write("wih : ")
        for line in self.wih:
            for data in line:
                file.write(str(data) + ' ')
            file.write(',')
        file.write('\nwho : ')
        for line in self.who:
            for data in line:
                file.write(str(data) + ' ')
            file.write(',')
        file.close()
        pass

    # 读取权值矩阵
    def read_train_data(self):
        print("Reading data...")
        file = open(self.config.save_path)
        train_data = file.readlines()
        for data in train_data:
            data_list = []
            all_values = data.strip(' ').split(':')
            if all_values[0].strip(' ') == 'wih':
                data_row = all_values[1].strip('\n').split(',')
                for row in data_row:
                    data = row.strip(' ').split(' ')
                    if(len(data) <= 1):
                        continue
                    data_list.append(data)
                self.wih = np.asfarray(data_list)
            elif all_values[0].strip(' ') == 'who':
                data_row = all_values[1].strip('\n').split(',')
                for row in data_row:
                    data = row.strip(' ').split(' ')
                    if (len(data) <= 1):
                        continue
                    data_list.append(data)
                self.who = np.asfarray(data_list)
        pass

    def predict(self):
        self.read_train_data()
        

    # 数字图像尺寸标准化（28*28）
    def n_resize(self):
        pass