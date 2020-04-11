import numpy as np
import scipy.special
from config import Config

class NeuralNetWork:
    def __init__(self):
        self.config = Config('../config/config.txt')
        self.inputnodes = self.config.input_dim
        self.hiddennodes = self.config.hidden_dim
        self.outputnodes = self.config.output_dim
        self.learnrate = self.config.learn_rate

        self.activation_function = lambda x: scipy.special.expit(x)

        self.wih = np.random.normal(0.0, pow(self.inputnodes, -0.5), (self.hiddennodes, self.inputnodes))
        self.who = np.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.outputnodes, self.hiddennodes))

    def train(self, train_data, target_data):
        inputs = np.array(train_data, ndmin=2).T
        targets = np.array(target_data, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs);
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

    def process(self):
        print("trainning data...")
        file = open(self.config.train_dataset_path)
        num_data = file.readlines()
        for e in range(self.config.iterator):
            for num in num_data:
                all_values = num.split(',')
                train_data = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                target_data = np.zeros(self.config.output_dim) + 0.01
                target_data[int(all_values[0])] = 0.99
                self.train(train_data, target_data)
            print("迭代次数：" + str(e) + '\n')
        self.save_train_data()
        file.close()
        pass

    def query(self, test_data):
        inputs = np.array(test_data, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        label = np.argmax(final_outputs)
        return label
        pass

    def detector(self):
        print("detectoring data...")
        file = open(self.config.test_dataset_path)
        num_data = file.readlines()
        scorecard = []
        for num in num_data:
            all_values = num.split(',')
            test_data = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            label = int(all_values[0])
            detectoe_label = self.query(test_data)
            print("识别结果：" + str(detectoe_label)+ "    测试标签：" + str(label) + '\n')
            if label == detectoe_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
        scorecard_array = np.asarray(scorecard)
        print("准确率: ", scorecard_array.sum() / scorecard_array.size)
        file.close()
        pass

    def save_train_data(self):
        print("Saving data...")
        file = open(self.config.save_path, 'w')
        file .write("wih : ");
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
        pass