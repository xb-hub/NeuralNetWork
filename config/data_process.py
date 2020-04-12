import numpy as np
import imageio
from net_work.config import Config

class DataProcess:
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.train_path = self.config.train_path
        self.test_path = self.config.test_path
        pass

    def read_train_image(self):
        file = open(self.train_path)
        save = open(self.config.train_dataset_path, 'w')
        data_list = file.readlines()
        for data in data_list:
            all_value = data.split(' ')
            image = imageio.imread(all_value[0], as_gray=True)
            image_data = image.reshape(self.config.image_height * self.config.image_width)
            save.write(all_value[1][0])
            for value in image_data:
                save.write(',' + str(int(value)))
            save.write('\n')
        file.close()
        save.close()
        pass

    def read_test_image(self):
        file = open(self.test_path)
        save = open(self.config.test_dataset_path, 'w')
        data_list = file.readlines()
        for data in data_list:
            all_value = data.split(' ')
            image = imageio.imread(all_value[0], as_gray=True)
            image_data = image.reshape(self.config.image_height * self.config.image_width)
            save.write(all_value[1][0])
            for value in image_data:
                save.write(',' + str(int(value)))
            save.write('\n')
        file.close()
        save.close()
        pass

def main():
    data_process = DataProcess("../config/config.txt")
    data_process.read_train_image()
    data_process.read_test_image()
    pass

if __name__ == '__main__':
    main()