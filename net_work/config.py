import configparser

class Config:
    def __init__(self, config_path):
        cf = configparser.ConfigParser()
        cf.read(config_path)

        self.image_width = int(cf.get('image', 'image_width'))
        self.image_height = int(cf.get('image', 'image_height'))

        self.iterator = int(cf.get('network', 'iterator_times'))
        self.input_dim = int(cf.get('network', 'input_dim'))
        self.hidden_dim = int(cf.get('network', "hidden_dim"))
        self.output_dim = int(cf.get('network', 'output_dim'))
        self.learn_rate = float(cf.get('network', 'learn_rate'))

        self.train_dataset_path = cf.get('datapath', 'train_dataset_path')
        self.test_dataset_path = cf.get('datapath', 'test_dataset_path')
        self.train_path = cf.get('datapath', 'train_path')
        self.test_path = cf.get('datapath', 'test_path')
        self.save_path = cf.get('datapath', 'save_path')
        self.mnist_train_path = cf.get('datapath', 'mnist_train_path')
        self.mnist_test_path = cf.get('datapath', 'mnist_test_path')

def main():
    config = Config('../config/config.txt')
    pass

if __name__ == '__main__':
    main()