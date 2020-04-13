import  os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from net_work.net_work import NeuralNetWork

def main():
    network = NeuralNetWork("../config/config.txt")
    network.process(network.config.mnist_train_path)
    network.detector(network.config.mnist_test_path)
    pass

if __name__ == '__main__':
    main()