import  os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")
from net_work.net_work import NeuralNetWork
from net_work.image_process import ImageProcess

def main():
    image_process = ImageProcess()
    network = NeuralNetWork("../config/config.txt")
    network.read_train_data()
    image_process.process(network)
    pass

if __name__ == "__main__":
    main()