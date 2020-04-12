import  os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from net_work.net_work import NeuralNetWork

def main():
    network = NeuralNetWork("../config/config.txt")
    if not os.path.getsize(network.config.save_path):
        print("Please train the data...")
        return
    network.read_train_data()
    network.detector_image("../image/num_test/4/4_5.bmp")
    pass

if __name__ == '__main__':
    main()