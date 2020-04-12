import os
from net_work.net_work import NeuralNetWork

def main():
    network = NeuralNetWork("../config/config.txt")
    if not os.path.getsize(network.config.save_path):
        print("Please train the data...")
        return
    network.read_train_data()
    network.detector()
    pass

if __name__ == '__main__':
    main()