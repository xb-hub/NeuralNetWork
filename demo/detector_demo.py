from net_work.net_work import NeuralNetWork

def main():
    network = NeuralNetWork()
    network.read_train_data()
    network.detector()
    pass

if __name__ == '__main__':
    main()