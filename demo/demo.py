from net_work.net_work import NeuralNetWork

def main():
    network = NeuralNetWork("../config/config.txt")
    network.process()
    network.detector()
    pass

if __name__ == '__main__':
    main()