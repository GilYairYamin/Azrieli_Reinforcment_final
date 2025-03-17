import os
import pandas as pd

from abalone_neural_network import AbaloneNetwork


def train_model():
    root_folder = os.path.join(os.getcwd(), "local_data")
    data_folder = os.path.join(root_folder, "result data")
    data_list = os.listdir(data_folder)

    model = AbaloneNetwork()
    model.train()

    for file in data_list:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_pickle(file_path)
        except Exception:
            print("couldn't load file")
            continue


if __name__ == "__main__":
    train_model()
