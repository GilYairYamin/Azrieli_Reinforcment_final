# from abalone import Abalone
import os
import pandas as pd
import torch

from abalone_neural_network import (
    AbaloneNetwork,
    convert_encoded_board_to_tensors,
)

if __name__ == "__main__":
    # game = Abalone(True)
    # board_state = game.encode()
    # board, extra = convert_encoded_board_to_tensors(board_state)
    # print(board.shape, extra)

    # root_folder = os.path.join(os.getcwd(), "local_data")
    # data_folder = os.path.join(root_folder, "result data")

    # file_list = os.listdir(data_folder)
    # file_name = file_list[0]
    # file_path = os.path.join(data_folder, file_name)

    # df = pd.read_pickle(file_path)
    # board_state = df.iloc[0]["board_state"]
    # legal_move_mask = df.iloc[0]["legal_move_mask"]
    # board, extra, legal_move_tensor = convert_encoded_board_to_tensors(
    #     board_state, legal_move_mask
    # )

    # print(board)
    # print(extra)
    # print(legal_move_tensor)
