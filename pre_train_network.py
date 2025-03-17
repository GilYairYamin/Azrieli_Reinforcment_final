import os
import pandas as pd
import numpy as np
import torch
from abalone import TECHNIAL_MOVE_AMOUNT

from tqdm import tqdm


def decode_board_state(state_data):
    np_board_state = state_data["game_states"]
    board = np_board_state["board"]
    black_captures = np_board_state["black_captures"]
    white_captures = np_board_state["white_captures"]
    player = np_board_state["current_player"]

    children_move_Q = state_data["children_move_Q"]
    children_move_idx = state_data["children_move_idx"]
    value = state_data["final_status"]

    smallest_negative = np.finfo(np.float64).min
    policy_temp = np.full(
        TECHNIAL_MOVE_AMOUNT, smallest_negative, dtype=np.float64
    )
    for index in range(len(children_move_idx)):
        child_idx = children_move_idx[index]
        policy_temp[child_idx] = children_move_Q[index]

    policy_tensor = torch.from_numpy(policy_temp)
    softmax_tensor = policy_tensor.softmax(0)
    policy = softmax_tensor.numpy()

    board_state = (board, player, black_captures, white_captures)
    return board_state, policy, value


def build_training_data():
    root_folder = os.path.join(os.getcwd(), "local_data")
    data_folder = os.path.join(root_folder, "data for game training")
    file_list = os.listdir(data_folder)

    data_snippet = pd.DataFrame(columns=["board_state", "policy", "value"])

    idx = 0
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_folder, file_name)
        try:
            np_data: pd.DataFrame = pd.read_pickle(file_path)
        except Exception:
            continue
        
        curr_size = data_snippet.shape[0]
        for idx, row in np_data.iterrows():
            curr_idx = idx + curr_size
            board_state, policy, value = decode_board_state(row)
            data_snippet.loc[curr_idx, "board_state"] = board_state
            data_snippet.loc[curr_idx, "policy"] = policy
            data_snippet.loc[curr_idx, "value"] = value

        # if data_snippet.shape[0] > 50000:
        #     file_name = f"final_data_{idx}.pickle"
        #     file_path = os.path.join(root_folder, file_name)
        #     data_snippet.to_pickle(f"final_data_{idx}.pickle")
        #     idx += 1
        #     data_snippet = pd.DataFrame(
        #         columns=["board_state", "policy", "value"]
        #     )

    if data_snippet.shape[0] > 0:
        file_name = f"final_data_{idx}.pickle"
        file_path = os.path.join(root_folder, file_name)
        data_snippet.to_pickle(file_path)


build_training_data()
