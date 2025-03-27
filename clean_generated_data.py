import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from abalone import TECHNIAL_MOVE_AMOUNT


def decode_board_state(state_data):
    np_board_state = state_data["game_states"]
    board = np_board_state["board"]
    black_captures = np_board_state["black_captures"]
    white_captures = np_board_state["white_captures"]
    player = np_board_state["current_player"]

    children_move_N = state_data["children_move_N"]
    children_move_idx = state_data["children_move_idx"]
    value = state_data["final_status"]

    smallest_negative = np.finfo(np.float64).min
    policy_temp = np.full(
        TECHNIAL_MOVE_AMOUNT, smallest_negative, dtype=np.float64
    )
    policy_temp = np.full(
        TECHNIAL_MOVE_AMOUNT, smallest_negative, dtype=np.float64
    )

    legal_move_mask = np.zeros(TECHNIAL_MOVE_AMOUNT, dtype=np.int8)

    for index in range(len(children_move_idx)):
        child_idx = children_move_idx[index]
        policy_temp[child_idx] = children_move_N[index]
        legal_move_mask[child_idx] = 1

    policy_tensor = torch.from_numpy(policy_temp)
    softmax_tensor = policy_tensor.softmax(0)
    policy = softmax_tensor.numpy()

    board_state = (board, black_captures, white_captures, player)

    return board_state, legal_move_mask, policy, value


def build_training_data_from_mcts():
    root_folder = os.path.join(os.getcwd(), "local_data")
    data_folder = os.path.join(root_folder, "data")
    res_data_folder = os.path.join(root_folder, "training data")
    os.makedirs(res_data_folder, exist_ok=True)
    file_list = os.listdir(data_folder)

    columns = ["board_state", "legal_move_mask", "policy", "value"]
    data_snippet = pd.DataFrame(columns=columns)

    name_idx = 0
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_folder, file_name)
        try:
            np_data: pd.DataFrame = pd.read_pickle(file_path)
        except Exception:
            continue

        curr_size = data_snippet.shape[0]
        for idx, row in np_data.iterrows():
            curr_idx = idx + curr_size
            board_state, legal_move_mask, policy, value = decode_board_state(
                row
            )
            data_snippet.loc[curr_idx, "board_state"] = board_state
            data_snippet.loc[curr_idx, "legal_move_mask"] = legal_move_mask
            data_snippet.loc[curr_idx, "policy"] = policy
            data_snippet.loc[curr_idx, "value"] = value

        if data_snippet.shape[0] > 20000:
            file_name = f"final_data_{name_idx}.pickle"
            file_path = os.path.join(res_data_folder, file_name)
            data_snippet.to_pickle(file_path)
            name_idx += 1
            data_snippet = pd.DataFrame(columns=columns)

    if data_snippet.shape[0] > 0:
        file_name = f"final_data_{name_idx}.pickle"
        file_path = os.path.join(res_data_folder, file_name)
        data_snippet.to_pickle(file_path)


def build_training_data_from_puct(data_folder, res_fle_name):
    file_list = os.listdir(data_folder)

    columns = ["board_state", "legal_move_mask", "policy", "value"]
    data_snippet = pd.DataFrame(columns=columns)

    name_idx = 0
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_folder, file_name)
        try:
            np_data: pd.DataFrame = pd.read_pickle(file_path)
        except Exception:
            continue

        data_snippet = pd.concat([data_snippet, np_data], axis=0)

    data_snippet.to_pickle(res_fle_name)


# build_training_data()
if __name__ == "__main__":
    # build_training_data_from_mcts()
    root_data_folder = os.path.join(os.getcwd(), "local_data", "puct data")
    data_folder = os.path.join(
        root_data_folder, "iteration 0 - 27.03.2025_02-55-07"
    )
    res_file_name = os.path.join(
        root_data_folder, "a.pickle"
    )
    
    build_training_data_from_puct(data_folder, res_file_name)
