import math
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from numba import int8, int64, njit, uint64
from numba.experimental import jitclass
from numba.typed import Dict
from numba.types import DictType, np_uint64

from abalone import ONGOING, TECHNIAL_MOVE_AMOUNT, BLACK, Abalone
from abalone_neural_network import (
    AbaloneNetwork,
    convert_encoded_board_to_tensors,
)
from abalone_puct_node import PUCTNode

_state_class = [
    ("child_idx", int64[:]),
    ("current_key", uint64[:]),
    ("size", int64),
    ("max_size", int64),
]


@jitclass(_state_class)
class StateClass:
    def __init__(self):
        self.max_size = 2000
        self.size = 0
        self.child_idx = np.zeros(self.max_size, dtype=np.int64)
        self.current_key = np.zeros(self.max_size, dtype=np.uint64)

    def _increase_size(self):
        old_max = int64(self.max_size)
        self.max_size *= int64(2)
        new_max = int64(self.max_size)

        temp_child_idx = np.zeros(new_max, dtype=np.int64)
        temp_child_idx[:old_max] = self.child_idx
        self.child_idx = temp_child_idx

        temp_current_key = np.zeros(new_max, dtype=np.uint64)
        temp_current_key[:old_max] = self.current_key
        self.current_key = temp_current_key

    def add_node(self, current_key, child_idx):
        if self.size >= self.max_size:
            self._increase_size()

        new_idx = int64(self.size)
        self.size += int64(1)
        self.current_key[new_idx] = uint64(current_key)
        self.child_idx[new_idx] = int64(child_idx)

    def is_empty(self):
        return self.size <= 0

    def get_size(self):
        return self.size

    def pop_node(self):
        self.size -= int64(1)
        idx = int64(self.size)
        current_key = self.current_key[idx]
        child_idx = self.child_idx[idx]
        return current_key, child_idx

    def head_node(self):
        idx = self.size - 1
        current_key = self.current_key[idx]
        child_idx = self.child_idx[idx]
        return current_key, child_idx


@njit
def select(
    move_stack: StateClass,
    tree_dict: DictType,
    game: Abalone,
    root_key: np_uint64,
    explore_constant,
):
    current_key = root_key
    current_node = tree_dict[current_key]

    while not current_node.is_leaf():
        best_child_idx = current_node.pick_child(explore_constant)
        move_stack.add_node(current_key, best_child_idx)
        row, col, dir_num = current_node.get_move(best_child_idx)

        game.make_move(int8(row), int8(col), int8(dir_num))
        current_key = uint64(game.current_hash)
        current_node = tree_dict[current_key]

    return current_node


@njit
def expand(
    move_stack: StateClass,
    tree_dict: DictType,
    game: Abalone,
    current_node: PUCTNode,
):
    if current_node.is_terminal:
        return current_node

    current_key = uint64(game.current_hash)
    child_idx = current_node.expand_node(game)
    move_stack.add_node(current_key, child_idx)

    new_key = uint64(game.current_hash)
    if new_key in tree_dict:
        child_node = tree_dict[new_key]
    else:
        child_node = PUCTNode(game)
        tree_dict[new_key] = child_node

    current_node.set_child_values(child_idx, child_node)
    return child_node


@njit
def propogate(
    move_stack: StateClass,
    tree_dict: DictType,
    game: Abalone,
    current_node: PUCTNode,
    reward,
):
    game.undo_move(move_stack.get_size())
    current_node.add_visit(-1, reward)
    while not move_stack.is_empty():
        reward = -reward
        node_key, child_idx = move_stack.pop_node()
        node_key = uint64(node_key)
        node: PUCTNode = tree_dict[node_key]
        node.add_visit(child_idx, reward)


class PUCTPlayer:
    def __init__(
        self,
        max_leaf_explore: int = 1000,
        explore_constant: float = math.sqrt(2),
    ):
        self.max_leaf_explore = max_leaf_explore
        self.explore_constant = explore_constant
        self.model = AbaloneNetwork()
        self.is_train = False
        self.train_list = []

    def train(self, is_train=True):
        self.is_train = is_train

    def get_move(self, original_game: Abalone):
        if original_game.status != ONGOING:
            return None

        game = original_game.copy()
        move_stack = StateClass()
        root_key = uint64(game.current_hash)
        root_node: PUCTNode = PUCTNode(game)
        tree_dict = Dict.empty(uint64, PUCTNode.class_type.instance_type)
        tree_dict[root_key] = root_node

        for _ in range(self.max_leaf_explore):
            current_node = select(
                move_stack, tree_dict, game, root_key, self.explore_constant
            )
            current_node = expand(move_stack, tree_dict, game, current_node)

            value = current_node.Q
            if not current_node.is_fully_expanded:
                board_state = game.encode()
                move_mask = game.get_legal_moves_mask()

                board_tensor, extra_tensor, mask_tensor = (
                    convert_encoded_board_to_tensors(
                        board_state, move_mask, single_state=True
                    )
                )

                policy_tensor, value_tensor = self.model.forward(
                    board_tensor, extra_tensor, mask_tensor
                )

                value = value_tensor.item()

                # If in training mode and at the root, apply Dirichlet noise
                if self.is_train and (game.current_hash == root_key):
                    epsilon = 0.5  # mixing factor for noise
                    alpha = 0.06  # Dirichlet concentration parameter
                    noise = np.random.dirichlet(
                        np.full(policy_tensor.shape, alpha)
                    )
                    policy_tensor = (
                        1 - epsilon
                    ) * policy_tensor + epsilon * noise

                np_policy = policy_tensor.detach().numpy().flatten()
                current_node.set_values_from_net_result(np_policy, value)
            propogate(move_stack, tree_dict, game, current_node, value)

        if self.is_train:
            # largest_negative_torch = torch.finfo(torch.float64).min
            results_policy = np.full(TECHNIAL_MOVE_AMOUNT, 0, dtype=np.float64)
            for i in range(len(root_node.children_Q)):
                child_idx = root_node.general_children_idx[i]
                child_q = root_node.children_Q[i]
                results_policy[child_idx] = child_q
            results_policy_tensor = torch.from_numpy(results_policy)

            board_state = original_game.encode()
            move_mask = original_game.get_legal_moves_mask()
            board_tensor, extra_tensor, mask_tensor = (
                convert_encoded_board_to_tensors(
                    board_state, move_mask, single_state=True
                )
            )

            training_tuple = (
                board_tensor,
                extra_tensor,
                mask_tensor,
                results_policy_tensor,
            )

            self.train_list.append(training_tuple)

        best_child_idx = root_node.pick_child(0)
        return root_node.get_move(best_child_idx)

    def pop_data_list(self):
        res = self.train_list
        self.train_list = []
        return res


# Converted simulation function using PUCTPlayer.
def simulate_games(pid, number: int = 50, max_game_depth=-1):
    root_dir = os.path.join(os.getcwd(), "local_data")
    data_dir = os.path.join(root_dir, "puct data")

    os.makedirs(data_dir, exist_ok=True)

    # Use PUCTPlayer instead of the old MCTSPlayer.
    computer_player = PUCTPlayer(max_leaf_explore=800)
    computer_player.train(is_train=True)  # Enable training mode.

    for game_id in tqdm(range(number)):
        now = datetime.now()
        readable_time = now.strftime("%d.%m.%Y_%H:%M:%S")
        time_str = f"current_time: {readable_time}"
        iteration_str = f"started iteration {game_id + 1}"
        tqdm.write(f"process {pid}, {time_str}, {iteration_str}")

        game = Abalone(True)
        df = pd.DataFrame(
            columns=[
                "board_tensor",
                "extra_tensor",
                "mask_tensor",
                "results_policy",
                "final_status",
            ]
        )

        move_amount = 0
        while game.status == ONGOING and move_amount != max_game_depth:
            move_amount += 1
            if (move_amount + 1) % 20 == 0:
                tqdm.write(
                    f"pid {pid}, game {game_id}, move {move_amount + 1}"
                )
            move = computer_player.get_move(game)
            row, col, dir_num = move
            game.make_move(row, col, dir_num)

        data_list = computer_player.pop_data_list()
        state_idx = start_idx = df.shape[0]
        for training_tuple in data_list:
            board_tensor, extra_tensor, mask_tensor, results_policy = (
                training_tuple
            )
            df.loc[state_idx, "board_tensor"] = board_tensor.numpy()
            df.loc[state_idx, "extra_tensor"] = extra_tensor.numpy()
            df.loc[state_idx, "mask_tensor"] = mask_tensor.numpy()
            df.loc[state_idx, "results_policy"] = results_policy.numpy()
            state_idx += 1

        end_idx = state_idx
        # Determine final status and backpropagate final outcomes.
        final_status = game.abalone_heuristic()

        if game.player == BLACK:
            final_status = -game.status

        for idx in range(end_idx - 1, start_idx - 1, -1):
            df.loc[idx, "final_status"] = final_status
            final_status = -final_status

    now = datetime.now()
    readable_time = now.strftime("%d.%m.%Y_%H-%M-%S")
    file_name = f"p{pid}_game{game_id}_{readable_time}.pickle"
    file_path = os.path.join(data_dir, file_name)
    df.to_pickle(file_path)
    print(f"finished pid {pid}, game {game_id}, {state_idx} moves")


def play_train_model(process_num=6):
    pass


def play_game():
    player = PUCTPlayer()
    game = Abalone(True)
    print(game.to_string())
    while game.status == ONGOING:
        move = player.get_move(game)
        row, col, dir_num = move
        game.make_move(row, col, dir_num)
        print(game.to_string())

    print(f"winner is {game.status}")


if __name__ == "__main__":
    # play_game()

    simulate_games(0, 1, 2)
