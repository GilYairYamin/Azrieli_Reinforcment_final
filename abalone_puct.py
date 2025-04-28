import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from numba import int64, njit, uint64
from numba.experimental import jitclass
from numba.typed import Dict
from numba.types import DictType, np_uint64
from tqdm import tqdm

from abalone import ONGOING, TECHNIAL_MOVE_AMOUNT, Abalone
from abalone_neural_network import (
    AbaloneNetwork,
    convert_encoded_board_to_tensors,
)
from abalone_puct_node import PUCTNode


@njit
def add_dirichlet_noise(policy, epsilon=0.25, alpha=0.03):
    # Sample noise from a Dirichlet distribution.
    noise = np.random.dirichlet([alpha] * len(policy))

    # Blend the original policy with the noise.
    noisy_policy = (1 - epsilon) * policy + epsilon * noise

    return noisy_policy


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

    def __len__(self):
        return self.size


@njit
def select(
    move_stack: StateClass,
    tree_dict: DictType,
    game: Abalone,
    root_key: np_uint64,
    explore_constant,
):
    current_key = root_key
    current_node: PUCTNode = tree_dict[current_key]

    while not current_node.is_leaf():
        best_child_idx = current_node.pick_child_by_uct(explore_constant)
        move_stack.add_node(current_key, best_child_idx)
        move_idx = current_node.get_move(best_child_idx)
        game.make_move(move_idx)
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
        max_leaf_explore: int = 800,
        explore_constant: float = math.sqrt(2),
        noise_level: int = 0,
        max_noise_e: float = 0.25,
        max_noise_alpha: float = 0.06,
    ):
        self.max_leaf_explore = max_leaf_explore
        self.explore_constant = explore_constant
        self.noise_level = noise_level
        self.max_noise_e = max_noise_e
        self.noise_alpha = max_noise_alpha

        self.model = AbaloneNetwork()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    def get_value_and_policy(self, game, current_node):
        value = current_node.Q
        if current_node.is_fully_expanded:
            return None, value

        board_state = game.encode()
        move_mask = game.get_legal_moves_mask()

        board_tensor, extra_tensor, mask_tensor = (
            convert_encoded_board_to_tensors(
                board_state, move_mask, single_state=True, device=self.device
            )
        )

        with torch.no_grad():
            policy_tensor, value_tensor = self.model.forward(
                board_tensor, extra_tensor, mask_tensor
            )

        value = value_tensor.item()
        np_policy = policy_tensor.cpu().detach().numpy().flatten()
        return np_policy, value

    def get_move_and_policy(self, original_game: Abalone, train=True):
        if original_game.status != ONGOING:
            return None

        game = original_game.copy()
        move_stack = StateClass()
        root_key = uint64(game.current_hash)
        root_node: PUCTNode = PUCTNode(game)
        tree_dict = Dict.empty(uint64, PUCTNode.class_type.instance_type)
        tree_dict[root_key] = root_node

        np_policy, value = self.get_value_and_policy(game, root_node)
        if train:
            np_policy = add_dirichlet_noise(np_policy, self.max_noise_e, 0.03)
        root_node.set_values_from_net_result(np_policy, value)

        for _ in range(self.max_leaf_explore):
            current_node = select(
                move_stack, tree_dict, game, root_key, self.explore_constant
            )
            current_node = expand(move_stack, tree_dict, game, current_node)

            np_policy, value = self.get_value_and_policy(game, current_node)

            if np_policy is not None:
                level = move_stack.size
                if train and level < self.noise_level:
                    np_policy = add_dirichlet_noise(
                        np_policy,
                        self.max_noise_e / (level + 1),
                        self.noise_alpha,
                    )
                current_node.set_values_from_net_result(np_policy, value)

            propogate(move_stack, tree_dict, game, current_node, value)

        largest_negative_torch = np.finfo(np.float32).min
        results_policy = np.full(
            TECHNIAL_MOVE_AMOUNT, largest_negative_torch, dtype=np.float32
        )

        for i in range(len(root_node.children_N)):
            child_idx = root_node.children_move_idx[i]
            child_N = root_node.children_N[i]
            results_policy[child_idx] = child_N

        policy_tensor = torch.from_numpy(results_policy)
        policy_tensor = policy_tensor.softmax(-1)
        np_policy = policy_tensor.detach().numpy()

        best_child_idx = root_node.pick_child_by_visits()
        return root_node.get_move(best_child_idx), np_policy

    def get_move(self, original_game: Abalone):
        if original_game.status != ONGOING:
            return None

        game = original_game.copy()
        move_stack = StateClass()
        root_key = uint64(game.current_hash)
        root_node: PUCTNode = PUCTNode(game)
        tree_dict = Dict.empty(uint64, PUCTNode.class_type.instance_type)
        tree_dict[root_key] = root_node

        np_policy, value = self.get_value_and_policy(game, root_node)
        root_node.set_values_from_net_result(np_policy, value)

        for _ in range(self.max_leaf_explore):
            current_node = select(
                move_stack, tree_dict, game, root_key, self.explore_constant
            )
            current_node = expand(move_stack, tree_dict, game, current_node)

            np_policy, value = self.get_value_and_policy(game, current_node)

            if np_policy is not None:
                current_node.set_values_from_net_result(np_policy, value)

            propogate(move_stack, tree_dict, game, current_node, value)

        best_child_idx = root_node.pick_child_by_uct(0)
        return root_node.get_move(best_child_idx)


def simulate_game(res_folder, game_id, max_game_depth=-1):
    os.makedirs(res_folder, exist_ok=True)

    game = Abalone(True)
    computer_player = PUCTPlayer()

    move_amount = 0
    training_list = []
    while game.status == ONGOING and move_amount != max_game_depth:
        move_amount += 1
        move, np_policy = computer_player.get_move_and_policy(game)
        training_list.append(
            (game.encode(), game.get_legal_moves_mask(), np_policy)
        )
        game.make_move(move)

    final_status = game.abalone_heuristic(game.player)

    training_dict = []
    for board_state, legal_move_mask, policy in training_list:
        training_dict.append(
            {
                "board_state": board_state,
                "legal_move_mask": legal_move_mask,
                "policy": policy,
                "value": final_status
                if game.player == board_state[3]
                else -final_status,
            }
        )
    df = pd.DataFrame(training_dict)
    now = datetime.now()
    readable_time = now.strftime("%d.%m.%Y_%H-%M-%S")
    file_name = f"game{game_id}_{readable_time}.pickle"
    file_path = os.path.join(res_folder, file_name)
    df.to_pickle(file_path)
    return file_name, move_amount


def simulate_games_processes(res_folder, executor, num_games, max_depth=-1):
    os.makedirs(res_folder, exist_ok=True)
    futures = {}
    for game_idx in range(num_games):
        future = executor.submit(
            simulate_game, res_folder, game_idx, max_depth
        )
        futures[future] = game_idx
    torch.cuda.empty_cache()

    for future in tqdm(as_completed(futures), total=len(futures)):
        process_index = futures[future]
        try:
            file_name, state_idx = future.result()
            str_start = f"game {process_index} completed with {state_idx}"
            str_end = f"moves, and the file {file_name}"
            tqdm.write(f"{str_start} {str_end}")
        except Exception as e:
            tqdm.write(f"game {process_index} failed with error: {e}")


def simulate_games_single(res_folder, num_games, max_depth=-1):
    for i in tqdm(range(num_games)):
        file_name, state_idx = simulate_game(
            res_folder, i, max_game_depth=max_depth
        )
        str_start = f"game {i} completed with {state_idx}"
        str_end = f"moves, and the file {file_name}"
        tqdm.write(f"{str_start} {str_end}")


def generate_data_and_train_network(
    num_iterations=1,
    games_per_iteration=10,
    max_depth_start=40,
    max_depth_increment=5,
):
    data_folder = os.path.join(os.getcwd(), "local_data", "puct data")

    futures = {}
    with ProcessPoolExecutor(max_workers=5) as executor:
        for iter in range(num_iterations):
            now = datetime.now()
            readable_time = now.strftime("%d.%m.%Y_%H-%M-%S")
            raw_data_folder = os.path.join(
                data_folder, f"iteration {iter} - {readable_time}"
            )

            for game_id in range(games_per_iteration):
                future = executor.submit(
                    simulate_game,
                    raw_data_folder,
                    game_id + games_per_iteration * iter,
                    max_depth_start + iter * max_depth_increment,
                )
                futures[future] = game_id

            for future in tqdm(as_completed(futures), total=len(futures)):
                process_index = futures[future]
                try:
                    file_name, state_idx = future.result()
                    str_start = (
                        f"game {process_index} completed with {state_idx}"
                    )
                    str_end = f"moves, and the file {file_name}"
                    tqdm.write(f"{str_start} {str_end}")
                except Exception as e:
                    tqdm.write(f"game {process_index} failed with error: {e}")

            # train_network(raw_data_folder)


def play_game():
    player = PUCTPlayer()
    game = Abalone(True)
    print(game.to_string())

    while game.status == ONGOING:
        move = player.get_move(game)
        game.make_move(move)
        print(game.to_string())

    print(f"winner is {game.status}")


if __name__ == "__main__":
    play_game()
    # generate_data_and_train_network(2, 5, 10, 5)
