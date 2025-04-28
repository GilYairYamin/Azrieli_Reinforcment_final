import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from numba import float32, int64, njit, uint64
from numba.experimental import jitclass
from numba.typed import Dict
from numba.types import DictType, np_uint64

from abalone import (
    BOARD_SIZE,
    ONGOING,
    BLACK,
    WHITE,
    Abalone,
)
from abalone_mcts_node import MCTSNode

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
    current_node: MCTSNode = tree_dict[current_key]

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
    current_node: MCTSNode,
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
        child_node = MCTSNode(game)
        tree_dict[new_key] = child_node

    current_node.set_child_values(child_idx, child_node)

    return child_node


@njit
def propogate(
    move_stack: StateClass,
    tree_dict: DictType,
    game: Abalone,
    current_node: MCTSNode,
    reward,
):
    game.undo_move(move_stack.get_size())
    current_node.add_visit(-1, reward)
    while not move_stack.is_empty():
        reward = -reward
        node_key, child_idx = move_stack.pop_node()
        node_key = uint64(node_key)
        node: MCTSNode = tree_dict[node_key]
        node.add_visit(child_idx, reward)


_MCTSNode_type = MCTSNode.class_type.instance_type


@njit
def get_move_and_data(
    max_leaf_explore, rollout_depth, explore_constant, original_game: Abalone
):
    if original_game.status != ONGOING:
        return None

    game = original_game.copy()
    move_stack = StateClass()
    root_key = uint64(game.current_hash)
    root_node: MCTSNode = MCTSNode(game)
    tree_dict = Dict.empty(uint64, _MCTSNode_type)
    tree_dict[root_key] = root_node

    for _ in range(max_leaf_explore):
        current_node = select(
            move_stack, tree_dict, game, root_key, explore_constant
        )
        current_node = expand(move_stack, tree_dict, game, current_node)
        reward = game.rollout(rollout_depth)
        propogate(move_stack, tree_dict, game, current_node, reward)

    best_child_idx = root_node.pick_child_by_uct(0)
    return root_node.get_move_and_values(best_child_idx)


class MCTSPlayer:
    def __init__(
        self,
        max_leaf_explore: int = 1000,
        explore_constant: float = math.sqrt(2),
        rollout_depth: int = 100,
    ):
        self.max_leaf_explore = max_leaf_explore
        self.explore_constant = explore_constant
        self.rollout_depth = rollout_depth

    def get_move_and_data(self, original_game: Abalone):
        return get_move_and_data(
            int64(self.max_leaf_explore),
            int64(self.rollout_depth),
            float32(self.explore_constant),
            original_game,
        )

    def get_move(self, original_game: Abalone):
        if original_game.status != ONGOING:
            return None

        game = original_game.copy()
        move_stack = StateClass()
        root_key = uint64(game.current_hash)
        root_node: MCTSNode = MCTSNode(game)
        tree_dict = Dict.empty(uint64, MCTSNode.class_type.instance_type)
        tree_dict[root_key] = root_node

        for _ in range(self.max_leaf_explore):
            current_node = select(
                move_stack, tree_dict, game, root_key, self.explore_constant
            )
            current_node = expand(move_stack, tree_dict, game, current_node)
            reward = game.rollout(self.rollout_depth)
            propogate(move_stack, tree_dict, game, current_node, reward)

        best_child_idx = root_node.pick_child_by_visits()
        return root_node.get_move(best_child_idx)


_BOARD_STATE_DTYPE = [
    ("board", np.int8, (2, BOARD_SIZE, BOARD_SIZE)),
    ("black_captures", np.int8),
    ("white_captures", np.int8),
    ("current_player", np.int8),
]


def convert_state_to_np_arr(game_state):
    board, black_captures, white_captures, current_player = game_state
    np_state = np.array(1, dtype=_BOARD_STATE_DTYPE)
    np_state["board"] = board
    np_state["black_captures"] = black_captures
    np_state["white_captures"] = white_captures
    np_state["current_player"] = current_player
    return np_state


def simulate_game(game_id, max_game_length: int = 500):
    current_dir = os.path.join(os.getcwd(), "local_data")
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    now = datetime.now()
    readable_time = now.strftime("%d.%m.%Y_%H-%M-%S")

    computer_player = MCTSPlayer(max_leaf_explore=1800, rollout_depth=10)
    game = Abalone(True)

    df = pd.DataFrame(
        columns=[
            "game_states",
            "Q",
            "children_move_idx",
            "children_move_N",
            "final_status",
        ]
    )

    state_idx = 0
    while game.status == ONGOING and max_game_length != state_idx:
        move, Q, children_move_idx, children_move_N = (
            computer_player.get_move_and_data(game)
        )

        game_state = convert_state_to_np_arr(game.encode())
        df.loc[state_idx, "game_states"] = game_state
        df.loc[state_idx, "Q"] = Q
        df.loc[state_idx, "children_move_idx"] = children_move_idx
        df.loc[state_idx, "children_move_N"] = children_move_N
        game.make_move(move)
        state_idx += 1

    final_status = game.abalone_heuristic(game.player)

    for idx in range(state_idx - 1, -1, -1):
        df.loc[idx, "final_status"] = final_status
        final_status = -final_status

    file_name = f"game-{game_id}_{readable_time}.pickle"
    file_path = os.path.join(data_dir, file_name)
    df.to_pickle(file_path)
    return file_name, state_idx


def run_simulations_in_parallel(num_processes: int, num_games: int):
    futures = {}

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for game_idx in range(num_games):
            future = executor.submit(simulate_game, game_idx, 500)
            futures[future] = game_idx

        for future in tqdm(as_completed(futures), total=len(futures)):
            process_index = futures[future]
            try:
                file_name, state_idx = future.result()
                str_start = f"game {process_index} completed with {state_idx}"
                str_end = f"moves, and the file {file_name}"
                tqdm.write(f"{str_start} {str_end}")
            except Exception as e:
                tqdm.write(f"game {process_index} failed with error: {e}")


def test_2():
    game = Abalone(True)

    print("Welcome to Abalone!")
    print("Player 1 is RED (R) and Player 2 is YELLOW (Y).\n")

    computer_player = MCTSPlayer(
        max_leaf_explore=1700,
        explore_constant=1,
        rollout_depth=10,
    )

    count = 0
    while game.status == ONGOING:
        count += 1
        print(game)
        print(
            "\nCurrent Player:",
            "BLACK" if game.player == BLACK else "WHITE",
        )

        if game.player == WHITE:
            move_idx = computer_player.get_move(game)
        else:
            move_idx = computer_player.get_move(game)

        game.make_move(move_idx)

    print(game)
    if game.status == BLACK:
        print("\nBLACK (Player 1) wins!")
    elif game.status == WHITE:
        print("\nWHITE (Player 2) wins!")
    else:
        print("\nIt's a draw!")

    print(f"finished with {count} moves")


if __name__ == "__main__":
    run_simulations_in_parallel(5, 10000)
