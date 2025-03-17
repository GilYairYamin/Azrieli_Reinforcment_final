import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from abalone_puct_node import PUCTNode
from abalone import (
    BOARD_SIZE,
    ONGOING,
    Abalone,
    WHITE_WIN,
)

from numba import int8, int64, njit, uint64, float64
from numba.experimental import jitclass
from numba.typed import Dict
from numba.types import DictType, np_uint64

from concurrent.futures import ProcessPoolExecutor, as_completed
from abalone_neural_network import AbaloneNetwork

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
        rollout_depth: int = 100,
    ):
        self.max_leaf_explore = max_leaf_explore
        self.explore_constant = explore_constant
        self.rollout_depth = rollout_depth
        self.model = AbaloneNetwork()

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
            if (
                not current_node.is_fully_expanded
            ):
                policy, value = self.model.forward(game)
                current_node.set_values_from_net_result(policy, value)

            propogate(move_stack, tree_dict, game, current_node, value)

        best_child_idx = root_node.pick_child(0)
        return root_node.get_move(best_child_idx)


_BOARD_STATE_DTYPE = [
    ("board", np.int8, (2, BOARD_SIZE, BOARD_SIZE)),
    ("black_captures", np.int8),
    ("white_captures", np.int8),
    ("current_player", np.int8),
]
