import numpy as np

from abalone import ONGOING, Abalone, move_to_idx
from numba import boolean, float64, int8, int16, int64, uint64
from numba.experimental import jitclass


_abalone_node_types = [
    ("N", int16),
    ("Q", float64),
    ("is_terminal", boolean),
    ("is_fully_expanded", boolean),
    ("children_visited_amount", int16),
    ("children_move_idx", int16[:]),
    ("children_N", int16[:]),
    ("children_Q", float64[:]),
    ("children_key", uint64[:]),
    ("is_in_current_path", boolean[:]),
]


@jitclass(_abalone_node_types)
class PUCTNode:
    def __init__(self, game: Abalone):
        self.N = int16(0)
        self.Q = float64(0)
        self.is_terminal = boolean(game.status != ONGOING)
        self.is_fully_expanded = boolean(False)
        self.children_visited_amount = int16(0)

        size = 0
        if not self.is_terminal:
            self.children_move_idx = game.legal_moves()
            np.random.shuffle(self.children_move_idx)
            size = self.children_move_idx.shape[0]
        else:
            self.children_move_idx = np.zeros(0, dtype=np.int16)

        self.children_N = np.zeros(size, dtype=np.int16)
        self.children_Q = np.zeros(size, dtype=np.float64)
        self.children_key = np.zeros(size, dtype=np.uint64)
        self.is_in_current_path = np.zeros(size, dtype=np.bool)

    def set_values_from_net_result(self, policy, value):
        self.Q = value
        if self.is_terminal:
            self.is_fully_expanded = boolean(True)
            return

        for i in range(len(self.children_Q)):
            row, col, dir_num = self.get_move(i)
            move_idx = move_to_idx(row, col, dir_num)
            self.children_Q[i] = policy[move_idx]

    def add_visit(self, child_idx, reward):
        self.N += 1
        self.Q = self.Q + (reward - self.Q) / self.N

        if child_idx < 0:
            return

        self.is_in_current_path[child_idx] = boolean(False)
        self.children_N[child_idx] += 1
        Q, N = self.children_Q[child_idx], self.children_N[child_idx]
        self.children_Q[child_idx] = Q + (reward - Q) / N

    def _calc_UCT(self, child_idx, explore_constant):
        child_N = self.children_N[child_idx]
        child_Q = self.children_Q[child_idx]
        explore = np.sqrt(np.log(self.N) / child_N) * explore_constant
        return child_Q + explore

    def pick_child(self, explore_constant):
        if self.children_visited_amount <= 0:
            return -1

        max_val = -500000
        max_idx = -1
        for child_idx in range(self.children_visited_amount):
            if self.is_in_current_path[child_idx]:
                continue

            child_val = self._calc_UCT(child_idx, explore_constant)
            if child_val < max_val:
                continue

            max_val = child_val
            max_idx = child_idx

        self.is_in_current_path[max_idx] = boolean(True)
        return max_idx

    def expand_node(self, game: Abalone):
        if self.is_terminal:
            return int64(-1)

        child_idx = int64(self.children_visited_amount)
        row, col, dir_num = self.get_move(child_idx)
        game.make_move(row, col, dir_num)
        self.children_key[child_idx] = uint64(game.current_hash)
        self.children_N[child_idx] = 0
        self.children_Q[child_idx] = 0
        self.general_children_idx[child_idx] = int16(move_to_idx(row, col, dir_num))

        self.children_visited_amount += 1
        if self.children_visited_amount >= self.children_key.shape[0]:
            self.is_fully_expanded = True

        return child_idx

    def set_child_values(self, child_idx, child_node):
        self.children_Q[child_idx] = -child_node.Q
        self.children_N[child_idx] = child_node.N

    def get_key(self, child_idx):
        return self.children_key[child_idx]

    def get_move(self, child_idx):
        return self.children_move_idx[child_idx]

    def get_move_and_values(self, child_idx):
        return (
            self.get_move(child_idx),
            self.Q,
            self.general_children_idx,
            self.children_Q,
        )

    def is_leaf(self):
        return not self.is_fully_expanded or self.is_terminal
