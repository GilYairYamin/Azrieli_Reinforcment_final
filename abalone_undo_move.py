import numpy as np
from numba import int8, int64
from numba.experimental import jitclass


_DEFAULT_MAX_SIZE = int64(20)

_undo_move_class_types = [
    ("max_size", int64),
    ("size", int64),
    ("move_list", int8[:, :]),
    ("change_list", int8[:, :, :]),
    ("change_amount", int8[:]),
    ("captured_piece_list", int8[:]),
]


@jitclass(_undo_move_class_types)
class UndoMoveList:
    def __init__(self):
        self.max_size = _DEFAULT_MAX_SIZE
        self.size = int64(0)
        self.move_list = np.zeros((_DEFAULT_MAX_SIZE, 3), dtype=np.int8)
        self.change_list = np.zeros((_DEFAULT_MAX_SIZE, 3, 3), dtype=np.int8)
        self.change_amount = np.zeros((_DEFAULT_MAX_SIZE), dtype=np.int8)
        self.captured_piece_list = np.zeros((_DEFAULT_MAX_SIZE), dtype=np.int8)

    def _increase_size(self):
        old_max = int64(self.max_size)
        self.max_size *= int64(2)
        new_max = int64(self.max_size)

        temp_captured = np.zeros(new_max, dtype=np.int8)
        temp_captured[:old_max] = self.captured_piece_list
        self.captured_piece_list = temp_captured

        temp_move = np.zeros((new_max, 3), dtype=np.int8)
        temp_move[:old_max, :] = self.move_list
        self.move_list = temp_move

        temp_change = np.zeros((new_max, 3, 3), dtype=np.int8)
        temp_change[:old_max, :, :] = self.change_list
        self.change_list = temp_change

        temp_change_amount = np.zeros(new_max, dtype=np.int8)
        temp_change_amount[:old_max] = self.change_amount
        self.change_amount = temp_change_amount

    def add_move(self, row, col, dir_num):
        if self.size >= self.max_size - 1:
            self._increase_size()

        self.move_list[self.size, 0] = int8(row)
        self.move_list[self.size, 1] = int8(col)
        self.move_list[self.size, 2] = int8(dir_num)

        self.change_amount[self.size] = 0
        self.captured_piece_list[self.size] = 0
        self.size += 1

        return self.size - 1

    def add_change(self, idx, row, col, original_piece):
        change_idx = self.change_amount[idx]
        self.change_list[idx, change_idx, 0] = row
        self.change_list[idx, change_idx, 1] = col
        self.change_list[idx, change_idx, 2] = original_piece
        self.change_amount[idx] += 1

    def add_captured_piece(self, idx, captured_piece):
        self.captured_piece_list[idx] = captured_piece

    def get_change_amount(self, idx):
        return self.change_amount[idx]

    def get_captured_piece(self, idx):
        return self.captured_piece_list[idx]

    def get_change(self, idx, change_idx):
        row = self.change_list[idx, change_idx, 0]
        col = self.change_list[idx, change_idx, 1]
        original_piece = self.change_list[idx, change_idx, 2]
        return row, col, original_piece

    def get_move(self, idx):
        row = self.move_list[idx, 0]
        col = self.move_list[idx, 1]
        dir_num = self.move_list[idx, 2]
        return row, col, dir_num

    def get_move_amount(self):
        return self.size

    def remove_last_move(self):
        self.size -= 1

    def is_empty(self):
        return self.size <= 0

    def copy(self):
        copy = UndoMoveList()
        copy.max_size = self.max_size
        copy.size = self.size
        copy.move_list = self.move_list.copy()
        copy.change_list = self.change_list.copy()
        copy.change_amount = self.change_amount.copy()
        copy.captured_piece_list = self.captured_piece_list.copy()
        return copy
