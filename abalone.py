import numpy as np
import torch
import os
from abalone_undo_move import UndoMoveList
from numba import int8, njit, uint64, int16
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType


@njit(cache=True)
def compute_row_ranges(board_size, mid_row_index):
    row_start_arr = np.zeros(board_size, dtype=np.int8)
    row_end_arr = np.full(board_size, board_size, dtype=np.int8)
    row_len_arr = np.full(board_size, board_size, dtype=np.int8)

    for row_index in range(BOARD_SIZE):
        row_offset = abs(row_index - mid_row_index)
        row_len = board_size - row_offset
        row_len_arr[row_index] = row_len

        if row_index < mid_row_index:
            row_end_arr[row_index] = row_len_arr[row_index]
        else:
            row_start_arr[row_index] = row_offset

    return row_start_arr, row_end_arr, row_len_arr


@njit(cache=True)
def create_conversion_types(board_size, row_start, row_end, num_directions):
    from_move_to_idx_dict = np.full(
        (board_size, board_size, num_directions), -1, dtype=np.int16
    )

    move_count = 0
    for row_idx in range(board_size):
        for col_idx in range(row_start[row_idx], row_end[row_idx]):
            for dir_num in range(num_directions):
                from_move_to_idx_dict[row_idx, col_idx, dir_num] = move_count
                move_count += 1

    idx = 0
    from_idx_to_move_dict = np.full((move_count, 3), -1, dtype=np.int16)
    for row_idx in range(board_size):
        for col_idx in range(row_start[row_idx], row_end[row_idx]):
            for dir_num in range(num_directions):
                from_idx_to_move_dict[idx, 0] = row_idx
                from_idx_to_move_dict[idx, 1] = col_idx
                from_idx_to_move_dict[idx, 2] = dir_num
                idx += 1

    return from_idx_to_move_dict, from_move_to_idx_dict


@njit(cache=True)
def create_np_mask(board_size, row_start, row_end):
    np_arr = np.zeros((board_size, board_size), dtype=np.int8)
    for row_idx in range(0, board_size):
        for col_idx in range(row_start[row_idx], row_end[row_idx]):
            np_arr[row_idx, col_idx] = 1

    return np_arr


@njit(cache=True)
def random_uint64():
    lo = np.random.randint(0, 2**32)
    hi = np.random.randint(0, 2**32)
    return np.uint64(hi) << 32 | np.uint64(lo)


@njit(cache=True)
def compute_zobrist_values(board_size, row_start, row_end):
    amount = board_size * board_size * 2 + 2
    numbers = np.zeros(amount, dtype=np.uint64)
    board_placement_hash = np.zeros(
        (board_size, board_size, 2), dtype=np.uint64
    )
    current_player_hash = np.array([numbers[0], numbers[1]], dtype=np.uint64)

    while amount > 0:
        should_continue = False
        random_int: np.uint64 = random_uint64()
        for i in range(amount):
            if numbers[i] == random_int:
                should_continue = True
                break

        if should_continue:
            continue

        numbers[amount] = random_int
        amount -= 1

    idx = 2
    for row_index in range(0, board_size):
        for col_index in range(row_start[row_index], row_end[row_index]):
            for piece in range(2):
                board_placement_hash[row_index, col_index, piece] = numbers[
                    idx
                ]
                idx += 1

    return board_placement_hash, current_player_hash


def load_zobrist_values(board_size, row_start, row_end):
    base_path = os.path.join(os.getcwd(), "local_data")
    dir_path = os.path.join(base_path, "zobrist")
    os.makedirs(dir_path, exist_ok=True)
    board_placement_file = os.path.join(dir_path, "board_placement.npy")
    current_player_file = os.path.join(dir_path, "players.npy")

    try:
        board_placement_hash = np.load(board_placement_file)
        current_player_hash = np.load(current_player_file)

        if board_placement_hash.shape != (board_size, board_size, 2):
            raise Exception("wrong shape")

        if current_player_hash.shape != (2,):
            raise Exception("wrong shape")

    except Exception as e:
        print("failed to load file")
        print(e)
        board_placement_hash, current_player_hash = compute_zobrist_values(
            board_size, row_start, row_end
        )
        np.save(board_placement_file, board_placement_hash)
        np.save(current_player_file, current_player_hash)

    finally:
        return board_placement_hash, current_player_hash


EMPTY = DRAW = int8(0)
BLACK = BLACK_WIN = int8(1)
WHITE = WHITE_WIN = int8(-1)
INVALID = ONGOING = int8(-5)

BLACK_INDEX = int8(0)
WHITE_INDEX = int8(1)

BOARD_SIZE = int8(9)
_MID_ROW_INDEX = int8(BOARD_SIZE // 2)

_PLAYER_PIECE_AMOUNT = int8(14)
_MAX_SIMILAR_PUSH = int8(3)
_MAX_CAPTURES = int8(6)

_DIRECTIONS = np.array(
    [(0, 1), (0, -1), (1, 0), (1, 1), (-1, 0), (-1, -1)], dtype=np.int8
)
_DIR_NAMES = List()
_DIR_NAMES.extend(["E", "W", "SW", "SE", "NE", "NW"])

_DIR_DICT = Dict()
for idx, dir in enumerate(_DIR_NAMES):
    _DIR_DICT[dir] = idx

_NUM_DIRECTIONS = np.int8(len(_DIRECTIONS))

_ROW_START, _ROW_END, _ROW_LEN = compute_row_ranges(BOARD_SIZE, _MID_ROW_INDEX)
VALID_BOARD_MASK = create_np_mask(BOARD_SIZE, _ROW_START, _ROW_END)

_FROM_INDEX_TO_MOVE_ARR, _FROM_MOVE_TO_INDEX_ARR = create_conversion_types(
    BOARD_SIZE, _ROW_START, _ROW_END, _NUM_DIRECTIONS
)

_BOARD_PLACEMENT_HASH, _PLAYER_HASHES = load_zobrist_values(
    BOARD_SIZE, _ROW_START, _ROW_END
)

_PIECE_START_RANGES = np.array([1, _PLAYER_PIECE_AMOUNT + 1], dtype=np.int8)
_PIECE_END_RANGES = np.array(
    [_PLAYER_PIECE_AMOUNT + 1, _PLAYER_PIECE_AMOUNT * 2 + 1], dtype=np.int8
)
_PIECE_AMOUNT = int8(_PLAYER_PIECE_AMOUNT * 2 + 1)

_ACTUAL_POSSIBLE_MOVE_AMOUNT = _PLAYER_PIECE_AMOUNT * _NUM_DIRECTIONS
TECHNIAL_MOVE_AMOUNT = _FROM_INDEX_TO_MOVE_ARR.shape[0]


@njit(inline="always", cache=True)
def move_to_idx(row, col, dir_num):
    row, col, dir_num = int8(row), int8(col), int8(dir_num)
    return int16(_FROM_MOVE_TO_INDEX_ARR[row, col, dir_num])


@njit(inline="always", cache=True)
def idx_to_move(idx):
    idx = int16(idx)
    row = _FROM_INDEX_TO_MOVE_ARR[idx, 0]
    col = _FROM_INDEX_TO_MOVE_ARR[idx, 1]
    dir_num = _FROM_INDEX_TO_MOVE_ARR[idx, 2]
    return row, col, dir_num


@njit(inline="always", cache=True)
def _is_in_bounds(row, col):
    return (
        row >= 0
        and row < BOARD_SIZE
        and col >= _ROW_START[row]
        and col < _ROW_END[row]
    )


@njit(inline="always", cache=True)
def _player_to_player_index(player):
    if player == BLACK:
        return BLACK_INDEX
    if player == WHITE:
        return WHITE_INDEX
    return -1


@njit(inline="always", cache=True)
def _piece_to_player(piece):
    if piece == 0:
        return EMPTY

    black_start = _PIECE_START_RANGES[BLACK_INDEX]
    white_start = _PIECE_START_RANGES[WHITE_INDEX]
    black_end = _PIECE_END_RANGES[BLACK_INDEX]
    white_end = _PIECE_END_RANGES[WHITE_INDEX]

    if black_start <= piece < black_end:
        return BLACK
    if white_start <= piece < white_end:
        return WHITE

    return INVALID


@njit(inline="always", cache=True)
def piece_to_player_index(piece):
    return _player_to_player_index(_piece_to_player(piece))


@njit(inline="always", cache=True)
def other_player(player):
    if player == BLACK:
        return WHITE
    if player == WHITE:
        return BLACK
    return INVALID


@njit(inline="always", cache=True)
def _value_to_str(val):
    if val == 0:
        return "_"

    player = _piece_to_player(val)
    if player == BLACK:
        return "B"

    if player == WHITE:
        return "W"

    return "E"


@njit(inline="always", cache=True)
def _switch_player_hash(hash):
    hash ^= _PLAYER_HASHES[BLACK_INDEX]
    hash ^= _PLAYER_HASHES[WHITE_INDEX]
    return hash


@njit(inline="always", cache=True)
def _switch_piece_hash(row, col, old_piece, new_piece, old_hash):
    if old_piece != EMPTY:
        player_index = piece_to_player_index(old_piece)
        old_hash ^= _BOARD_PLACEMENT_HASH[row, col, player_index]

    if new_piece != EMPTY:
        player_index = piece_to_player_index(new_piece)
        old_hash ^= _BOARD_PLACEMENT_HASH[row, col, player_index]

    return old_hash


@njit(inline="always", cache=True)
def _check_not_repeat(past_state_hashes, next_key):
    if next_key in past_state_hashes:
        return past_state_hashes[next_key] < 2
    return True


numbalone_class_types = [
    ("status", int8),
    ("player", int8),
    ("current_hash", uint64),
    ("board", int8[:, :]),
    ("captures", int8[:]),
    ("active_positions", int8[:, :]),
    ("past_state_hashes", DictType(uint64, int8)),
    ("undo_move_stack", UndoMoveList.class_type.instance_type),
]


@jitclass(numbalone_class_types)
class Abalone:
    def _init_board(self):
        row_start_arr = _ROW_START
        row_end_arr = _ROW_END
        board_size = BOARD_SIZE
        for row_i in range(board_size):
            for col_i in range(row_start_arr[row_i], row_end_arr[row_i]):
                self.board[row_i][col_i] = EMPTY

    def __init__(self, init_board):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), INVALID, dtype=np.int8)
        self.status = ONGOING
        self.current_hash = uint64(0)
        self.player = BLACK
        self.captures = np.zeros((2,), dtype=np.int8)
        self.active_positions = np.full((_PIECE_AMOUNT, 2), -1, dtype=np.int8)
        self.past_state_hashes = Dict.empty(uint64, int8)
        self.undo_move_stack = UndoMoveList()

        if init_board:
            self._init_board()
            self._populate_board(WHITE)
            self._populate_board(BLACK)

    def _populate_board(self, player):
        piece_amount = _PLAYER_PIECE_AMOUNT
        row_start_arr = _ROW_START
        row_end_arr = _ROW_END
        row_len_arr = _ROW_LEN

        row_index_end = _MID_ROW_INDEX
        if player == WHITE:
            direction = np.int8(1)
            row_index_start = np.int8(0)
        else:
            direction = np.int8(-1)
            row_index_start = np.int8(BOARD_SIZE - 1)

        current_piece = _PIECE_START_RANGES[_player_to_player_index(player)]

        for row_index in range(row_index_start, row_index_end, direction):
            if piece_amount <= 0:
                break

            row_start = row_start_arr[row_index]
            row_end = row_end_arr[row_index]
            row_len = row_len_arr[row_index]

            if piece_amount < row_len:
                half_pieces = np.int8(piece_amount // 2)
                row_mid = np.int8((row_start + row_end) // 2)
                if direction < 0 and piece_amount % 2 == 0:
                    row_mid += 1

                row_start = np.int8(row_mid - half_pieces)
                row_end = row_mid + (piece_amount - half_pieces)
                piece_amount = int8(0)

            else:
                piece_amount -= row_len

            for col_index in range(row_start, row_end):
                self._switch_piece(row_index, col_index, current_piece)
                current_piece += 1

    def _capture_piece(self, piece):
        if piece == 0:
            return

        player = _piece_to_player(piece)
        if player == BLACK:
            self.captures[WHITE_INDEX] += int8(1)
            if self.captures[WHITE_INDEX] >= _MAX_CAPTURES:
                self.status = WHITE_WIN

        if player == WHITE:
            self.captures[BLACK_INDEX] += int8(1)
            if self.captures[BLACK_INDEX] >= _MAX_CAPTURES:
                self.status = BLACK_WIN

    def switch_player(self):
        self.player = other_player(self.player)
        self.current_hash = _switch_player_hash(self.current_hash)

    def _switch_piece(self, row, col, new_piece):
        current_piece = self.board[row, col]
        if current_piece != EMPTY:
            self.active_positions[current_piece, 0] = -1
            self.active_positions[current_piece, 1] = -1

        self.board[row][col] = new_piece
        if new_piece != EMPTY:
            self.active_positions[new_piece, 0] = row
            self.active_positions[new_piece, 1] = col

        self.current_hash = _switch_piece_hash(
            row, col, current_piece, new_piece, self.current_hash
        )

        return current_piece

    def copy(self, clone_stack=False):
        game = Abalone(False)
        game.board = self.board.copy()
        game.captures = self.captures.copy()
        game.active_positions = self.active_positions.copy()
        game.past_state_hashes = self.past_state_hashes.copy()

        game.status = self.status
        game.player = self.player
        game.current_hash = self.current_hash

        if clone_stack:
            game.undo_move_stack = self.undo_move_stack.copy()

        return game

    def compare_board(self, other: "Abalone"):
        return (
            self.status == other.status
            and self.player == other.player
            and self.current_hash == other.current_hash
            and self.past_state_hashes == other.past_state_hashes
            and np.array_equal(self.board, other.board)
            and np.array_equal(self.captures, other.captures)
            and np.array_equal(self.active_positions, other.active_positions)
        )

    def to_string(self):
        return self.__str__()

    def __str__(self):
        result = ""
        _row_start = _ROW_START
        _row_end = _ROW_END
        _board_size = BOARD_SIZE

        for row_index in range(BOARD_SIZE):
            row_start = _row_start[row_index]
            row_end = _row_end[row_index]

            left_offset = _board_size - row_index
            right_offset = (
                1 + 2 * row_start if row_index > _MID_ROW_INDEX else 1
            )
            str = " " * left_offset + f"{row_index + 1}" + " " * right_offset

            for col_index in range(row_start, row_end):
                str += f"{_value_to_str(self.board[row_index, col_index])} "

            result += f"{str}\n"
        result += "  A B C D E F G H I\n"
        result += f"black score: {self.captures[BLACK_INDEX]}\n"
        result += f"white score: {self.captures[WHITE_INDEX]}\n"
        return result

    def _is_legal_move(self, row, col, dir_num):
        d_row = _DIRECTIONS[dir_num, 0]
        d_col = _DIRECTIONS[dir_num, 1]

        count_similar = 0
        curr_row, curr_col = int8(row), int8(col)
        new_hash = _switch_player_hash(self.current_hash)
        new_hash = _switch_piece_hash(
            curr_row, curr_col, self.player, EMPTY, new_hash
        )
        while (
            _is_in_bounds(curr_row, curr_col)
            and _piece_to_player(self.board[curr_row][curr_col]) == self.player
        ):
            count_similar += 1
            curr_row += d_row
            curr_col += d_col

        if (
            count_similar == 0
            or count_similar > _MAX_SIMILAR_PUSH
            or not _is_in_bounds(curr_row, curr_col)
        ):
            return False

        if self.board[curr_row, curr_col] == EMPTY:
            new_hash = _switch_piece_hash(
                curr_row, curr_col, EMPTY, self.player, new_hash
            )
            return _check_not_repeat(self.past_state_hashes, new_hash)

        count_opposing = 0
        opponent = other_player(self.player)
        new_hash = _switch_piece_hash(
            curr_row, curr_col, opponent, self.player, new_hash
        )

        while (
            _is_in_bounds(curr_row, curr_col)
            and _piece_to_player(self.board[curr_row, curr_col]) == opponent
        ):
            count_opposing += 1
            curr_row += d_row
            curr_col += d_col

        if count_opposing >= count_similar:
            return False

        if not _is_in_bounds(curr_row, curr_col):
            return _check_not_repeat(self.past_state_hashes, new_hash)

        if self.board[curr_row, curr_col] != EMPTY:
            return False

        new_hash = _switch_piece_hash(
            curr_row, curr_col, EMPTY, opponent, new_hash
        )
        return _check_not_repeat(self.past_state_hashes, new_hash)

    def legal_moves(self):
        legal_moves = np.empty((0, 3), dtype=np.int8)

        if self.status != ONGOING:
            return legal_moves

        temp_legal_moves = np.zeros(
            (_ACTUAL_POSSIBLE_MOVE_AMOUNT, 3), dtype=np.int8
        )
        player_index = _player_to_player_index(self.player)

        piece_start = _PIECE_START_RANGES[player_index]
        piece_end = _PIECE_END_RANGES[player_index]

        count = 0
        for piece_index in range(piece_start, piece_end):
            for dir_num in range(_NUM_DIRECTIONS):
                position = self.active_positions[piece_index]
                row_i = position[0]
                if row_i < 0:
                    continue

                col_i = position[1]
                is_legal = self._is_legal_move(row_i, col_i, dir_num)
                if is_legal:
                    temp_legal_moves[count, int8(0)] = int8(row_i)
                    temp_legal_moves[count, int8(1)] = int8(col_i)
                    temp_legal_moves[count, int8(2)] = int8(dir_num)
                    count += 1

        legal_moves = np.empty((count, 3), dtype=np.int8)
        legal_moves[:count, :] = temp_legal_moves[:count, :]
        return legal_moves

    def legal_moves_idx_filter(self):
        legal_moves = self.legal_moves()
        filter_size = TECHNIAL_MOVE_AMOUNT
        np_filter = np.zeros(filter_size, dtype=np.bool)
        for move in legal_moves:
            np_filter[move] = True
        return np_filter

    def _make_move_function(self, row, col, dir_num):
        undo_idx = self.undo_move_stack.add_move(row, col, dir_num)
        d_row = _DIRECTIONS[dir_num, 0]
        d_col = _DIRECTIONS[dir_num, 1]

        curr_row, curr_col = row, col
        curr_piece = self._switch_piece(curr_row, curr_col, EMPTY)
        curr_player = _piece_to_player(curr_piece)
        self.undo_move_stack.add_change(
            undo_idx, curr_row, curr_col, curr_piece
        )

        while curr_piece != EMPTY:
            next_row = curr_row + d_row
            next_col = curr_col + d_col

            if not _is_in_bounds(next_row, next_col):
                self._capture_piece(curr_piece)
                self.undo_move_stack.add_captured_piece(undo_idx, curr_piece)
                break

            next_player = _piece_to_player(self.board[next_row, next_col])
            if curr_player != next_player:
                next_piece = self._switch_piece(next_row, next_col, curr_piece)
                self.undo_move_stack.add_change(
                    undo_idx, next_row, next_col, next_piece
                )
                curr_piece, curr_player = next_piece, next_player

            curr_row, curr_col = next_row, next_col

        self.switch_player()
        if self.undo_move_stack.get_change_amount(undo_idx) > 3:
            print(self)
            raise Exception("oh no")

    def make_move(self, row, col, dir_num, check_legal=True):
        row, col, dir_num = int8(row), int8(col), int8(dir_num)
        if check_legal:
            is_legal = self._is_legal_move(row, col, dir_num)
            if not is_legal:
                return

        self._make_move_function(row, col, dir_num)

        if self.current_hash in self.past_state_hashes:
            self.past_state_hashes[self.current_hash] += int8(1)
        else:
            self.past_state_hashes[self.current_hash] = int8(1)

    def _restore_piece(self, captured_piece):
        self.status = ONGOING
        player = _piece_to_player(captured_piece)
        if player == BLACK:
            self.captures[WHITE_INDEX] -= 1

        if player == WHITE:
            self.captures[BLACK_INDEX] -= 1

    def _undo_last_move(self):
        undo_idx = self.undo_move_stack.get_move_amount() - 1
        self.switch_player()

        change_amount = self.undo_move_stack.get_change_amount(undo_idx)
        for change_idx in range(change_amount - 1, -1, -1):
            row, col, piece = self.undo_move_stack.get_change(
                undo_idx, change_idx
            )
            self._switch_piece(row, col, piece)

        captured_piece = self.undo_move_stack.get_captured_piece(undo_idx)
        if captured_piece != EMPTY:
            self._restore_piece(captured_piece)

        self.undo_move_stack.remove_last_move()

    def undo_move(self, amount=1):
        for _ in range(amount):
            if self.undo_move_stack.is_empty():
                break

            if self.current_hash in self.past_state_hashes:
                if self.past_state_hashes[self.current_hash] <= 1:
                    self.past_state_hashes.pop(self.current_hash)
                else:
                    self.past_state_hashes[self.current_hash] -= 1

            self._undo_last_move()

    def abalone_heuristic(self, res_player):
        res = self.status
        if self.status == ONGOING:
            res = (
                self.captures[BLACK_INDEX] - self.captures[WHITE_INDEX]
            ) / _MAX_CAPTURES

        if res_player == WHITE:
            res = -res
        return res

    def rollout(self, maxDepth=-1):
        res_player = self.player
        count = 0
        while self.status == ONGOING and maxDepth != 0:
            maxDepth = -1 if maxDepth < 0 else maxDepth - 1
            row, col, dir_num = self.get_random_legal_move()
            self.make_move(row, col, dir_num, False)
            count += 1

        result = self.abalone_heuristic(res_player)
        self.undo_move(count)
        return result

    def encode(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for row_i in range(BOARD_SIZE):
            for col_i in range(BOARD_SIZE):
                cell = self.board[row_i][col_i]
                player_index = piece_to_player_index(cell)
                if player_index < 0:
                    continue
                board[player_index, row_i, col_i] = np.int8(1)

        black_captures = np.int8(self.captures[BLACK_INDEX])
        white_captures = np.int8(self.captures[WHITE_INDEX])
        current_player = np.int8(self.player)
        return board, black_captures, white_captures, current_player

    def get_random_legal_move(self):
        moves = self.legal_moves()
        index = int8(np.random.randint(0, int8(len(moves))))
        move = moves[index]
        row = move[0]
        col = move[1]
        dir_num = move[2]
        return row, col, dir_num


def request_user_input():
    print("Enter your move in the format: 'A1 E' where:")
    print(
        " - The first token is the cell (e.g. A1) with a letter for the column and a number for the row."
    )
    print(" - The second token is the direction (e.g. E, W, SW, etc.).")
    print("Possible directions:", list(_DIR_DICT.keys()))

    input_val = input("Your move: ")
    inputs = input_val.split(" ")
    if len(inputs) < 2:
        raise ValueError("Not enough input tokens.")

    # Parse cell position: for example, "A1"
    pos = inputs[0]
    col = ord(pos[0].upper()) - ord("A")
    # Assuming user enters rows as 1-indexed; convert to 0-indexed
    row = int(pos[1]) - 1

    # Use _DIR_DICT to convert direction string to its corresponding index.
    dir_str = inputs[1].upper()
    if dir_str not in _DIR_DICT:
        raise ValueError(f"Direction '{dir_str}' not recognized.")
    dir_num = _DIR_DICT[dir_str]

    return row, col, dir_num


def test_two():
    game = Abalone(True)
    print(game.to_string())

    while game.status == ONGOING:
        row, col, dir_num = request_user_input()
        row, col, dir_num = int8(row), int8(col), int8(dir_num)
        game.make_move(row, col, dir_num)
        print(game.to_string())


@njit
def test_one():
    game = Abalone(True)

    game_state_list = List()
    game_state_list.append(game.copy(True))
    print(game.to_string())

    while game.status == ONGOING:
        row, col, dir_num = game.get_random_legal_move()
        game.make_move(row, col, dir_num)
        # if not compare_hashes(game):
        #     return
        rollout_res = game.rollout(100)
        print("rollout", rollout_res)
        print(game.to_string())
        game_state_list.append(game.copy())

    print(f"amount of moves {game.undo_move_stack.get_move_amount()}")

    while len(game_state_list) > 0:
        copy = game_state_list.pop()
        if not game.compare_board(copy):
            print("shit")
            print(game.to_string())
            print(copy.to_string())
            if game.status == copy.status:
                print(game.status, copy.status)
            if game.player != copy.player:
                print(game.player, copy.player)
            if game.past_state_hashes != copy.past_state_hashes:
                print(game.past_state_hashes, copy.past_state_hashes)
            if not np.array_equal(game.captures, copy.captures):
                print(game.captures, copy.captures)
            if not np.array_equal(
                game.active_positions, copy.active_positions
            ):
                print(game.active_positions == copy.active_positions)
            break
        # print(game)

        game.undo_move()


if __name__ == "__main__":
    # test_one()

    game = Abalone(True)
    game_state = game.encode()
