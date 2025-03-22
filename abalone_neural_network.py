import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# import torch.optim as optim
from abalone import Abalone, TECHNIAL_MOVE_AMOUNT, VALID_BOARD_MASK


def convert_encoded_board_to_tensors(board_state, legal_moves_mask, unsqueeze=False):
    board, black_captures, white_captures, player = board_state

    board_tensor = torch.tensor(board, dtype=torch.float32)

    extra_arr = np.array([black_captures, white_captures, player], dtype=np.float32)
    extra_tensor = torch.tensor(
        extra_arr, device=board_tensor.device, dtype=torch.float32
    )

    if unsqueeze:
        board_tensor = board_tensor.unsqueeze(0)
        extra_tensor = extra_tensor.unsqueeze(0)

    legal_move_tensor = torch.tensor(
        legal_moves_mask, device=board_tensor.device, dtype=torch.bool
    )
    return board_tensor, extra_tensor, legal_move_tensor


class AbaloneNetwork(nn.Module):
    CURR_DIR_PATH = os.path.join(os.getcwd(), "local_data")
    WEIGHTS_DIR_PATH = os.path.join(CURR_DIR_PATH, "model_weights")
    WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_DIR_PATH, "current_weights.pth")
    SNAPSHOTS_DIR_PATH = os.path.join(WEIGHTS_DIR_PATH, "snapshots")

    def __init__(self, load_model: bool = True):
        super(AbaloneNetwork, self).__init__()

        self.board_mask = VALID_BOARD_MASK.copy().flatten()
        num_valid_cells = self.board_mask.sum().item()

        self.conv1 = nn.Conv2d(2, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(num_valid_cells * 32, 128)
        self.fc2 = nn.Linear(128 + 3, 128)
        self.fc3 = nn.Linear(128, 256)
        self.relu = nn.ReLU()

        self.policy_head = nn.Linear(256, TECHNIAL_MOVE_AMOUNT)
        self.softmax = nn.Softmax(dim=-1)

        self.value_head = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

        if load_model:
            self.load_model()

    def forward(self, board, extra, legal_move_mask):
        x = board
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        batch_size = x.size(0)
        x = x.view(batch_size, 32, -1)
        x = x[:, :, self.board_mask]
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))

        x = torch.cat([x, extra], dim=1)

        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        policy_logits = self.policy_head(x)

        largest_negative_torch = torch.finfo(torch.float32).min
        policy_logits = policy_logits.masked_fill(
            ~legal_move_mask, largest_negative_torch
        )

        policy = self.softmax(policy_logits)
        value = self.tanh(self.value_head(x))
        return policy, value

    def save_model(self):
        os.makedirs(self.WEIGHTS_DIR_PATH, exist_ok=True)
        os.makedirs(self.SNAPSHOTS_DIR_PATH, exist_ok=True)

        current_time = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
        snapshot_name = f"{current_time}_weights.pth"
        snapshot_path = os.path.join(self.SNAPSHOTS_DIR_PATH, snapshot_name)

        torch.save(self.state_dict(), self.WEIGHTS_FILE_PATH)
        torch.save(self.state_dict(), snapshot_path)

    def load_model(self):
        try:
            self.load_state_dict(torch.load(self.WEIGHTS_FILE_PATH))
        except Exception:
            print("failed")
        self.eval()


# בדיקה בסיסית
if __name__ == "__main__":
    model = AbaloneNetwork()

    game = Abalone(True)

    # policy, value = model.forward()

    # print("Policy Output Shape:", policy.shape)
    # print("Value Output:", value)

    # print(policy[0].item())
    model.save_model()
