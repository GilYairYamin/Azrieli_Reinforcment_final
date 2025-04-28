import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# import torch.optim as optim
from abalone import TECHNIAL_MOVE_AMOUNT, VALID_BOARD_MASK, Abalone


def convert_encoded_board_to_tensors(
    board_state, legal_moves_mask, single_state=False, device=None
):
    if device is None:
        device = torch.device("cpu")

    board, black_captures, white_captures, player = board_state
    board_tensor = torch.tensor(board, dtype=torch.float32, device=device)
    extra_arr = np.array(
        [black_captures, white_captures, player],
        dtype=np.float32,
    )
    extra_tensor = torch.tensor(extra_arr, dtype=torch.float32, device=device)

    if single_state:
        board_tensor = board_tensor.unsqueeze(0)
        extra_tensor = extra_tensor.unsqueeze(0)

    legal_move_tensor = torch.tensor(
        legal_moves_mask, dtype=torch.bool, device=device
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

        self.conv1 = nn.Conv2d(
            2, 8, 5, stride=1, padding=2, dtype=torch.float32
        )

        self.fc1 = nn.Linear(num_valid_cells * 8, 64, dtype=torch.float32)
        self.fc2 = nn.Linear(64 + 3, 64, dtype=torch.float32)
        self.relu = nn.ReLU()

        self.policy_head = nn.Linear(
            64, TECHNIAL_MOVE_AMOUNT, dtype=torch.float32
        )

        self.softmax = nn.Softmax(dim=-1)

        self.value_head = nn.Linear(64, 1, dtype=torch.float32)
        self.tanh = nn.Tanh()

        if load_model:
            self.load_model()

    def forward(self, board_tensor, extra_tensor, legal_moves_tensor):
        x = board_tensor
        x = self.relu(self.conv1(x))

        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)
        x = x[:, :, self.board_mask]
        x = x.flatten(start_dim=1)

        x = self.relu(self.fc1(x))

        x = torch.cat([x, extra_tensor], dim=1)

        x = self.relu(self.fc2(x))

        policy_logits = self.policy_head(x)

        largest_negative_torch = torch.finfo(torch.float32).min
        policy_logits = policy_logits.masked_fill(
            ~legal_moves_tensor, largest_negative_torch
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
            print("failed to load model")
        self.eval()


# בדיקה בסיסית
if __name__ == "__main__":
    model = AbaloneNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    game = Abalone(True)
    board_state = game.encode()

    board_tensor, extra_tensor, legal_move_tensor = (
        convert_encoded_board_to_tensors(
            board_state,
            game.get_legal_moves_mask(),
            single_state=True,
            device=device,
        )
    )

    policy, value = model.forward(
        board_tensor, extra_tensor, legal_move_tensor
    )

    print("Policy Output Shape:", policy.shape)
    print("Value Output:", value)

    print(policy.cpu().detach().numpy().flatten())
    model.save_model()
    pass
