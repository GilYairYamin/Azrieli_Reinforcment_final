import os
from datetime import datetime

import torch
import torch.nn as nn

# import torch.optim as optim
from abalone_numba import Abalone, TECHNIAL_MOVE_AMOUNT, VALID_BOARD_MASK


class AbaloneNetwork(nn.Module):
    CURR_DIR_PATH = os.getcwd()
    WEIGHTS_DIR_PATH = os.path.join(CURR_DIR_PATH, "model_weights")
    SNAPSHOTS_DIR_PATH = os.path.join(WEIGHTS_DIR_PATH, "snapshots")
    WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_DIR_PATH, "current_weights.pth")

    def __init__(self, load_model: bool = True):
        super(AbaloneNetwork, self).__init__()

        self.board_mask = VALID_BOARD_MASK.clone().flatten()
        num_valid_cells = self.board_mask.flatten().sum().item()

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

    def forward(self, game: Abalone):
        (board, player, black_captures, white_captures) = game.encode()

        x = torch.tensor(
            board, device=next(self.parameters()).device, dtype=torch.float32
        )

        x = x.unsqueeze(0)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        batch_size = x.size(0)
        x = x.view(batch_size, 32, -1)
        x = x[:, :, self.board_mask]
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))

        extra = torch.tensor(
            [player, black_captures, white_captures],
            device=x.device,
        ).unsqueeze(0)

        extra = extra.expand(batch_size, -1)
        x = torch.cat([x, extra], dim=1)

        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        policy_logits = self.policy_head(x)

        policy = self.softmax(policy_logits).flatten()
        value = self.tanh(self.value_head(x)).item()

        return policy, value

    def save_model(self):
        os.makedirs(self.WEIGHTS_DIR_PATH, exist_ok=True)
        os.makedirs(self.SNAPSHOTS_DIR_PATH, exist_ok=True)

        torch.save(self.state_dict(), self.WEIGHTS_FILE_PATH)

        current_time = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
        snapshot_name = f"weights_{current_time}.pth"
        file_path = os.path.join(self.SNAPSHOTS_DIR_PATH, snapshot_name)
        torch.save(self.state_dict(), file_path)

    def load_model(self):
        try:
            self.load_state_dict(torch.load(self.WEIGHTS_FILE_PATH))
        except Exception:
            print("failed")
        self.eval()


# בדיקה בסיסית
if __name__ == "__main__":
    model = AbaloneNetwork()

    game = Abalone()
    policy, value = model.forward(game)

    print("Policy Output Shape:", policy.shape)
    print("Value Output:", value)

    print(policy[0].item())
    model.save_model()
