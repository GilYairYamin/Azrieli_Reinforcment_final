import os
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
from abalone_neural_network import (
    AbaloneNetwork,
    convert_encoded_board_to_tensors,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AbalonePretrainingDataset(Dataset):
    def __init__(self, pickle_file):
        # Load the DataFrame from the pickle file
        self.data = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        board_state = row["board_state"]
        legal_move_mask = row["legal_move_mask"]
        target_policy = row["policy"]
        target_value = row["value"]

        board_tensor, extra_tensor, legal_move_tensor = (
            convert_encoded_board_to_tensors(board_state, legal_move_mask)
        )
        target_policy = torch.tensor(target_policy, dtype=torch.float64)
        target_value = torch.tensor([target_value], dtype=torch.float64)

        return (
            board_tensor,
            extra_tensor,
            legal_move_tensor,
            target_policy,
            target_value,
        )


class AbalonePUCTDataset(Dataset):
    def __init__(self, pickle_file):
        # Load the DataFrame from the pickle file
        self.data = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        board_state = row["board_state"]
        legal_move_mask = row["legal_move_mask"]
        target_policy = row["policy"]
        target_value = row["value"]

        board_tensor, extra_tensor, legal_move_tensor = (
            convert_encoded_board_to_tensors(board_state, legal_move_mask)
        )
        target_policy = torch.tensor(target_policy, dtype=torch.float64)
        target_value = torch.tensor([target_value], dtype=torch.float64)

        return (
            board_tensor,
            extra_tensor,
            legal_move_tensor,
            target_policy,
            target_value,
        )


def train_network(
    model: AbaloneNetwork,
    dataloader,
    num_epochs=10,
    learning_rate=1e-3,
    device="cpu",
):
    # Move the model to the desired device
    model.to(device)
    # Set the model in training mode
    model.train()

    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(num_epochs), desc="epochs"):
        epoch_loss = 0.0
        for batch in dataloader:
            (
                board_tensor,
                extra_tensor,
                legal_move_mask,
                target_policy,
                target_value,
            ) = batch

            board_tensor = board_tensor.to(device)
            extra_tensor = extra_tensor.to(device)
            legal_move_mask = legal_move_mask.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)

            optimizer.zero_grad()

            policy, value = model(board_tensor, extra_tensor, legal_move_mask)

            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8))

            value_loss = mse_loss(value, target_value)

            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.save_model()
    return model


def train_on_pickle_files(
    model,
    data_folder,
    batch_size=32,
    num_epochs=5,
    learning_rate=1e-2,
    device="cpu",
):
    pickle_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(".pickle")
    ]

    for pickle_file in tqdm(pickle_files, desc="files"):
        # tqdm.write(f"\nTraining on file: {pickle_file}\n")
        dataset = AbalonePretrainingDataset(pickle_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = train_network(
            model,
            dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )
    return model


def pre_train_model():
    res_data_folder = os.path.join(os.getcwd(), "local_data", "training data")
    model = AbaloneNetwork(load_model=True)

    trained_model = train_on_pickle_files(
        model,
        res_data_folder,
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-2,
        device="cuda",
    )

    trained_model.save_model()


if __name__ == "__main__":
    pre_train_model()
