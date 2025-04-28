import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch

from abalone_puct import simulate_games_processes, simulate_games_single
from clean_generated_data import build_training_data_from_puct
from train_model import train_on_pickle_file


def train_the_model(
    num_iterations=5, num_games_per_iteration=50, max_depth=50, executor=None
):
    root_folder = os.path.join(os.getcwd(), "local_data", "puct_data")
    training_data_folder = os.path.join(root_folder, "raw_training_data")
    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(training_data_folder, exist_ok=True)

    for iteration in range(num_iterations):
        now = datetime.now()
        readable_time = now.strftime("%d.%m.%Y_%H-%M-%S")

        res_folder = os.path.join(
            root_folder, f"iteration-{iteration}_{readable_time}"
        )

        if executor is None:
            simulate_games_single(
                res_folder, num_games_per_iteration, max_depth
            )
        else:
            simulate_games_processes(
                res_folder, executor, num_games_per_iteration, max_depth
            )

        now = datetime.now()
        readable_time = now.strftime("%d.%m.%Y_%H-%M-%S")
        res_file_name = os.path.join(
            training_data_folder, f"iteration-{iteration}_{readable_time}"
        )

        build_training_data_from_puct(res_folder, res_file_name)
        train_on_pickle_file(res_file_name)


if __name__ == "__main__":
    # train_the_model(100, 10, 150)
    print("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(max_workers=2) as executor:
        train_the_model(100, 10, 150, executor)
