import os
import random
import sys
import argparse

import numpy as np
import torch
from gym import spaces

from matplotlib import pyplot as plt

from PIL import Image

import policy
import observation_transformers
import environment

import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config

def makedir(directory):
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "both"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    config = get_baselines_config(args.exp_config, args.opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    makedir(config.CHECKPOINT_FOLDER)
    makedir(config.TENSORBOARD_DIR)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_init(config)

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval()
    elif args.run_type == "both":
        trainer.train()
        trainer.eval()