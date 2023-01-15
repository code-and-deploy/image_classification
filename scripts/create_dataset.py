import pandas as pd
import argparse
import os, torch, time

import wandb
from min_dalle import MinDalle

from haven import haven_wizard as hw
from haven import haven_utils as hu




if __name__ == "__main__":
    model = MinDalle(
        models_root='./data/pretrained',
        dtype=torch.float32,
        device='cuda',
        is_mega=False, 
        is_reusable=True
    )

    for label in ["cats", "dogs"]:
        image = model.generate_image(
            text='Nuclear explosion broccoli',
            seed=-1,
            grid_size=1,
            is_seamless=False,
            temperature=1,
            top_k=256,
            supercondition_factor=32,
            is_verbose=False
        )

        print()