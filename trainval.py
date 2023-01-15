import pandas as pd
import argparse
import os, torch, time, exp_configs

from src import datasets, models

from haven import haven_wizard as hw
from haven import haven_utils as hu


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Get Pytorch datasets
    train_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="train"
    )
    val_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="val"
    )

    # Tip: Test dataset to avoid bugs later (Tip: visualize if possible in tmp folder)
    sample = train_set[0]
    # hu.save_image('.tmp/tmp.png', sample[0].reshape((8,8)))

    # Create data loader (Tip: increase number of threads & drop last batch)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, num_workers=0)

    # Get Model
    model = models.get_model(exp_dict=exp_dict, device=args.device)
    
    # Train and Validate
    score_list = []
    for epoch in range(exp_dict['epochs']):
        # Train for one epoch
        s_time = time.time()
        train_dict = model.train_on_loader(train_loader)
        train_time = time.time() - s_time

        # Validate
        val_dict = model.val_on_loader(val_loader)

        # Get Metrics (Tip: add time and size)
        score_dict = {
            "epoch": epoch,
            "train_time": train_time,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "train_acc": train_dict["train_acc"],
            "train_loss": train_dict["train_loss"],
            "val_acc": val_dict["val_acc"],
        }

        # Save Metrics
        score_list += [score_dict]
        hu.save_pkl(os.path.join(savedir, "score_list.pkl"), score_list)
        hu.torch_save(os.path.join(savedir, "model.pth"), model.get_state_dict())

        # Report scores
        print(pd.DataFrame(score_list).tail())
        print()

    print("Experiment done\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()

    # Get list of experiments
    parser.add_argument(
        "-e",
        "--exp_group"
    )

    # Define directory where experiments are saved
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True
    )

    # Reset or resume experiment
    parser.add_argument(
        "-r", "--reset", default=0, type=int
    )

    # Select device (important for those without GPU)
    parser.add_argument(
        "-d", "--device", default='cuda'
    )
    args, others = parser.parse_known_args()

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        results_fname="results/results.ipynb",
        args=args,
    )