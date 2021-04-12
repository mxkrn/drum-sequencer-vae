import logging
import os
from pathlib import Path
import torch
from typing import Dict, Union
import wandb

from dsvae.data.loader import NoteSequenceDataLoader
from dsvae.models.vae import VAE, TrainTask
from dsvae.utils import (
    get_device,
    get_hparams,
    init_logger,
    linear_anneal,
    reconstruction_loss,
)


def train(hparams: Dict[str, Union[str, int, float, bool]], logger: logging.Logger):
    device = get_device(hparams)

    # data
    loaders = dict()
    lengths = dict()

    path_to_data = Path(os.environ["DATA_SOURCE_DIR"])
    if DEBUG:
        path_to_data = Path(os.environ["DATA_SOURCE_DIR"]) / "test"
    else:
        path_to_data = Path(os.environ["DATA_SOURCE_DIR"]) / "full"
    if path_to_data.is_dir():
        logger.info(f"Loading data from {path_to_data}")
    else:
        f"Invalid path to data {path_to_data}"

    for split in ["train", "valid", "test"]:
        loaders[split] = NoteSequenceDataLoader(
            path_to_data=path_to_data,
            batch_size=hparams.batch_size,
            split=split,
            file_shuffle=hparams.file_shuffle,
            pattern_shuffle=hparams.pattern_shuffle,
            scale_factor=hparams.scale_factor,
            num_workers=hparams.num_workers,
        )
        lengths[split] = len([x for x in loaders[split]])
    logger.info(f"Batches per split: {lengths}")
    logger.info(f"Data loader is using {hparams.num_workers} worker threads")

    # model
    model = VAE(hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, verbose=True, threshold=1e-7
    )
    model = model.to(device)

    # run
    best_loss = 1e10
    early_stop_count = 0
    logger.info("Starting training...")

    # constants
    delta_z = torch.zeros(
        (hparams.batch_size, hparams.latent_size), dtype=torch.float, device=device
    )

    for epoch in range(hparams.epochs):

        logger.info(f"Epoch {epoch}")

        # annealing
        if hparams.task == str(TrainTask.SYNCOPATE):
            # We anneal the teacher force ratio
            teacher_force_ratio = torch.tensor(
                0.8 * linear_anneal(epoch, hparams.max_anneal) + 0.2,
                dtype=torch.float,
                device=device,
            )
        elif hparams.task == str(TrainTask.GROOVE):
            # By allowing the decoder to learn to copy directly to its outputs,
            # we incentivize this encoder to ignore onsets  only learn a useful
            # representation for generating velocities and offsets.
            teacher_force_ratio = torch.tensor(1.0, dtype=torch.float, device=device)

        logger.info(f"Teacher forcing ratio is {teacher_force_ratio}")

        # beta_factor we need an inverse anneal
        # we also force beta_factor to equal 1 in the first three epochs
        if epoch < 3:
            beta_factor = torch.tensor(1, dtype=torch.float, device=device)
        else:
            beta_factor = torch.tensor(
                hparams.beta * (1 - linear_anneal(epoch, hparams.warm_latent)),
                dtype=torch.float,
                device=device,
            )
        logger.info(f"Beta factor (KL-divergence weight) is {beta_factor}")

        # initialize losses
        loss_dict = dict(train=0, valid=0, test=0)
        r_losses = 0.0
        z_losses = 0.0

        model.train()
        for (input, target, frame_index, filename) in loaders["train"]:
            # forward
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            onsets, offsets, velocities, z, z_loss = model(
                input, delta_z, teacher_force_ratio
            )
            output = torch.cat((onsets, velocities, offsets), -1)

            # loss
            z_loss *= beta_factor  # scale the KL-divergence
            r_loss = reconstruction_loss(output, target, loaders["train"].channels)
            loss = (r_loss + z_loss) / hparams.batch_size
            loss_dict["train"] += loss.item()
            r_losses += r_loss.item()
            z_losses += z_loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for split in ["valid", "test"]:
            with torch.no_grad():
                for (input, target, frame_index, filename) in loaders[split]:
                    # forward
                    input = input.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    onsets, velocities, offsets, z, z_loss = model(
                        input, delta_z, teacher_force_ratio
                    )
                    output = torch.cat((onsets, velocities, offsets), -1)

                    # loss
                    z_loss *= beta_factor  # scale the KL-divergence
                    z_losses += z_loss.item()
                    r_loss = reconstruction_loss(
                        output, target, loaders[split].channels
                    )
                    r_losses += r_loss.item()
                    loss = (r_loss + z_loss) / hparams.batch_size
                    loss_dict[split] += loss.item()

        # normalize losses
        for split, loss in loss_dict.items():
            loss_dict.update({split: loss / lengths[split]})
        r_losses /= sum(lengths.values())
        z_losses /= sum(lengths.values())

        scheduler.step(loss_dict["valid"])

        # monitoring and logging
        # TODO: Add more evaluation metrics
        monitor_dict = dict()
        for k, v in loss_dict.items():
            k = f"{k}_loss"
            monitor_dict[k] = v
        monitor_dict["r_loss_avg"] = r_losses
        monitor_dict["z_loss_avg"] = z_losses

        wandb.log(monitor_dict)
        for k, v in monitor_dict.items():
            logger.info(f"{k}: {v}")

        # best model
        if epoch >= hparams.max_anneal - 1:
            if loss_dict["test"] < best_loss:
                early_stop_count = 0
                # save
                best_loss = loss_dict["test"]
                save_dir = Path(f"outputs/models/{run.name}")
                if not save_dir.is_dir():
                    os.mkdir(save_dir)
                save_path = f"{save_dir}/latest.pt"

                if DEBUG:
                    logger.warning("Model will not be saved in debug mode")
                else:
                    logger.info(
                        f"Saving model snapshot to {save_dir}/latest.pt with loss: {best_loss}"
                    )
                    torch.save(model.state_dict(), f"{save_dir}/latest.pt")
            else:
                early_stop_count += 1

            # early stopping
            if early_stop_count >= hparams.early_stop:
                logger.info(
                    f"Best loss: {best_loss}; model location: {save_dir}/latest.pt"
                )
                # wandb.save('../outputs/models/latest.pt')
                logger.info(
                    f"Reached early stopping threshold of {hparams.early_stop} epochs."
                )
                wandb.save(f"{save_dir}/latest.pt")
                break
    logger.info("Reached maximum number of epochs")


if __name__ == "__main__":
    logger = init_logger(logging.DEBUG)
    hparams = get_hparams(filename=str(Path.cwd() / "config/groovae.yml"))
    global DEBUG
    if bool(hparams.debug):
        DEBUG = bool(hparams.debug)
    else:
        DEBUG = False

    # ops
    if DEBUG:
        run = wandb.init(dir="outputs", mode="offline")
        # logger = init_logger(logging.DEBUG)
        logger.debug("Running train.py in debug mode")
    else:
        run = wandb.init(
            dir="outputs",
        )
        logger = init_logger(logging.INFO)

    # get hparams and add to config
    wandb.config.update(hparams, allow_val_change=True)

    # run
    train(hparams, logger)
