import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, Union
import wandb
import yaml

from dsvae.data.loader import NoteSequenceDataLoader
from dsvae.models.vae import VAE
from dsvae.utils import (
    get_device,
    init_seed,
    init_logger,
    linear_anneal,
    reconstruction_loss,
)


def train(run_name: str, hparams: Dict[str, Union[str, int, float, bool]], logger: logging.Logger):
    
    save_dir = Path(f"outputs/models/{run_name}")
    if not save_dir.is_dir():
        os.mkdir(save_dir)

    device = get_device(hparams)
    logger.info(f"Using device {device}")
    logger.info(f"hyperparameters: \n{hparams}")

    # data
    loaders = dict()
    num_batches = dict()

    path_to_data = Path(os.environ["DATA_SOURCE_DIR"])
    if bool(int(os.environ["DEBUG"])):
        logger.info('DEBUG')
        path_to_data = Path(os.environ["DATA_SOURCE_DIR"]) / 'test'
    else:
        path_to_data = Path(os.environ["DATA_SOURCE_DIR"]) / 'full'
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
        num_batches[split] = len([x for x in loaders[split]])
    logger.info(f"Batches per split: {num_batches}")
    logger.info(f"Data loader is using {hparams.num_workers} worker threads")

    # model
    model = VAE(hparams, loaders["train"].channels)
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
        teacher_force_ratio = torch.tensor(
            0.8 * linear_anneal(epoch, hparams.max_anneal) + 0.2,
            dtype=torch.float,
            device=device,
        )
        logger.info(f"Teacher forcing ratio is {teacher_force_ratio}")

        # beta_factor we need an inverse anneal
        # we also force beta_factor to equal 1 in the first three epochs 
        beta_threshold = 1e-3
        if epoch < 3:
            beta_factor = torch.tensor(
                1, dtype=torch.float, device=device
            )
        else:
            beta_factor = torch.tensor(
                hparams.beta * (1 - linear_anneal(epoch, hparams.warm_latent)),
                dtype=torch.float,
                device=device,
            )
        logger.info(f"Beta factor (KL-divergence weight) is {beta_factor}")

        # initialize losses
        loss_dict = dict(train=0, valid=0, test=0)
        r_losses = 0.
        z_losses = 0.

        model.train()
        for (input, target, frame_index, filename) in loaders["train"]:
            # forward
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, z, z_loss = model(input, delta_z, teacher_force_ratio)

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
                    output, z, z_loss = model(input, delta_z, teacher_force_ratio)

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
            loss_dict.update({split: loss / num_batches[split]})
        r_losses /= sum(num_batches.values())
        z_losses /= sum(num_batches.values())

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
            if (loss_dict["test"] < best_loss):
                early_stop_count = 0
                # save
                best_loss = loss_dict["test"]

                if bool(int(os.environ["DEBUG"])):
                    logger.warning("Model will not be saved in DEBUG mode")
                else:
                    logger.info(f"Saving model snapshot to {save_dir}/latest.pt with loss: {best_loss}")
                    torch.save(model.state_dict(), f"{save_dir}/latest.pt")
            else:
                early_stop_count += 1

            # early stopping
            if (early_stop_count >= hparams.early_stop):
                logger.info(f"Best loss: {best_loss}; model location: {save_dir}/latest.pt")
                # wandb.save('../outputs/models/latest.pt')
                logger.info(f"Reached early stopping threshold of {hparams.early_stop} epochs.")
                wandb.save(f"{save_dir}/latest.pt")
                break
    wandb.save(f"{save_dir}/latest.pt")
    wandb.save(f"{save_dir}/config.yml")
    logger.info("Reached maximum number of epochs")


def resume(run_name, logger):
    raise NotImplementedError
