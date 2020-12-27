import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, Union
import wandb

from dsvae.data.loader import NoteSequenceDataLoader
from dsvae.models.vae import VAE
from dsvae.utils import (
    get_device,
    get_hparams,
    init_seed,
    init_logger,
    linear_anneal,
    reconstruction_loss,
)


logger = logging.getLogger(__name__)


def train(hparams: Dict[str, Union[str, int, float, bool]]):
    # ops
    device = get_device(hparams)
    logger.info(f"Using device {device}")
    logger.info(f"Using hyperparameters: \n{hparams}")

    run = wandb.init(
        dir="outputs",
    )
    for k, v in hparams.__dict__.items():
        wandb.config[k] = v

    # data
    loaders = dict()
    lengths = dict()
    logger.info(f'Loading data from {os.environ["DATA_SOURCE_DIR"]}')
    for split in ["train", "valid", "test"]:
        loaders[split] = NoteSequenceDataLoader(
            path_to_data=Path(os.environ["DATA_SOURCE_DIR"]),
            batch_size=hparams.batch_size,
            split=split,
            shuffle=True,
            num_workers=hparams.num_workers,
        )
        lengths[split] = len([x for x in loaders[split]])
    logger.info(f'Data loader is using {hparams.num_workers} worker threads')

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
        # TODO: Implement in VAE
        teacher_force_ratio = torch.tensor(
            1.0 * linear_anneal(epoch, hparams.max_anneal),
            dtype=torch.float,
            device=device,
        )
        logger.info(f"Teacher forcing ratio is {teacher_force_ratio}")
        beta_factor = torch.tensor(
            hparams.beta * linear_anneal(epoch, hparams.max_anneal),
            dtype=torch.float,
            device=device,
        )
        logger.info(f"Beta factor (KL-divergence weight) is {beta_factor}")

        # initialize losses
        loss_dict = dict(train=0, valid=0, test=0)

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
                    r_loss = reconstruction_loss(
                        output, target, loaders[split].channels
                    )
                    loss = (r_loss + z_loss) / hparams.batch_size
                    loss_dict[split] += loss.item()

        for split, loss in loss_dict.items():
            loss_dict.update({split: loss / lengths[split]})
            # loss /= lengths[split]

        scheduler.step(loss_dict["valid"])

        # monitoring
        # TODO: Add more evaluation metrics
        for k, v in loss_dict.items():
            k = f"{k}_loss"
            wandb.log({k: v})

        # best model
        if loss_dict["test"] < best_loss:
            training_step = 0
            best_loss = loss_dict["test"]

            save_dir = Path(f"outputs/models/{run.name}")
            if not save_dir.is_dir():
                os.mkdir(save_dir)
            save_path = f"{save_dir}/latest.pt"

            logger.info(f"Saving model snapshot to {save_path} with loss: {best_loss}")
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_count += 1

        # early stopping
        if early_stop_count >= hparams.early_stop:
            logger.info(f"Best loss: {best_loss}; model location: {save_path}")
            logger.info(f"Reached early stopping threshold of {training_step} epochs.")
            break
    logger.info("Reached maximum number of epochs")


if __name__ == "__main__":
    init_logger()
    hparams = get_hparams()
    train(hparams)
