from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import wandb

from dsvae.utils import init_logger, AttrDict
from models.vae import VAE


DEVICE = "cuda:0"
RUN_NAME = "autumn-energy-213"
RUN_PATH = "mxkrn/drum-sequencer-vae-dsvae/3729sf5a"
BATCH_SIZE = 10
OPSET_VERSION = 12
LOGGER = init_logger()


def get_dummy_input(
    batch_size: int, sequence_length: int, input_size: int, latent_size: int
) -> torch.Tensor:
    input = torch.zeros(
        (batch_size, sequence_length, input_size), dtype=torch.float, device=DEVICE
    )
    delta_z = torch.zeros(latent_size, dtype=torch.float, device=DEVICE)
    note_dropout = torch.tensor([0.5], dtype=torch.float, device=DEVICE)
    return (input, delta_z, note_dropout)


def get_model(run_path: str, run_name: str) -> nn.Module:
    # get hyperparameters
    config = yaml.load(
        wandb.restore("config.yaml", run_path=run_path, replace=True),
        Loader=yaml.FullLoader,
    )
    hparams = {}
    for k, v in config.items():
        if not k in ["_wandb", "wandb_version"]:
            if isinstance(v, dict):
                hparams[k] = v["value"]
    hparams = AttrDict(hparams)
    hparams.batch_size = BATCH_SIZE

    # load model
    path_to_state_dict = f"outputs/models/{run_name}/latest.pt"
    _state_dict_io = wandb.restore(path_to_state_dict, run_path=RUN_PATH)
    model = VAE(hparams)
    model.load_state_dict(torch.load(_state_dict_io.name))
    model = model.to(DEVICE)

    return model.eval(), path_to_state_dict


def onnx_export(run_path: str, run_name: str) -> None:
    model, path_to_state_dict = get_model(run_path, run_name)

    save_path = Path(path_to_state_dict.replace("latest", run_name)).with_suffix(
        ".onnx"
    )

    dummy_input = get_dummy_input(
        model.batch_size, model.sequence_length, model.input_size, model.latent_size
    )
    LOGGER.info(f"Using dummy input shape: {dummy_input[0].shape}")
    input_names = ["input", "delta_z", "note_dropout"]
    output_names = ["onsets", "velocities", "offsets", "z", "z_loss"]
    assert len(input_names) == len(dummy_input)

    # TODO: Regression tests

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence"},
            "onsets": {0: "batch_size", 1: "sequence"},
            "velocities": {0: "batch_size", 1: "sequence"},
            "velocities": {0: "batch_size", 1: "sequence"},
            "z": {0: "batch_size"},
        },
    )
    LOGGER.info(f"Exported model to {save_path}")


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument(
    #     "--run_name", default=RUN_NAME, type=str, help="Wandb run name of best run"
    # )
    # parser.add_argument(
    #     "--run_path", default=RUN_PATH, type=str, help="Wandb run path of best run"
    # )
    # args = parser.parse_args()
    onnx_export(RUN_PATH, RUN_NAME)
