import logging
import numpy as np
import os
from pathlib import Path
import random
import torch


def init_logger(level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger()
    if not len(logger.handlers):
        logger.setLevel(level)
        handler = logging.StreamHandler()
        format = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        handler.setFormatter(format)
        handler.setLevel(level)
        logger.addHandler(handler)
    return logger


def init_seed(hparams):
    if hparams.deterministic:
        SEED = 1234
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(hparams):
    if hparams.device == "cpu":
        device = torch.device("cpu")
    else:
        try:
            device = torch.device(torch.cuda.current_device())
            torch.backends.cudnn.benchmark = True  # Enable CuDNN optimization
        except RuntimeError:
            device = torch.device("cpu")
    device_properties = torch.cuda.get_device_properties(device)
    logging.getLogger(__name__).info(
        f"Using device:\nName: {device_properties.name}\nVRAM: {device_properties.total_memory}\nProcessor_count: {device_properties.multi_processor_count}"
    )
    return device
