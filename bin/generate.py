from argparse import ArgumentParser, Namespace
from pathlib import Path


def main(run_name: str, midi_file: str):
    assert Path(midi_file).is_file()


def sigmoid(array, gradient=1):
    sigmoid = 1 / (1 + np.exp(np.multiply(array, -gradient)))
    curve = np.subtract(sigmoid, 0.5)
    return np.multiply(curve, 2)


def threshold(array, threshold=0.5):
    ones = np.array(1.0)
    zeros = np.array(0.0)
    output = np.where(array >= threshold, ones, zeros)
    return output


def onnx_predict(
    input: torch.Tensor,
    delta_z: torch.Tensor,
    note_dropout: torch.Tensor,
    onnx_model_path: str,
):
    session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {
        "input": input.detach().cpu().numpy(),
        "delta_z": delta_z.cpu().numpy(),
        "note_dropout": note_dropout.unsqueeze(0).cpu().numpy(),
    }
    output = session.run(None, ort_inputs)
    return output[0], output[1], output[2], output[3], output[4]


def get_hparams(run_path):

    config = yaml.load(
        wandb.restore(f"config.yaml", run_path=run_path, replace=True),
        Loader=yaml.FullLoader,
    )
    hparams = {}

    for k, v in config.items():
        if not k in ["_wandb", "wandb_version"]:
            if isinstance(v, dict):
                hparams[k] = v["value"]
    hparams = AttrDict(hparams)
    hparams.batch_size = BATCH_SIZE
    return hparams


def get_onnx_model_path(run_name: str):
    onnx_model_path = Path.cwd() / f"outputs/models/{run_name}/{run_name}.onnx"
    assert load_path.is_file()
    print(f"Loading ONNX model from: {load_path}")
    model = onnx.load(load_path)
    onnx.checker.check_model(model)
    return load_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
    )
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--midi_file", type=str)
