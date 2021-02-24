# drum-sequencer-vae

This repository exists to contain the training process of several drum sequencer features.

1. GrooVAE - learns to add expression to quantized drum rhythms
2. Syncopate - learns to generate syncopated variations of drum rhythms
3. Fills - the network can be conditioned on beat type to selectively generate fills or beats

## Installation

Create the conda environment:

    conda env update -f environment.yml

This will create an environment named `dsvae`. Activate the
environment as follows:

    conda activate dsvae

Inside the `dsvae` conda environment, the `dsvae`
package can be installed in [editable
mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
as follows. From the same directory as this README, issue the
following command:

    pip install -e .

## Developer environment

#### Secrets management
One option for maintaining sensitive data is to store values as environment variables in a `.env` file and use the `direnv` program to source this file each time you enter the repo.

We provide a sample file `example.env`, which stores environment variables expected in different parts of the code base. Simply copy this file to `.env` and fill in expected values.

You will also need to install direnv for your system.


## Testing

To run the tests, run:

    pytest
