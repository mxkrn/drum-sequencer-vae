from dsvae.utils.hparams import AttrDict, process, get_hparams


def test_attr_dict():
    expected = "bar"

    attr_dict = AttrDict(foo=expected)
    assert attr_dict.foo == expected


def test_get_hparams():
    hparams = get_hparams()

    expected = dict(
        dataset="gmd",
        num_workers=0,
        batch_size=4,
        channels=9,  # number of instruments
        sequence_length=16,
        file_shuffle=True,  # shuffles data loading across different MIDI patterns
        pattern_shuffle=True,  # shuffle sub-patterns within a MIDI pattern
        scale_factor=2,
        model="vae",
        bidirectional=False,
        n_layers=2,
        hidden_size=512,
        latent_size=8,
        lstm_dropout=0.1,
        # teacher_force_ratio=0.0,
        beta=1e4,
        max_anneal=200,
        attention=False,
        disentangle=False,
        epochs=300,
        lr=1e-4,
        warm_latent=100,
        early_stop=30,
        device="",
    )
    for k, v in expected.items():
        assert hparams[k] == v


def test_process():
    hparams = AttrDict(
        n_layers=2,
        bidirectional=True,
        channels=9,
        epochs=100,
        max_anneal=50,
        warm_latent=50,
        sequence_length=16,
    )
    hparams = process(hparams)
    assert hparams.hidden_factor == 4
    assert hparams.input_size == 27
    assert hparams.input_shape == (hparams.sequence_length, 27)
