import torch

from dsvae.data.loader import Loader


def test_data_loader(path_to_data):
    batch_size = 4
    dataset_name = "gmd"
    loader = Loader(path_to_data, dataset_name, batch_size)
    for batch in loader:
        assert batch[0].shape == torch.Size([4, 16, 27])
        assert batch[1].shape == torch.Size([4, 16, 27])


# def test_gmd_dataset_constructor(path_to_data: Path, files: List[Path]):
#     ds = GMDMidiStreamDataset(path_to_data)

#     for fname in path_to_data.glob("**/*.json"):
#         with open(fname, "r") as f:
#             pitch_mapping = json.loads(f.read())
#     assert ds.pitch_mapping == pitch_mapping

#     for i, filepath in enumerate(islice(ds.files, 8)):
#         assert filepath.is_file()
#         assert filepath == files[i]

#     batch_size = 4
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
#     frame_indices = torch.tensor([0, 0, 1, 2])
#     file1 = "182_afrocuban_105_fill_4-4.mid"
#     file2 = "183_afrocuban_105_beat_4-4.mid"
#     sample_names = (file1, file2, file2, file2)

#     for f in loader:
#         torch.all(torch.eq(f[2], frame_indices))
#         assert f[3] == sample_names
#         break
