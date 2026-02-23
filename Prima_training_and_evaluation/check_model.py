import torch

checkpoint = torch.load('fullmodel107.pt', map_location='cpu')


if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        print(key)