import torch

checkpoint = torch.load('Prima_training_and_evaluation/ckpts/fullmodel107.pt', map_location='cpu', weights_only=False)
