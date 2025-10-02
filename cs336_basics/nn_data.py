import numpy as np
import torch
np.random.seed(100)

def get_batch(dataset, batch_size, context_length, device):
    n = len(dataset)
    inputs = np.empty((batch_size, context_length))
    targets = np.empty((batch_size, context_length))
    for i in range(batch_size):
        start = np.random.randint(0, n - context_length)
        end = start + context_length
        inputs[i] = dataset[start:end]
        targets[i] = dataset[start+1:end+1]
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
