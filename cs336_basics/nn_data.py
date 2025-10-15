import numpy as np
import torch
np.random.seed(100)

def get_batch(dataset, batch_size, context_length, device):
    n = len(dataset)
    inputs = np.empty((batch_size, context_length), dtype=np.uint16)
    targets = np.empty((batch_size, context_length), dtype=np.uint16)
    for i in range(batch_size):
        start = np.random.randint(0, n - context_length)
        end = start + context_length
        inputs[i] = dataset[start:end]
        targets[i] = dataset[start+1:end+1]
    return torch.from_numpy(inputs).long().to(device), torch.from_numpy(targets).long().to(device)

def get_batch_seq(dataset, batch_size, context_length, device):
    n = len(dataset)
    inputs = np.empty((batch_size, context_length), dtype=np.uint16)
    targets = np.empty((batch_size, context_length), dtype=np.uint16)
    assert n >= (batch_size * context_length)
    assert batch_size > 0
    assert context_length > 0

    start = 0

    while start + batch_size * context_length + 1 < n:
        s = start
        for i in range(batch_size):
            inputs[i] = dataset[s:s+context_length]
            targets[i] = dataset[s+1: s+context_length+1]
            s += context_length

        yield torch.from_numpy(inputs).long().to(device), torch.from_numpy(targets).long().to(device)
        start += batch_size * context_length

    remaining_batch = (n - start - 1) // context_length
    if remaining_batch > 0:
        inputs = dataset[start:start+remaining_batch*context_length].reshape(remaining_batch, context_length).copy()
        targets = dataset[start+1:start+remaining_batch*context_length+1].reshape(remaining_batch, context_length).copy()
        yield torch.from_numpy(inputs).long().to(device), torch.from_numpy(targets).long().to(device)