import math
import torch
import pickle


def mt_sampler(data, y, preds, size, top_k=True):
    # Given size is ratio of training data
    if math.isclose(size, 1.0):
        return data, y, None, None
    elif type(size) == float:
        n = int(size * len(data))
    # Given size is actual number of training data
    else:
        n = int(size)

    # mt sampling (returns indices)
    dif = torch.sum(torch.abs(y-preds), 1)
    if top_k:
        _, idx = torch.topk(dif, n)
    else:
        idx = torch.randperm(len(data))[:n]

    # get sampled data
    sampled_data = data[idx]
    sampled_y = y[idx]
    
    return sampled_data, sampled_y, idx, dif


def save_samples(sample_history, step, max_steps, samples, file_name):
    sample_history[str(step)] = samples.detach().cpu().numpy()
    with open(file_name, 'wb') as f:
        pickle.dump(sample_history, f)
    print("Sampling history saved at %s." % file_name)


def save_losses(loss_history, step, max_steps, losses, file_name):
    loss_history[str(step)] = losses.detach().cpu().numpy()
    with open(file_name, 'wb') as f:
        pickle.dump(loss_history, f)
    print("Loss history saved at %s." % file_name)