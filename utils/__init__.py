import pdb
import json
import torch

def k_index(l, v, k):
    """
    Find the index of (k-th) v in l.
    Note that k start from 1
    """
    assert v in l
    assert 0 < k <= l.count(v)
    count = 1
    prefix = 0
    # pdb.set_trace()
    while count < k:
        count += 1
        prefix += l.index(v) + 1
        l = l[l.index(v) + 1: ]
    return prefix + l.index(v)

# k_index([1,2,3,1], 1, 1)


def read_multiline_json(file_path):
    f = open(file_path)
    data = [json.loads(line) for line in f.readlines()]
    f.close()
    return data

def read_multiline_data(file_path):
    f = open(file_path)
    data = f.readlines()
    f.close()
    return data

def avg(l):
    return sum(l) / len(l)


def standalize(t):
    return (t - t.mean(dim=-1).unsqueeze(dim=-1)) / t.std(dim=-1).unsqueeze(dim=-1)


def compute_matrix_scale(m):
    if isinstance(m, torch.Tensor):
        m = m.to(torch.float32)
    return (m**2).sum()**0.5 / m.shape[0]