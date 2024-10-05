import pdb
import torch
from torch import nn
from tqdm import tqdm
from utils import standalize


def softmax_with_temperature(logits, temperature):
    logits = logits / temperature
    return nn.functional.softmax(logits, dim=-1)


def block_cosine_similarity_no_grad(tensor1, tensor2, block_size=100):
    """
    tensor1: (M, d)
    tensor2: (N, d)
    """
    with torch.no_grad():
        size1 = tensor1.size()
        size2 = tensor2.size()
        result = torch.zeros(size1[0], size2[0])
        for i in tqdm(range(0, size1[0], block_size)):
            for j in range(0, size2[0], block_size):
                result[i:i + block_size, j:j + block_size] = torch.cosine_similarity(
                    tensor1[i:i + block_size].unsqueeze(1), tensor2[j:j + block_size].unsqueeze(0), dim=-1)
        torch.cuda.empty_cache()
        return result

def block_cosine_similarity(tensor1, tensor2, block_size=2000):
    """
    tensor1: (M, d)
    tensor2: (N, d)
    """
    size1 = tensor1.size()
    size2 = tensor2.size()
    result = torch.zeros(size1[0], size2[0]).to(tensor1.device)
    result = []
    # result.requires_grad = True
    # pdb.set_trace()
    # for i in tqdm(range(0, size1[0], block_size)):
    torch.cuda.empty_cache()
    for i in range(0, size1[0], block_size):
        result_line = []
        for j in range(0, size2[0], block_size):
            result_line.append(torch.cosine_similarity(
                tensor1[i:i + block_size].unsqueeze(1), tensor2[j:j + block_size].unsqueeze(0), dim=-1))
        result_line = torch.cat(result_line, dim=1)
        result.append(result_line)
    result = torch.cat(result, dim=0)
    # print(result.requires_grad)
            # result[i:i + block_size, j:j + block_size] = result[i:i + block_size, j:j + block_size] + torch.cosine_similarity(
            #     tensor1[i:i + block_size].unsqueeze(1), tensor2[j:j + block_size].unsqueeze(0), dim=-1)
    
    # pdb.set_trace()
    return result

def block_dot(tensor1, tensor2, block_size=2000):
    """
    tensor1: (M, d)
    tensor2: (N, d)
    """
    size1 = tensor1.size()
    size2 = tensor2.size()
    result = torch.zeros(size1[0], size2[0]).to(tensor1.device)
    result = []

    tensor2 = tensor2.to(tensor1.dtype)
    # result.requires_grad = True
    # pdb.set_trace()
    # for i in tqdm(range(0, size1[0], block_size)):
    torch.cuda.empty_cache()

    for i in range(0, size1[0], block_size):
        result_line = []
        for j in range(0, size2[0], block_size):
            result_line.append(
                tensor1[i:i + block_size] @ tensor2[j:j + block_size].T)
        result_line = torch.cat(result_line, dim=1)
        result.append(result_line)
    result = torch.cat(result, dim=0)
    # print(result.requires_grad)
            # result[i:i + block_size, j:j + block_size] = result[i:i + block_size, j:j + block_size] + torch.cosine_similarity(
            #     tensor1[i:i + block_size].unsqueeze(1), tensor2[j:j + block_size].unsqueeze(0), dim=-1)
    result = standalize(result)
    # pdb.set_trace()

    return result