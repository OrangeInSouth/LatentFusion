from tqdm import tqdm
import torch
from termcolor import colored
import pdb


def print_histogram(data, N):
    if isinstance(data, torch.Tensor):
        data = data.tolist()
    min_v = min(data) - 1e-6
    max_v = max(data) + 1e-6
    histogram = [0 for i in range(N)]
    chunk = (max_v - min_v) / N
    # pdb.set_trace()
    for i in tqdm(data):
        try:
            histogram[int((i-min_v) / chunk)] += 1
        except Exception:
            print("遗留的BUG，没找到原因")
            pdb.set_trace()
    for i in range(N):
        ratio = histogram[i] / len(data)
        start = min_v + i * chunk
        end = min_v + (i + 1) * chunk
        print(f"{str(round(start, 3)):>6}~{str(round(end, 3)):>6}: {str(round(ratio, 4)):>6} {'*'*int(ratio*N*50)}")


def compare_histogram(src_data, tgt_data, N):
    if isinstance(src_data, torch.Tensor):
        src_data = src_data.tolist()
    if isinstance(tgt_data, torch.Tensor):
        tgt_data = tgt_data.tolist()

    epsilon = 1e-4
    min_v = min(src_data + tgt_data) - epsilon
    max_v = max(src_data + tgt_data) + epsilon

    src_histogram = [0 for i in range(N)]
    tgt_histogram = [0 for i in range(N)]
    chunk = (max_v - min_v) / N
    # pdb.set_trace()
    for i in tqdm(src_data):
        try:
            src_histogram[int((i-min_v) / chunk)] += 1
        except Exception as e:
            pdb.set_trace()
    for i in tqdm(tgt_data):
        tgt_histogram[int((i-min_v) / chunk)] += 1

    for i in range(N):
        src_ratio = src_histogram[i] / len(src_data)
        tgt_ratio = tgt_histogram[i] / len(src_data)
        start = min_v + i * chunk
        end = min_v + (i + 1) * chunk
        print(f"{str(round(start, 3)):>6}~{str(round(end, 3)):>6}: {str(round(src_ratio, 4)):>6}", colored(f"{'*'*int(src_ratio*N*50)}", "red", "on_red"))
        print(f"{str(round(start, 3)):>6}~{str(round(end, 3)):>6}: {str(round(tgt_ratio, 4)):>6}", colored(f" {'*'*int(tgt_ratio*N*50)}", "green", "on_green"))
        print()
