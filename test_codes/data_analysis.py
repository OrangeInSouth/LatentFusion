from tqdm import tqdm
import pdb
def print_histogram(data, N):
    min_v = min(data)
    max_v = max(data)
    histogram = [0 for i in range(N)]
    chunk = (max_v - min_v) / N
    # pdb.set_trace()
    for i in tqdm(data):
        histogram[int((i // chunk))] += 1
    for i in range(N):
        ratio = histogram[i] / len(data)
        start = min_v + i * chunk
        end = min_v + (i + 1) * chunk
        print(f"{str(round(start, 3)):>6}~{str(round(end, 3)):>6}: {str(round(ratio, 4)):>6} {'*'*int(ratio*N*50)}")


