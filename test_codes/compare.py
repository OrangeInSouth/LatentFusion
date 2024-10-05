import sys
sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")
from utils import read_multiline_json

start = int(sys.argv[1])
# end = int(sys.argv[2])
llama = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/NQ_old_version_0506/LLaMA+Mistral/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
mistral = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/NQ_old_version_0506/baseline/Mistral/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
ensemble = read_multiline_json("experiments/NQ/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr1.0_learning_epochs_nums20_fuse40-32_beta0.0_p1.0.jsonl")

llama = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/TriviaQA_old_version_0506/baseline/LLaMA/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
mistral = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/TriviaQA_old_version_0506/baseline/Mistral/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
ensemble = read_multiline_json("experiments/TriviaQA/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr1.0_learning_epochs_nums5_fuse40-32_beta0.0_p1.0.jsonl")
llama_predictions = [i['prediction'] for i in llama]
mistral_predictions = [i['prediction'] for i in mistral]
ensemble_predictions = [i['prediction'] for i in ensemble]

for i in range(start, start + 10):
    print(f"{i+1}: {llama_predictions[i]}  +  {mistral_predictions[i]}  =   {ensemble_predictions[i]}  |||  {ensemble[i]['answer']}")