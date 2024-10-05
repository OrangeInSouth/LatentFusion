import unicodedata
import string
import re
import pdb
import sys
sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")

from utils import read_multiline_json, avg

dataset = "NQ"
def normalize_answer(s):
  """Normalize answer."""
  s = unicodedata.normalize("NFD", s)

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def cal_accuracy(predictions, answers):
    assert len(predictions) == len(answers)
    return avg([1.0 if normalize_answer(predictions[i]) in answers[i] else 0.0 for i in range(len(predictions))])

def cal_diff_ratio(l1, l2):
    assert len(l1) == len(l2)
    # pdb.set_trace()
    return avg([int(l1[i] != l2[i]) for i in range(len(l1))])
# start = int(sys.argv[1])
# end = int(sys.argv[2])

if dataset == "NQ":
  # NQ
  llama = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/NQ_old_version_0506/LLaMA+Mistral/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
  mistral = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/NQ_old_version_0506/baseline/Mistral/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
  # ensemble = read_multiline_json("experiments/NQ/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr1.0_learning_epochs_nums5_fuse40-32_beta0.0_p1.0.jsonl")
  ensemble = read_multiline_json("experiments/NQ/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr1.0_learning_epochs_nums5_fuse23-21_beta0.0.jsonl")
elif dataset == "TriviaQA":
  # TriviaQA
  llama = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/TriviaQA_old_version_0506/baseline/LLaMA/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
  mistral = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/TriviaQA_old_version_0506/baseline/Mistral/dev/ensemble_lr0.0_anchor_point_count32000_learning_epochs_nums5.jsonl")
  # 6-5 lr=0.01
  # ensemble = read_multiline_json("experiments/TriviaQA/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr0.01_learning_epochs_nums5_fuse6-5_beta0.0_p1.0.jsonl")
  # 40-32 lr=1
  # ensemble = read_multiline_json("experiments/TriviaQA/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr1.0_learning_epochs_nums5_fuse40-32_beta0.0_p1.0.jsonl")
  # 40-32 lr=0.5
  ensemble = read_multiline_json("experiments/TriviaQA/dev/llama2-13b_mistral-7b_10000anchors/ensemble_lr0.5_learning_epochs_nums5_fuse40-32_beta0.0_p1.0.jsonl")
  # ensemble = read_multiline_json("/data1/cpfu/baohang/Experiments/DeepEN_v0422/eval/TriviaQA_old_version_0506/LLaMA+Mistral/dev/ensemble_lr0.1_anchor_point_count32000_learning_epochs_nums5.jsonl")
else:
  raise Exception("Unknown Dataset")
answers = [[normalize_answer(a) for a in i['answer']] for i in llama]
llama_predictions = [i['prediction'] for i in llama]
mistral_predictions = [i['prediction'] for i in mistral]
ensemble_predictions = [i['prediction'] for i in ensemble]


print("Accuracy:")
print(f"LLaMA: {cal_accuracy(llama_predictions, answers) * 100}")
print(f"Mistral: {cal_accuracy(mistral_predictions, answers) * 100}")
print("Ensemble:")
print(f"Upper-bound: {avg([1.0 if normalize_answer(llama_predictions[i]) in answers[i] or normalize_answer(mistral_predictions[i]) in answers[i] else 0.0 for i in range(len(answers))]) * 100}")
print(f"Lower-bound: {avg([1.0 if normalize_answer(llama_predictions[i]) in answers[i] and normalize_answer(mistral_predictions[i]) in answers[i] else 0.0 for i in range(len(answers))]) * 100}")
print(f"Ours: {cal_accuracy(ensemble_predictions, answers) * 100}")

print(f"\nModel Output Diff: {cal_diff_ratio(llama_predictions, mistral_predictions) * 100}")
print(f"Flip Ratio: {cal_diff_ratio(llama_predictions, ensemble_predictions) * 100}")
print(f"Transfer Ratio: {avg([1.0 if llama_predictions[i] != mistral_predictions[i] and ensemble_predictions[i] == mistral_predictions[i] else 0.0 for i in range(len(answers))]) * 100}")

# for i in range(start, start + 10):
#     print(f"{i+1}: {llama_predictions[i]}  +  {mistral_predictions[i]}  =   {ensemble_predictions[i]}  |||  {ensemble[i]['answer']}")