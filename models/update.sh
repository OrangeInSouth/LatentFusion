src_path=/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/models/
tgt_path=/data/home/cpfu/anaconda3/envs/ychuang/lib/python3.10/site-packages/transformers/models/

cp ${src_path}/modeling_llama.py ${tgt_path}/llama/modeling_llama.py
cp ${src_path}/modeling_mistral.py ${tgt_path}/mistral/modeling_mistral.py