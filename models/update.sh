src_path=/share/home/fengxiaocheng/ychuang/LatentFusion/models/
tgt_path=/share/home/fengxiaocheng/miniconda3/envs/ychuang/lib/python3.10/site-packages/transformers/models/

cp ${src_path}/modeling_llama.py ${tgt_path}/llama/modeling_llama.py
cp ${src_path}/modeling_mistral.py ${tgt_path}/mistral/modeling_mistral.py