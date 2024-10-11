#!/bin/bash
#SBATCH -J LatentFusion-TriviaQA-Dev                               # 作业名为 test
#SBATCH -o /share/home/fengxiaocheng/ychuang/LatentFusion/job_scripts/TriviaQA/test.out                          # stdout 重定向到 test.out
#SBATCH -e /share/home/fengxiaocheng/ychuang/LatentFusion/job_scripts/TriviaQA/test.err                           # stderr 重定向到 test.err
#SBATCH -p gpu02                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 8:00:00                            # 任务运行的最长时间为 1 小时
##SBATCH -w gpu02                             # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:1           # 申请 1 卡 A100 80GB，如果只申请CPU可以删除本行
#SBATCH --cpus-per-task=8

source ~/.bashrc

conda activate ychuang
export CUDA_VISIBLE_DEVICES=7
mode=dev
seed=1
anchor_num=1000000
tgt_layer=32
src_layer=40
tgt_model="mistral-7b"
src_mdoel="llama2-13b"

proj_path=/share/home/fengxiaocheng/ychuang/LatentFusion
export PYTHONPATH=${proj_path}
cd ${proj_path}


# python src/main.py --config confs/TriviaQA/llama2-13b_mistral-7b.json \
# --models llama2-13b mistral-7b \
# --layer-alignment 6 5 \
# --anchors-path ${proj_path}/experiments/anchor_embeddings/llama2-13b_mistral-7b_1000000anchors_seed1_bug.pt \
# --embedding-projection-path ${proj_path}/experiments//embedding_projection/EstimationEmbeddingProjection_${anchor_num}anchors_seed1_layer6-5.pt \
# --result_save_dir ${proj_path}/experiments/TriviaQA/${mode}/llama2-13b_mistral-7b_${anchor_num}anchors_seed${seed} \
# --sampling-anchor-num 20000 \
# --ensemble_weight 0.001 0.999 \
# --run_mode ${mode} 

for weight1 in 0.0001 0.8 0.7 0.5; do

    # weight2=$((1.0-$weight1))
    weight2=$(echo "scale=4; 1.0 - $weight1" | bc)

    python src/main.py --config confs/TriviaQA/llama2-13b_mistral-7b.json \
    --models llama2-13b mistral-7b \
    --layer-alignment $tgt_layer $src_layer \
    --anchors-path ${proj_path}/experiments/anchor_embeddings/llama2-13b_mistral-7b_1000000anchors_seed1_bug.pt \
    --embedding-projection-path ${proj_path}/experiments//embedding_projection/${tgt_model}_${src_mdoel}/EstimationEmbeddingProjection_${anchor_num}anchors_seed1_layer${tgt_layer}-${src_layer}.pt \
    --result_save_dir ${proj_path}/experiments/TriviaQA/${mode}/${tgt_model}_${src_mdoel}_${anchor_num}anchors_seed${seed} \
    --sampling-anchor-num 20000 \
    --ensemble_weight $weight1 $weight2 \
    --run_mode ${mode} 

done

