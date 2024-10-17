#!/bin/bash
#SBATCH -J LatentFusion-TriviaQA-Dev                               # 作业名为 test
#SBATCH -o /share/home/fengxiaocheng/ychuang/LatentFusion/job_scripts/TriviaQA/test.out                          # stdout 重定向到 test.out
#SBATCH -e v                       # stderr 重定向到 test.err
#SBATCH -p gpu02                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 8:00:00                            # 任务运行的最长时间为 1 小时
##SBATCH -w gpu02                             # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=8

source ~/.bashrc

conda activate ychuang

mode=dev
seed=1
anchor_num=1000000
tgt_layer=39
src_layer=31
tgt_model="llama2-13b"
src_model="mistral-7b"

proj_path=/share/home/fengxiaocheng/ychuang/LatentFusion
export PYTHONPATH=${proj_path}
cd ${proj_path}


gpu_count=5
gpu_id=0  # 初始化第一个GPU
for weight1 in 0.5 0.6 0.7 0.8 0.9; do  # 

    for sr in 1; do

        # weight2=$((1.0-$weight1))
        weight2=$(echo "scale=4; 1.0 - $weight1" | bc)
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo $gpu_id
            python src/main.py --config confs/TriviaQA/llama2-13b_mistral-7b.json \
        --models ${tgt_model} ${src_model} \
        --layer-alignment $tgt_layer $src_layer \
        --embedding-projection-path ${proj_path}/experiments//embedding_projection/${tgt_model}_${src_model}/EstimationEmbeddingProjection_${anchor_num}anchors_seed1_layer${tgt_layer}-${src_layer}.pt \
        --result_save_dir ${proj_path}/experiments/TriviaQA/${mode}/${tgt_model}_${src_model}_${anchor_num}anchors_seed${seed} \
        --sampling-anchor-num ${anchor_num} \
        --ensemble_weight $weight1 $weight2 \
        --subspace-ratio ${sr} \
        --run_mode ${mode} &

        gpu_id=$(( (gpu_id + 1) % gpu_count ))

        if [ $gpu_id -eq 0 ]; then
            echo "Waiting for GPU tasks to complete..."
            wait
        fi

    done

done

wait
