#!/bin/bash
#SBATCH -J LatentFusion-TriviaQA-Dev                               # 作业名为 test
#SBATCH -o /share/home/fengxiaocheng/ychuang/LatentFusion/job_scripts/TriviaQA/test.out                           # stdout 重定向到 test.out
#SBATCH -e /share/home/fengxiaocheng/ychuang/LatentFusion/job_scripts/TriviaQA/test.err                           # stderr 重定向到 test.err
#SBATCH -p gpu02                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 8:00:00                            # 任务运行的最长时间为 1 小时
##SBATCH -w gpu02                             # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:1           # 申请 1 卡 A100 80GB，如果只申请CPU可以删除本行
#SBATCH --cpus-per-task=8

source ~/.bashrc

conda activate ychuang
export CUDA_VISIBLE_DEVICES=0
mode=dev
seed=1

proj_path=/share/home/fengxiaocheng/ychuang/LatentFusion
export PYTHONPATH=${proj_path}
cd ${proj_path}

python utils/anchors/extract_anchor_embedding_last_hidden_v2.py