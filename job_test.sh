#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GTX3090
#SBATCH --qos=low
#SBATCH -J model49_NUS_bce_r101
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#此处不使用conda activate，而使用source activate激活环境test
source activate pytorch
cd /public/home/hpc2307070100001/project01/new_practice/GCN_2024 || exit


#salloc  --partition=GTX3090 --qos=low -J Modeltest2 --nodes=1 --ntasks-per-node=12 --gres=gpu:1 --time=2:00:00
salloc  --partition=A40 --qos=low -J Modeltest3 --nodes=1 --ntasks-per-node=6 --gres=gpu:1 --time=2:00:00
ssh GPU03
conda activate pytorch112_LoRA
cd /public/home/hpc2307070100001/MultiLabelClassification/SARL


