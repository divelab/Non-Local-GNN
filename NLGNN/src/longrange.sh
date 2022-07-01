#!/bin/bash

### Our results

GPU=9

###gcn-based
echo "===Chameleon==="
CUDA_VISIBLE_DEVICES=${GPU} python longrange.py --dataset=chameleon --runs=10 --weight_decay=0 --dropout1=0 --dropout2=0 --lr=0.05 --epochs=6000 --model=NLGCN
echo "===Squirrel==="
CUDA_VISIBLE_DEVICES=${GPU} python longrange.py --dataset=squirrel --runs=10 --weight_decay=0 --dropout1=0 --dropout2=0.5 --lr=0.05 --hidden=96 --model=NLGCN

### mlp-based
echo "===Film(Actor)==="
CUDA_VISIBLE_DEVICES=${GPU} python longrange.py --dataset=film --runs=10 --weight_decay=0.0005 --dropout1=0 --dropout2=0.5 --lr=0.05 --model=NLMLP
echo "===Cornell==="
CUDA_VISIBLE_DEVICES=${GPU} python longrange.py --dataset=cornell --runs=10 --weight_decay=0.00005 --dropout1=0.5 --dropout2=0 --lr=0.05 --kernel=3 --model=NLMLP
echo "===Texas==="
CUDA_VISIBLE_DEVICES=${GPU} python longrange.py --dataset=texas --runs=10 --weight_decay=0.0005 --dropout1=0.5 --dropout2=0 --lr=0.05 --model=NLMLP
echo "===Wisconsin==="
CUDA_VISIBLE_DEVICES=${GPU} python longrange.py --dataset=wisconsin --runs=10 --weight_decay=0.0005 --dropout1=0.5 --dropout2=0 --lr=0.05 --kernel=3 --model=NLMLP
