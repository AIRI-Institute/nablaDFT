#!/bin/bash

# this example uses a single node (`NUM_NODES=1`) w/ 4 GPUs (`NUM_GPUS_PER_NODE=4`)
export NCCL_P2P_LEVEL=NVL
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# launch your script w/ `torch.distributed.launch`
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    train_ddp.py \
    @configs/args_nablaDFT.txt
