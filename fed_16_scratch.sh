#!/bin/bash



#for v in {0..7}
#do

v=0
# rm -rf "checkpoints_3d_variation"$v/*
#rm -rf "evals_variation"$v/*
#done

use_iid_data=False
noise_multiplier=0.3

fp16=False
l2_norm_clip=5.0
batch_size=8
num_microbatches=1
use_only_synths=False


i=0
clients=16
local_epochs=1
client_frac=0.9
site_idx=0
#     use_iid_data=False
#     l2_norm_clip=10.0
#     noise_multiplier=0.001

global_client_idx=0
CUDA_VISIBLE_DEVICES="0" python3 fed_prox_16modelsInsideOne.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
	$use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
