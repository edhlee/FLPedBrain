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


for i in {0..310}
  do
     echo "Current epoch $i"

    clients=16
    local_epochs=1
    client_frac=0.9
    site_idx=0
#     use_iid_data=False
#     l2_norm_clip=10.0
#     noise_multiplier=0.001

    global_client_idx=0
    CUDA_VISIBLE_DEVICES="0" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    #wait
    global_client_idx=1
    CUDA_VISIBLE_DEVICES="1" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=2
    CUDA_VISIBLE_DEVICES="2" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=3
    CUDA_VISIBLE_DEVICES="3" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=4
    # wait
    CUDA_VISIBLE_DEVICES="4" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=5
    CUDA_VISIBLE_DEVICES="5" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=6
    CUDA_VISIBLE_DEVICES="6" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=7
    CUDA_VISIBLE_DEVICES="7" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &

    wait
    global_client_idx=8
    CUDA_VISIBLE_DEVICES="0" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=9
    CUDA_VISIBLE_DEVICES="1" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=10
    CUDA_VISIBLE_DEVICES="2" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=11
    CUDA_VISIBLE_DEVICES="3" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &


    # wait
    global_client_idx=12
    CUDA_VISIBLE_DEVICES="4" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &

    global_client_idx=13
    CUDA_VISIBLE_DEVICES="5" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=14
    CUDA_VISIBLE_DEVICES="6" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &
    global_client_idx=15
    CUDA_VISIBLE_DEVICES="7" python3 fed_prox.py $global_client_idx $batch_size True 10 0.9 False $i $global_client_idx $clients $local_epochs \
        $use_iid_data $client_frac $v $site_idx $l2_norm_clip $noise_multiplier $num_microbatches $use_only_synths  &



    wait
#     CUDA_VISIBLE_DEVICES="0" python3 reduce.py $i $clients #--starting_epoch=$i --epoch=$((i+1)) 8
     CUDA_VISIBLE_DEVICES="0" python3 reduce_iid_2.5Dv2.py $i $clients $v #--starting_epoch=$i --epoch=$((i+1)) 8

done
