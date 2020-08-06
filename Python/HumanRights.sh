#!/bin/bash

for method in dtw opw greedy OT
do
    for v in 4 6 8
    do
        python Evaluate_humanRights.py \
            --method $method \
            --v_rate $v \
            --classnum 60 &
    done
done
# for method in sinkhorn
# do
#     for v in `seq 2 3`
#     do
#         for r in 0.004 0.003 0.002
#         do
#             python Evaluate_MSRAction3D_60.py \
#                 --method $method \
#                 --v_length $v \
#                 --reg $r &
#         done
#     done
# done