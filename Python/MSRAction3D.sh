#!/bin/bash

for method in greedy
do
    for v in `seq 2 5`
    do
        python Evaluate_MSRAction3D_60.py \
            --method $method \
            --v_length $v &
    done
done