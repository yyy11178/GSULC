#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2
###step1--drop_rate 0.25 --queue_size 10 --warm_up 15
###step2--drop_rate 0.25 --queue_size 10 --warm_up 10
python main.py --dataset web-aircraft --n_classes 100 --base_lr 0.001 --batch_size 64 --epoch 100 --drop_rate 0.25 --queue_size 10 --warm_up 15 --weight_decay 1e-8 --step 1
sleep 30
python main.py --dataset web-aircraft --n_classes 100 --base_lr 0.0001 --batch_size 18 --epoch 100 --drop_rate 0.25 --queue_size 10 --warm_up 10 --weight_decay 1e-5 --step 2
