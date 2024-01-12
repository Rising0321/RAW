#!/bin/bash

python3 -m minRAW train --train_path /workdir/temp/0 \
                        --eval_path /workdir/temp/0 \
                        --save_model_path ./minRAW/log/ \
                        --batch_train 15 \
                        --batch_eval 15 \
                        --layers 12 \
                        --dims 768 \
                        --head 16 \
                        --gpus 6 \
                        --seq_len 960 \
                        --epochs 10000 \
                        --lr 1e-4 