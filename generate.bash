#!/bin/bash

python3 -m minRAW generate --train_path /workdir/files/all/0 \
                            --eval_path /workdir/files/all/0\
                            --save_model_path ./minRAW/log/ \
                            --batch_train 10 \
                            --batch_eval 10 \
                            --layers 16 \
                            --dims 768 \
                            --head 16 \
                            --gpus 1 \
                            --load_path ./minRAW/log/-400.pth \
                            --epochs 1 \
                            --seq_len 960
