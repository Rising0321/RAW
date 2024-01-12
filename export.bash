#!/bin/bash

python3 -m minRAW export --train_path /workdir/temp/0 \
                            --eval_path /workdir/temp/0 \
                            --save_model_path ./minRAW/log/ \
                            --batch_train 40 \
                            --batch_eval 40 \
                            --layers 12 \
                            --dims 768 \
                            --head 16 \
                            --gpus 6 \
                            --load_path ./minRAW/log/-2200.pth \
                            --seq_len 960
                            