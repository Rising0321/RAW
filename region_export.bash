#!/bin/bash

python3 -m minRAW dump --train_path /workdir/temp/0 \
                            --eval_path /workdir/temp/0 \
                            --embedding_path /workdir/script/RAW/transfer/embedding/2023-12-26-14 \
                            --save_model_path ./minRAW/log/ \
                            --batch_train 1 \
                            --batch_eval 1 \
                            --layers 12 \
                            --dims 768 \
                            --head 16 \
                            --gpus 1 \
                            --seq_len 960 \
                            