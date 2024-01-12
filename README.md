# RAW

## Pre Process Data

The training data should be put in args.train_folder.

The evaluation data should be put in args.eval_folder.

If the data size is too large, the data can be divided into multiple files and put in the same folder.The calc_files function in train_model.py will walk your directory and generate a file list.

The data format of the input is an N * L * 2 GPS trajectories, where N is the number of the data samples, L is GPS the sequence length.

## Train Model

When everything is prepared run train.bash to train the model.

## Generate Human Embedding

When the training process is over, run export.bash to generate human embeddings. 
The human's trajectories should be input in args.eval_folder.

## Generate Region Embedding

When the human embedding generation process is over, run region_export.bash to generate region embeddings. 
The human's embedding should be input in args.embedding_path.
