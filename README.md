# Improve Text Encoder via Graph Neural Network
Code for the BERT version implementation of the TextGNN model in WWW 2021 paper: [TextGNN: Improve Text Encoder via Graph Neural Network](https://arxiv.org/abs/2101.06323)


## Requirements: 
* **Tensorflow 2.2.0**
* Python 3.7
* CUDA 10.1+ (For GPU)
* HuggingFace transformers
* HuggingFace wandb (For logging)

## Example Training Command
	$ python train.py --do_train --do_eval --train_data_size 400000000 --train_data_path ../data/QK_Neighbor/Teacher/ --eval_train_data_path ../data/QK_Neighbor/Teacher_Eval/ --eval_data_path ../data/QK_Neighbor/Validation/ --config_path ../config/model.config --output_dir ../outputs/model --logging_dir ../logging/model --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --evaluate_during_training --overwrite_output_dir --learning_rate 1e-4 --warmup_steps 2000 --num_train_epochs 2.0 --pretrained_bert_name bert-base-uncased --eval_steps 10000 --logging_steps 10000 --save_steps 10000

## Example Inference Command
	$ python train.py --do_predict --test_data_path ../data/QK_Neighbor/Test/ --config_path ../config/model.config --output_dir ../outputs/model --logging_dir ../logging/model

## Acknowledgements:
This code base was heavily adapted from the HuggingFace Transformers repository: https://github.com/huggingface/transformers.
