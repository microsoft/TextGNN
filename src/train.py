"""Main script for training"""

import logging
import os
import math
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from data_preprocessing import read_preprocessed_datasets, read_preprocessed_datasets_twinbert, read_preprocessed_datasets_beijing
from metric import metrics
from twinbertgnn import TwinBERTGNN
from transformers import TFTrainingArguments, HfArgumentParser, EvalPrediction
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trainer import TFTrainer
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    train_data_path: Optional[str] = field(
        default='../data/QK_Neighbor/Teacher/', metadata={"help": "Path to training data"}
    )

    train_data_size: Optional[int] = field(
        default=None, metadata={"help": "Training data size"}
    )

    train_data_int_label: bool = field(
        default=False, metadata={"help": "Whether the training dataset use an int label or float roberta score"}
    )

    do_eval_train: bool = field(
        default=False, metadata={"help": "Whether to run evaluation on a small training dataset."}
    )

    eval_train_data_path: Optional[str] = field(
        default='../data/QK_Neighbor/Training/', metadata={"help": "Path to training eval data"}
    )

    eval_data_path: Optional[str] = field(
        default='../data/QK_Neighbor/Validation/', metadata={"help": "Path to eval data"}
    )

    test_data_path: Optional[str] = field(
        default='../data/QK_Neighbor/Validation/', metadata={"help": "Path to eval data"}
    )

    is_twinbert_format: bool = field(
        default=False, metadata={"help": "Whether the dataset is from the previous twinbert model format"}
    )

    finetune: bool = field(
        default=False, metadata={"help": "Whether the this training is finetune"}
    )

    finetune_previous_epoch: Optional[int] = field(
        default=10, metadata={"help": "Number of pretrained epochs"}
    )

    neigh_weights: bool = field(
        default=False, metadata={"help": "Whether to read the neighbor weights from TF datasets"}
    )


@dataclass
class ModelArguments:
    model_checkpoint_path: str = field(
        default=None, metadata={"help": "Path to pre-trained model weights checkpoint"}
    )

    is_tf_checkpoint: bool = field(
        default=True, metadata={"help": "Whether the checkpoint is from a torch model or twinbert tf model"}
    )

    config_path: Optional[str] = field(
        default="../config/model.config", metadata={"help": "Path to model config"}
    )

    checkpoint_dict: Optional[str] = field(
        default='../config/twinbert_checkpoint_dict.txt',
        metadata={"help": "Mapping between model weights to checkpoint weights"}
    )

    pretrained_bert_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Bert name, used by bpe tokenizer models"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Load format, is twinbert?")
    print(data_args.is_twinbert_format)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(
        "n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.n_gpu,
        bool(training_args.n_gpu > 1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    with training_args.strategy.scope():
        if model_args.model_checkpoint_path:
            model = TwinBERTGNN.load_from_checkpoint(model_args.config_path, model_args.model_checkpoint_path,
                                                     model_args.checkpoint_dict,
                                                     is_tf_checkpoint=model_args.is_tf_checkpoint)
        elif model_args.pretrained_bert_name:
            model = TwinBERTGNN.load_from_bert_pretrained(model_args.config_path, model_args.pretrained_bert_name)
        else:
            model = TwinBERTGNN(model_args.config_path)
            model(model.dummy_input, training=False)

    logger.info("Model finished loading.")
    if (training_args.do_train and not os.path.exists(data_args.train_data_path)) or (training_args.do_eval and (not os.path.exists(data_args.eval_train_data_path) or not os.path.exists(data_args.eval_data_path))) or (training_args.do_predict and not os.path.exists(data_args.test_data_path)):
        raise ValueError(
            f"Please run the create_tf_record_data.py script to first generate tf record files."
        )

    is_triletter = model.config.embedding_type == 'triletter'

    train_dataset = (
        read_preprocessed_datasets(data_args.train_data_path,
                                   max_n_letters=model.config.triletter_max_letters_in_word,
                                   max_seq_len=model.config.max_seq_len,
                                   is_triletter=is_triletter, int_label=data_args.train_data_int_label,
                                   a_fanouts=model.config.a_fanouts,
                                   b_fanouts=model.config.b_fanouts, neigh_weights=data_args.neigh_weights)
        if training_args.do_train
        else None
    )

    eval_train_dataset = (
        read_preprocessed_datasets(data_args.eval_train_data_path,
                                   max_n_letters=model.config.triletter_max_letters_in_word,
                                   max_seq_len=model.config.max_seq_len,
                                   is_triletter=is_triletter, int_label=True,
                                   a_fanouts=model.config.a_fanouts,
                                   b_fanouts=model.config.b_fanouts, neigh_weights=data_args.neigh_weights)
        if data_args.do_eval_train
        else None
    )

    eval_dataset = (
        read_preprocessed_datasets(data_args.eval_data_path,
                                   max_n_letters=model.config.triletter_max_letters_in_word,
                                   max_seq_len=model.config.max_seq_len,
                                   is_triletter=is_triletter, int_label=True, a_fanouts=model.config.a_fanouts,
                                   b_fanouts=model.config.b_fanouts, neigh_weights=data_args.neigh_weights)
        if training_args.do_eval
        else None
    )

    test_dataset = (
        read_preprocessed_datasets(data_args.test_data_path,
                                   max_n_letters=model.config.triletter_max_letters_in_word,
                                   max_seq_len=model.config.max_seq_len,
                                   is_triletter=is_triletter, int_label=True, a_fanouts=model.config.a_fanouts,
                                   b_fanouts=model.config.b_fanouts, neigh_weights=data_args.neigh_weights)
        if training_args.do_predict
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metrics(p.predictions, p.label_ids)

    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_train_dataset=eval_train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        train_size=data_args.train_data_size,
    )

    if training_args.do_train:
        logger.info("*** Start Training ***")
        trainer.train(finetune=data_args.finetune, previous_epoch=data_args.finetune_previous_epoch, do_eval_train=data_args.do_eval_train)
        trainer.save_model()

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        trainer.gradient_accumulator.reset()

        with trainer.args.strategy.scope():
            trainer.train_steps = math.ceil(trainer.num_train_examples / trainer.args.train_batch_size)
            optimizer, lr_scheduler = trainer.get_optimizers()
            iterations = optimizer.iterations
            folder = os.path.join(trainer.args.output_dir, PREFIX_CHECKPOINT_DIR)
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=trainer.model)
            trainer.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=trainer.args.save_total_limit)

            if trainer.model.ckpt_manager.latest_checkpoint:
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", trainer.model.ckpt_manager.latest_checkpoint
                )

                ckpt.restore(trainer.model.ckpt_manager.latest_checkpoint).expect_partial()

        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")

            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)

        return results

    if training_args.do_predict:
        logger.info("*** Inference ***")
        trainer.gradient_accumulator.reset()

        with trainer.args.strategy.scope():
            trainer.num_train_examples = 0
            trainer.train_steps = math.ceil(trainer.num_train_examples / trainer.args.train_batch_size)
            optimizer, lr_scheduler = trainer.get_optimizers()
            iterations = optimizer.iterations
            folder = os.path.join(trainer.args.output_dir, PREFIX_CHECKPOINT_DIR)
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=trainer.model)
            trainer.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder,
                                                                    max_to_keep=trainer.args.save_total_limit)

            if trainer.model.ckpt_manager.latest_checkpoint:
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint",
                    trainer.model.ckpt_manager.latest_checkpoint
                )

                ckpt.restore(trainer.model.ckpt_manager.latest_checkpoint).expect_partial()
        predictions = trainer.predict(test_dataset)
        path_prediction = os.path.join(training_args.output_dir, "predictions.npz")
        np.savez(path_prediction, predictions=predictions.predictions, labels=predictions.label_ids)

        for key, value in predictions.metrics.items():
            logger.info("  %s = %s", key, value)


if __name__ == "__main__":
    main()
