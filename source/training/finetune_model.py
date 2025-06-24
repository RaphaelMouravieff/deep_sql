#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The Microsoft and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for tapex on table-based question answering tasks.
Adapted from script: https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
"""

import logging
import os
import sys

from functools import partial


from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


from source.utils.args import ModelArguments, DataArguments
from source.data_modules.data_loader import load_datasets, load_data_collator
from source.data_modules.preprocessing import preprocess_datasets
from source.models.ft_model_setup import load_config, load_tokenizer, load_model_ft
from source.utils.sanity_checks import check_parameters
from source.utils.logger import setup_logger
from source.utils.last_checkpoint import load_latest_checkpoint
from source.utils.metrics import compute_metrics
from source.training.hf_training import run_train, run_eval, run_predict


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    logger = setup_logger(training_args)
    
    # Detecting last checkpoint.
    last_checkpoint = load_latest_checkpoint(training_args)

    

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_datasets(data_args, logger)

    config = load_config(model_args, logger)

    tokenizer = load_tokenizer(model_args, logger)

    model = load_model_ft(model_args, config, logger)

    check_parameters(model_args, data_args, training_args, model, logger)

    train_dataset, eval_dataset, predict_dataset = preprocess_datasets(datasets, tokenizer, data_args, model_args, training_args, logger)
    
    data_collator = load_data_collator(tokenizer, model, data_args, training_args)

    compute_metrics_ = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_ if training_args.predict_with_generate else None,
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        run_train(
            trainer=trainer,
            training_args=training_args,
            data_args=data_args,
            last_checkpoint=last_checkpoint,
            train_dataset=train_dataset
        )

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        run_eval(
            trainer=trainer,
            data_args=data_args,
            eval_dataset=eval_dataset
        )

    if training_args.do_predict:
        logger.info("*** Predict ***")

        run_predict(
            trainer=trainer,
            data_args=data_args,
            predict_dataset=predict_dataset,
            training_args=training_args,
            tokenizer=tokenizer)







