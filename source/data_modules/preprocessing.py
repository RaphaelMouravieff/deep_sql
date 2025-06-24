
from functools import partial
import os
import pandas as pd

def preprocess_tableqa_function(examples, tokenizer, data_args, padding, is_training=False):
        """
        The is_training FLAG is used to identify if we could use the supervision
        to truncate the table content if it is required.
        """

        questions = [question.lower() for question in examples["question"]]
        example_tables = examples["table"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]

        answers = examples["answers"]

        if is_training:
            model_inputs = tokenizer(
                table=tables,
                query=questions,
                answer=answers,
                max_length=data_args.max_source_length,
                padding=padding,
                truncation=True,
            )
        else:
            model_inputs = tokenizer(
                table=tables, query=questions, max_length=data_args.max_source_length, padding=padding, truncation=True
            )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                answer=[", ".join(answer) for answer in answers],
                max_length=data_args.max_target_length,
                padding=padding,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


def preprocess_datasets(datasets, tokenizer, data_args, model_args, training_args, logger):

    output = {}
    do_ = [training_args.do_train, training_args.do_eval, training_args.do_predict]
    modes = ['train', "validation", "test"]

    for _, (do, mode) in enumerate(zip(do_, modes)):
        if not do:
            output[mode] = None
            continue

        is_training = mode == "train"
        logger.info(f'Is training ? {is_training}')

        preprocess_tableqa_function_adapt = partial(preprocess_tableqa_function,
                                                   tokenizer=tokenizer,
                                                   data_args=data_args,
                                                   padding="max_length" if data_args.pad_to_max_length else False,
                                                   is_training=True)
        
        column_names = datasets[mode].column_names
        output[mode] = datasets[mode].map(preprocess_tableqa_function_adapt,
                                        batched=True, 
                                        num_proc=data_args.preprocessing_num_workers,
                                        remove_columns=column_names,
                                        load_from_cache_file=not data_args.overwrite_cache)
        
        logger.info(f"Dataset {mode}")
        logger.info(output[mode])


    train_dataset, eval_dataset, predict_dataset = output.get('train'), output.get('validation'), output.get('test')


    return output.get('train'), output.get('validation'), output.get('test')