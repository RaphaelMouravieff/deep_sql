from datasets import load_dataset, load_from_disk

from transformers import DataCollatorForSeq2Seq

def load_datasets(data_args, logger):

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name in ["wikitablequestions"] : 
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
            logger.info(f'Load dataset {data_args.dataset_name}')

        else:
            try:
                logger.info(f"try load from disk ..")
                datasets = load_from_disk(data_args.dataset_name)
                logger.info(f"Load dataset From Disk PATH = {data_args.dataset_name}")
            except:
                pass
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        
        datasets = load_dataset(extension, data_files=data_files)

    logger.info("datasets",datasets)

    
    return datasets


def load_data_collator(tokenizer, model, data_args, training_args):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    return data_collator