
from transformers import AutoConfig, BartForConditionalGeneration, TapexTokenizer


def load_config(model_args, logger):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path, )

    config.no_repeat_ngram_size = 0
    config.max_length = 1024
    config.early_stopping = False

    logger.info(f"Using model config: {config}")

    return config


def load_tokenizer(model_args, logger):
    tokenizer = TapexTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer, add_prefix_space=True,
    )
    return tokenizer


def load_model_ft(model_args, config, logger):

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config, )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    logger.info(f"Using model: {model}")

    return model