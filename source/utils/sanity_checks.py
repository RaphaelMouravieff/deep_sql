


from source.utils.logger import setup_logger

logger = setup_logger(__name__)


def check_parameters(model_args, data_args, training_args, model):
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "Label smoothing is enabled, but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will use more memory."
        )