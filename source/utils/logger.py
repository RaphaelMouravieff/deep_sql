import logging
import sys
from transformers.trainer_utils import is_main_process
from transformers.utils.logging import set_verbosity_info


def setup_logger(name: str = __name__, training_args=None) -> logging.Logger:
    """
    Set up a logger that works in both training and non-training contexts.

    Args:
        name: The name of the logger (typically use __name__).
        training_args: Optional, Hugging Face training args.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if training_args is not None:
        # For distributed setups with transformers
        level = logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
        logger.setLevel(level)

        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            f"distributed training: {training_args.local_rank != -1}, 16-bit training: {training_args.fp16}"
        )

        if is_main_process(training_args.local_rank):
            set_verbosity_info()
            logger.info(f"Training/evaluation parameters: {training_args}")
    else:
        logger.setLevel(logging.INFO)

    return logger