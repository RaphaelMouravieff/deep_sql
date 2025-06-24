import logging
import sys
from transformers.trainer_utils import is_main_process
from transformers.utils.logging import set_verbosity_info

def setup_logger(training_args=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)

    if training_args is not None:
        logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            f"distributed training: {training_args.local_rank != -1}, 16-bits training: {training_args.fp16}"
        )

        if is_main_process(training_args.local_rank):
            set_verbosity_info()
            logger.info(f"Training/evaluation parameters {training_args}")
    else:
        logger.setLevel(logging.INFO)

    return logger