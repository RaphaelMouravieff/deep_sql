import os

def run_train(trainer, training_args, data_args, last_checkpoint, train_dataset):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        print('Resuming from checkpoint,', checkpoint)
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print('Resuming from checkpoint,', checkpoint)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def run_eval(trainer, data_args, eval_dataset):
    metrics = trainer.evaluate(
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams,
        metric_key_prefix="eval"
    )
    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def run_predict(trainer, data_args, predict_dataset, training_args, tokenizer):
    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams,
    )
    metrics = predict_results.metrics
    metrics["predict_samples"] = len(predict_dataset)

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    if trainer.is_world_process_zero() and training_args.predict_with_generate:
        predictions = tokenizer.batch_decode(
            predict_results.predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predictions = [pred.strip() for pred in predictions]
        output_prediction_file = os.path.join(training_args.output_dir, "tapex_predictions.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))