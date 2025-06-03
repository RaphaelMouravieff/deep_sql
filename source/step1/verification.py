#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import torch
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import numpy as np

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def preprocess_for_verification(
    table: pd.DataFrame, 
    question: str, 
    tokenizer, 
    max_source_length: int = 1024, 
    padding: bool = False
):
    """
    Preprocess the table and question for the fine-tuned model
    
    Args:
        table: The dirty table as a pandas DataFrame
        question: The question from LLM1
        tokenizer: The tokenizer for the fine-tuned model
        max_source_length: Maximum source length for tokenization
        padding: Whether to pad the input
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    model_inputs = tokenizer(
        table=table, 
        query=question.lower(), 
        max_length=max_source_length, 
        padding=padding, 
        truncation=True,
        return_tensors="pt"
    )
    
    return model_inputs

def verify_sql_answer(
    table: pd.DataFrame,
    question: str,
    answer_llm: str,
    model,
    tokenizer,
    device: str = None,
    max_source_length: int = 1024,
    max_target_length: int = 128,
    num_beams: int = 5
) -> Dict[str, Any]:
    """
    Verify if the SQL answer matches the fine-tuned model's answer
    
    Args:
        table: The dirty table as a pandas DataFrame
        question: The question from LLM1
        answer_llm: The answer from the SQL execution
        model: The fine-tuned model
        tokenizer: The tokenizer for the fine-tuned model
        device: The device to run the model on
        max_source_length: Maximum source length for tokenization
        max_target_length: Maximum target length for generation
        num_beams: Number of beams for beam search
        
    Returns:
        Dictionary with verification results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the input
    model_inputs = preprocess_for_verification(
        table=table,
        question=question,
        tokenizer=tokenizer,
        max_source_length=max_source_length
    )
    
    # Move inputs to device
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    
    # Generate answer with the fine-tuned model
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode the generated answer
    answer_ft = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0].strip()
    
    # Compare answers using denotation accuracy
    is_equivalent = evaluate_example(answer_llm.lower(), answer_ft.lower())
    
    return {
        "is_equivalent": is_equivalent,
        "answer_ft": answer_ft,
        "answer_llm": answer_llm,
        "question": question
    }

def evaluate_example(predict_str: str, ground_str: str) -> bool:
    """
    Evaluate if two answer strings are equivalent using denotation accuracy
    
    Args:
        predict_str: The predicted answer string
        ground_str: The ground truth answer string
        
    Returns:
        Boolean indicating if the answers are equivalent
    """
    delimiter = ", "
    predict_spans = predict_str.split(delimiter)
    ground_spans = ground_str.split(delimiter)
    
    predict_values = {}
    ground_values = {}
    
    # Count occurrences in prediction
    for span in predict_spans:
        span = span.strip()
        try:
            value = float(span)
        except ValueError:
            value = span
        
        predict_values[value] = predict_values.get(value, 0) + 1
    
    # Count occurrences in ground truth
    for span in ground_spans:
        span = span.strip()
        try:
            value = float(span)
        except ValueError:
            value = span
        
        ground_values[value] = ground_values.get(value, 0) + 1
    
    # Check if the value distributions are the same
    return predict_values == ground_values

def get_denotation_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Calculate denotation accuracy for a list of predictions and references
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Denotation accuracy score
    """
    assert len(predictions) == len(references)
    correct_num = 0
    
    for predict_str, ground_str in zip(predictions, references):
        is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
        if is_correct:
            correct_num += 1
            
    return correct_num / len(predictions) if len(predictions) > 0 else 0.0

def main():
    """
    Main function to run verification on real data using the inference model
    """
    import logging
    import os
    import sys
    import argparse
    import json
    from transformers import (
        HfArgumentParser,
        TapexTokenizer,
        BartForConditionalGeneration,
        set_seed,
    )
    # Add the parent and grandparent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from source.utils.args import ModelArguments, DataArguments
    from datasets import load_dataset, load_from_disk
    import pandas as pd
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    set_seed(42)
    
    # Load dataset
    try:
        logger.info(f"Loading dataset from {data_args.dataset_name}")
        dataset = load_dataset(data_args.dataset_name, split=data_args.split_name)
    except:
        logger.info(f"Loading dataset from disk: {data_args.dataset_name}")
        dataset = load_from_disk(data_args.dataset_name)
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    tokenizer = TapexTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        add_prefix_space=True,
    )
    
    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
    )
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process examples and calculate accuracy
    results = []
    correct_count = 0
    
    # Get column names for preprocessing
    column_names = dataset.column_names
    
    # Preprocess the dataset
    def preprocess_tableqa_function(examples):
        questions = [question.lower() for question in examples["question"]]
        example_tables = [table for table in examples["table"]]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        return {"processed_tables": tables, "processed_questions": questions}
    
    # Process the dataset
    processed_dataset = dataset.map(
        preprocess_tableqa_function,
        batched=True,
        remove_columns=column_names
    )
    
    for i, example in enumerate(processed_dataset):
        if i % 10 == 0:
            logger.info(f"Processing example {i}/{len(processed_dataset)}")
        
        # Get processed table and question
        table = example["processed_tables"]
        question = example["processed_questions"]
        
        # Get reference answer
        reference_answer = ", ".join(example["answers"]).lower()
        
        # Generate prediction with model
        model_inputs = preprocess_for_verification(
            table=table,
            question=question,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length
        )
        
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=data_args.max_target_length if hasattr(data_args, 'max_target_length') else 128,
                num_beams=5,
                early_stopping=True
            )
        
        prediction = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0].strip().lower()
        
        # Evaluate prediction
        is_correct = evaluate_example(prediction, reference_answer)
        if is_correct:
            correct_count += 1
        
        # Store result
        results.append({
            "id": i,
            "question": question,
            "prediction": prediction,
            "reference": reference_answer,
            "is_correct": is_correct
        })
    
    # Calculate overall accuracy
    accuracy = correct_count / len(processed_dataset) if len(processed_dataset) > 0 else 0.0
    logger.info(f"Overall denotation accuracy: {accuracy:.4f}")
    
    # Save results to file
    output_dir = os.path.join(os.path.dirname(data_args.dataset_name), "verification_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "verification_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    return accuracy

if __name__ == "__main__":
    main()
