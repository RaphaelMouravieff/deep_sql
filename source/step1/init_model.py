#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import argparse
import torch
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    TapexTokenizer,
    set_seed,
    HfArgumentParser,
)
from datasets import load_dataset, load_from_disk
import pandas as pd
import json
from typing import List, Optional
import numpy as np
from ollama import Client
import subprocess
# Add the parent directory to the Python path
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from source.utils.args import ModelArguments, DataArguments

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using ModelArguments and DataArguments"""
    parser = HfArgumentParser((ModelArguments, DataArguments))
    
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set default values for required arguments with absolute paths
    model_args = ModelArguments(
        model_name_or_path=os.path.join(project_root, "../models/bart_large_step0/checkpoint-10000"),
        ollama_model_name_or_path="qwen2.5:14b",
    )
    
    data_args = DataArguments(
        dataset_name=os.path.join(project_root, "../data/training_dataset/step0"),
        max_source_length=1024,
    )
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args

def log_gpu_usage():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
    logger.info(output.decode())

def check_gpu_memory():
    """Check if GPU is available and print memory stats"""
    log_gpu_usage()
    if torch.cuda.is_available():
        logger.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Reserved GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")
        logger.info(f"Allocated GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        logger.info(f"Free reserved GPU memory: {free_memory / 1e9:.2f} GB")
        return True
    else:
        logger.warning("No GPU available, using CPU")
        return False

def load_models(model_args, ollama_model_loaded=False):
    """Load both the fine-tuned model and LLM"""
    # Check if model path exists
    if hasattr(model_args, "fine_tuned_model_path") and model_args.fine_tuned_model_path is not None:
        model_args.model_name_or_path = model_args.fine_tuned_model_path
        logger.info(f"Using fine-tuned model from {model_args.model_name_or_path}")
    else:
        logger.info(f"Using bart-large-cnn model from {model_args.model_name_or_path}")
        
    print(f"Checking if {model_args.model_name_or_path} exists")
    if not os.path.exists(model_args.model_name_or_path):
        print(f"Model path does not exist: {model_args.model_name_or_path}")
        # check if the first folder in the path exists (example: "../models/bart_large_step0/checkpoint-10000" is not a valid path -> check if "./models/bart_large_step0" exists)
        print(f"Checking if {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))} exists")
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_args.model_name_or_path)):
            model_args.model_name_or_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_args.model_name_or_path)
            print(f"Model path found: {model_args.model_name_or_path}")
        else:
            raise FileNotFoundError(f"Model path does not exist: {model_args.model_name_or_path}")
    
    # List files in model directory
    # logger.info(f"Files in model directory: {os.listdir(model_args.model_name_or_path)}")
    
    # Load fine-tuned model
    logger.info(f"Loading fine-tuned model from {model_args.model_name_or_path}")
    try:
        # Load config first
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            local_files_only=True
        )
        config.max_length = 1024
        config.early_stopping = False
        
        # Load tokenizer
       
        tokenizer = TapexTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer, 
            add_prefix_space=True,
            local_files_only=True
        )
        # Try different loading methods
        try:
            # First try: Load with safetensors
            model = BartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
            )
        except Exception as e1:
            logger.warning(f"First loading attempt failed: {str(e1)}")
            try:
                # Second try: Load without safetensors
                model = BartForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    use_safetensors=False
                )
            except Exception as e2:
                logger.warning(f"Second loading attempt failed: {str(e2)}")
                # Third try: Load with local_files_only
                model = BartForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    local_files_only=True
                )
        
        # Check decoder start token
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ollama_model_loaded:
        # Initialize Ollama client for LLM
        logger.info(f"Initializing Ollama client with model {model_args.ollama_model_name_or_path}")
        ollama_client = Client()
        
        # Test Ollama connection
        try:
            ollama_client.list()
            logger.info("Successfully connected to Ollama")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.warning("Continuing without Ollama LLM")
        
        # test ollama model
        response = ollama_client.chat(model=model_args.ollama_model_name_or_path, messages=[{"role": "user", "content": "Hello, how are you?"}])
        logger.info(f"Ollama response: {response}")
    else:
        ollama_client = None
    
    # see gpu memory
    check_gpu_memory()
    
    return model, tokenizer, ollama_client, device

def load_dataset_for_inference(data_args):
    """Load the dataset for inference"""
    logger.info(f"Loading dataset from {data_args.dataset_name}")
    try:
        dataset = load_from_disk(data_args.dataset_name)
    except:
        dataset = load_dataset(data_args.dataset_name)
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    return dataset

def main():
    model_args, data_args = parse_args()
    set_seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs("../data/step1_output", exist_ok=True)
    
    # Check and fix model files if needed
    #check_and_fix_model_files(model_args.model_name_or_path)
    
    # Check GPU memory
    has_gpu = check_gpu_memory()
    
    # Load models
    model, tokenizer, ollama_client, device = load_models(model_args, ollama_model_loaded=True)
    
    # Load dataset
    dataset = load_dataset_for_inference(data_args)
    
    check_gpu_memory()
    logger.info("Successfully loaded both models and dataset")
    logger.info(f"Fine-tuned model parameters: {model.num_parameters():,}")
    
    logger.info("Model loading test completed successfully")

if __name__ == "__main__":
    main()
