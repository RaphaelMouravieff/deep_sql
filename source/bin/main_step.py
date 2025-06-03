from typing import Dict, Any, Optional
from uuid import uuid4
from langchain_core.documents import Document
from source.step1.verification import preprocess_for_verification
import torch
import pandas as pd
from source.prepare_data.sql_executor import SQLExecutor
import torch.nn.functional as F

# add logger
import logging
logger = logging.getLogger(__name__)

# print logger debug
logger.setLevel(logging.DEBUG)


def calculate_answer_likelihood(model, tokenizer, input_ids, attention_mask, target_answer, device):
    """
    Calculate the likelihood of the model generating the target answer
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer (TAPEX tokenizer)
        input_ids: Input token ids (already encoded table + question)
        attention_mask: Attention mask
        target_answer: The target answer string
        device: Device to run on
        
    Returns:
        log_likelihood: Log likelihood of the target answer
        avg_log_likelihood: Average log likelihood per token
    """
    try:
        # Clean and prepare target answer
        target_answer = str(target_answer).strip()
        if not target_answer:
            logger.warning("Empty target answer")
            return float('-inf'), float('-inf')
        
        # For TAPEX tokenizer, we need to tokenize just the target text (not table+query)
        # Use encode method directly for the target answer
        target_tokens = tokenizer.encode(
            target_answer,
            add_special_tokens=False,  # Don't add special tokens for target
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Check if target tokens are valid
        if target_tokens.shape[1] == 0:
            logger.warning("No tokens generated for target answer")
            return float('-inf'), float('-inf')
        
        # Prepare decoder input ids and labels for teacher forcing
        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.bos_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = tokenizer.pad_token_id
        
        # Create labels: target tokens with EOS token
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.sep_token_id
        
        labels = torch.cat([
            target_tokens,
            torch.tensor([[eos_token_id]], device=device)
        ], dim=1)
        
        # Create decoder input: start token + target tokens (shifted right)
        decoder_input_ids = torch.cat([
            torch.tensor([[decoder_start_token_id]], device=device),
            target_tokens
        ], dim=1)
        
        # Forward pass with teacher forcing
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
        
        # Use the loss if available (most reliable)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # Loss is negative log likelihood averaged over tokens
            loss = outputs.loss.item()
            avg_log_likelihood = -loss
            log_likelihood = avg_log_likelihood * labels.shape[1]
        else:
            # Fallback: calculate manually from logits
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probabilities for each target token
            target_log_probs = []
            
            # Note: logits sequence should match labels sequence
            seq_len = min(logits.shape[1], labels.shape[1])
            
            for i in range(seq_len):
                target_token_id = labels[0, i].item()
                if i < log_probs.shape[1] and target_token_id != tokenizer.pad_token_id:
                    target_log_prob = log_probs[0, i, target_token_id].item()
                    target_log_probs.append(target_log_prob)
            
            # Calculate total and average log likelihood
            if target_log_probs:
                log_likelihood = sum(target_log_probs)
                avg_log_likelihood = log_likelihood / len(target_log_probs)
            else:
                log_likelihood = float('-inf')
                avg_log_likelihood = float('-inf')
        
        logger.debug(f"Target answer: '{target_answer}'")
        logger.debug(f"Target tokens shape: {target_tokens.shape}")
        logger.debug(f"Labels shape: {labels.shape}")
        logger.debug(f"Calculated avg_log_likelihood: {avg_log_likelihood}")
        
        return log_likelihood, avg_log_likelihood
        
    except Exception as e:
        logger.error(f"Error in calculate_answer_likelihood: {str(e)}")
        logger.debug(f"Target answer was: '{target_answer}'")
        return float('-inf'), float('-inf')


def run_pipeline_step(prompt_manager, 
                      agents,
                      tools, 
                      fine_tuned_model,
                      tokenizer, 
                      is_use_fine_tuned_model_loaded=False,
                      likelihood_threshold_high=-2.0,  # Examples with avg log likelihood > this are too easy
                      likelihood_threshold_low=-10.0) -> Optional[Dict[str, Any]]:  # Examples with avg log likelihood < this are too hard

    table_id = prompt_manager.table_manager.current_table_id

    print("Generating a new entry...")

    try:
        # Generate a question
        question_prompt = prompt_manager.get_question_prompt()
        question = agents["question_generator"].run(question_prompt)
        print(f"Generated question: {question}")

        # Translate to SQL
        sql_prompt = prompt_manager.get_traductor_prompt(question)
        sql_query = agents["sql_translator"].run(sql_prompt)
        print(f"Generated SQL: {sql_query}")

        validation_result = tools["execute_sql"].forward(sql_query)

        if "Error executing SQL" in str(validation_result) or len(str(validation_result)) < 3:
            print(f"Query failed or returned empty: {validation_result}")
            return None

        print(f"SQL validation successful! Found {str(validation_result)[:100]} results.")

        # Get the dirty table from prompt manager
        dirty_table = prompt_manager.table_manager.get_durty_table()
        
        # Execute SQL query to get the answer
        conn = prompt_manager.table_manager.conn
        sql_executor = SQLExecutor(conn=conn)
        sql_result = sql_executor.forward(sql_query)
        
        answer_llm = str(sql_result).strip()
        print(f"SQL execution result: {answer_llm}")
        
        logger.debug(f"is_use_fine_tuned_model_loaded: {is_use_fine_tuned_model_loaded}")
        
        if is_use_fine_tuned_model_loaded:
            # Use likelihood-based filtering to determine if we should keep this example
            
            # Get device from model
            device = next(fine_tuned_model.parameters()).device
            
            # Ensure dirty_table is a pandas DataFrame and properly formatted
            if not isinstance(dirty_table, pd.DataFrame):
                try:
                    # If it's a dictionary with header and rows
                    if isinstance(dirty_table, dict) and 'header' in dirty_table and 'rows' in dirty_table:
                        dirty_table = pd.DataFrame(dirty_table['rows'], columns=dirty_table['header'])
                    else:
                        dirty_table = pd.DataFrame(dirty_table)
                except Exception as e:
                    logger.error(f"Failed to convert table to DataFrame: {e}")
                    return None
            
            logger.debug(f"Table type: {type(dirty_table)}")
            logger.debug(f"Table columns: {dirty_table.columns.tolist()}")
            logger.debug(f"Table shape: {dirty_table.shape}")
            
            # Preprocess the input for likelihood calculation
            model_inputs = preprocess_for_verification(
                table=dirty_table,
                question=question,
                tokenizer=tokenizer,
                max_source_length=1024
            )
            
            logger.debug(f"Model inputs prepared")

            # Move inputs to device
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)
            
            try:
                # Calculate likelihood of the correct answer
                log_likelihood, avg_log_likelihood = calculate_answer_likelihood(
                    model=fine_tuned_model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_answer=answer_llm.lower(),
                    device=device
                )
                
                logger.debug(f"Log likelihood: {log_likelihood}")
                logger.debug(f"Average log likelihood: {avg_log_likelihood}")
                
                # Filter based on likelihood thresholds
                if avg_log_likelihood > likelihood_threshold_high:
                    print(f"Example too easy (avg log likelihood: {avg_log_likelihood:.3f} > {likelihood_threshold_high}) - SKIPPING")
                    return None
                elif avg_log_likelihood < likelihood_threshold_low:
                    print(f"Example too difficult (avg log likelihood: {avg_log_likelihood:.3f} < {likelihood_threshold_low}) - SKIPPING")
                    return None
                else:
                    print(f"Example has good difficulty (avg log likelihood: {avg_log_likelihood:.3f}) - KEEPING")
                    
            except Exception as e:
                logger.error(f"Error calculating likelihood: {e}")
                # If likelihood calculation fails, skip this example
                print("Failed to calculate likelihood - SKIPPING")
                return None

        # Generate entry and variations (only if we reach here)
        entry = []
        vector_id = str(uuid4())
        tools["retriever_tool"].vectordb.add_documents(
            documents=[Document(question)], ids=[vector_id]
        )
        entry.append({
            "tables_id": table_id,
            "question": question.lower(),
            "sql": sql_query,
            "orginal": True
        })

        # Generate variations
        diversity_prompt = prompt_manager.get_extra_prompt_divers(question, sql_query)
        variations = agents["question_diversity"].run(diversity_prompt)
        print(f"Generated variations: {variations}")

        for variation in variations:
            vector_id = str(uuid4())
            tools["retriever_tool"].vectordb.add_documents(
                documents=[Document(str(variation["question"]))], ids=[vector_id]
            )
            entry.append({
                "tables_id": table_id,
                "question": variation["question"].lower(),
                "sql": variation["sql"],
                "orginal": False
            })

        return entry

    except Exception as e:
        # print in red
        print(f"\033[91mPipeline step failed: {e}\033[0m")
        return None