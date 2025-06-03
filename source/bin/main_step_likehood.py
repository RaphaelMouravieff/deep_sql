from typing import Dict, Any, Optional
from uuid import uuid4
from langchain_core.documents import Document
from source.step1.verification import preprocess_for_verification
import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from source.prepare_data.sql_executor import SQLExecutor
import torch.nn.functional as F
import os
# add logger
import logging
import csv

# print logger debug
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def clean_sql_result(result_str):
    """
    Clean SQL result to match model's expected output format
    Handles various SQL result formats including:
    - Single values: [('platinum',)]
    - Multiple rows: [('1821', 1208192.0), ('1831', 1443041.0), ...]
    - Multiple columns: [('1801', 171202.0, 73268.0, 40642.0, 215382.0)]
    """
    import re
    import ast
    
    try:
        # Try to parse as Python literal
        parsed = ast.literal_eval(result_str)
        
        if isinstance(parsed, list) and len(parsed) > 0:
            # Single row, single value
            if len(parsed) == 1 and isinstance(parsed[0], tuple) and len(parsed[0]) == 1:
                return str(parsed[0][0]).strip()
            
            # Multiple rows or columns - concatenate with delimiter
            results = []
            for row in parsed:
                if isinstance(row, tuple):
                    # Join all values in the row with spaces
                    row_str = ' '.join(str(val) for val in row)
                    results.append(row_str)
                else:
                    results.append(str(row))
            
            # Join all rows with comma delimiter (matching TAPEX format)
            return ', '.join(results).strip()
        
        # Fallback for non-list results
        return str(result_str).strip()
        
    except:
        # Last resort: return as is
        return str(result_str).strip()


def format_sql_result_for_tapex(result_str, include_all_values=True):
    """
    Format SQL result specifically for TAPEX model expectations
    TAPEX typically expects comma-separated values for multi-value answers
    
    Args:
        result_str: String representation of SQL result
        include_all_values: If True, include all values from tuples; if False, only first column
    """
    import ast
    
    try:
        parsed = ast.literal_eval(result_str)
        
        if isinstance(parsed, list) and len(parsed) > 0:
            # For aggregation results (single row, single value)
            if len(parsed) == 1 and isinstance(parsed[0], tuple) and len(parsed[0]) == 1:
                value = parsed[0][0]
                # Handle numeric formatting
                if isinstance(value, (int, float)):
                    # Remove decimal point for whole numbers
                    if isinstance(value, float) and value.is_integer():
                        return str(int(value))
                    return str(value)
                return str(value).strip()
            
            # For multi-row results
            if all(isinstance(row, tuple) for row in parsed):
                if include_all_values:
                    # Flatten ALL values from all tuples into a single list
                    all_values = []
                    for row in parsed:
                        for val in row:
                            all_values.append(str(val))
                    return ', '.join(all_values)
                else:
                    # Extract only the first column (original behavior)
                    values = [str(row[0]) for row in parsed]
                    return ', '.join(values)
        
        return str(result_str).strip()
        
    except:
        return str(result_str).strip()


def calculate_answer_likelihood(model, tokenizer, input_ids, attention_mask, target_answer, device):
    """
    Calculate how likely the model is to generate the target answer.
    Uses the CORRECT log probability calculation as shown by your friend.
    """
    import torch.nn.functional as F
    import difflib
    
    try:
        # Use the improved cleaning function with all values
        cleaned_target = format_sql_result_for_tapex(str(target_answer), include_all_values=True).lower().strip()
        if not cleaned_target:
            return float('-inf'), float('-inf')
        
        with torch.no_grad():
            # First, try to calculate EXACT likelihood using the correct method
            try:
                # Tokenize the target answer with special tokens
                target_encoding = tokenizer(
                    answer = cleaned_target, 
                    max_length=128,
                    add_special_tokens=True,  # BART needs <s> and </s>
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"  
                )
                
                target_ids = target_encoding.input_ids.to(device)
                
                # For BART: decoder_input_ids = <s> token1 token2 ...
                #           labels = token1 token2 ... </s>
                decoder_input_ids = target_ids[:, :-1]
                labels = target_ids[:, 1:]
                
                # Get model outputs WITHOUT loss calculation
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True
                )
                
                # Get raw logits
                logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
                
                # Calculate log probabilities (this is what your friend showed)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Gather log probabilities of the correct tokens
                labels_expanded = labels.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
                selected_log_probs = log_probs.gather(2, labels_expanded).squeeze(-1)
                # Shape: [batch_size, seq_len]
                
                # Handle padding tokens
                pad_mask = labels != tokenizer.pad_token_id
                selected_log_probs = selected_log_probs * pad_mask
                
                # Calculate total and average log likelihood
                total_log_likelihood = selected_log_probs.sum().item()
                num_real_tokens = pad_mask.sum().item()
                
                if num_real_tokens > 0:
                    avg_log_likelihood = total_log_likelihood / num_real_tokens
                    likelihood_type = "EXACT_LIKELIHOOD"

                    # save the log likelihood to a file or not 
                    save_file = False
                    
                    if not save_file:
                        logger.debug(f"Target: '{target_answer}' -> Cleaned: '{cleaned_target}' | "
                                    f"avg_log_likelihood: {avg_log_likelihood:.3f} ({likelihood_type}) | "
                                    f"num_tokens: {num_real_tokens}")
                        
                    else:
                        # Generate answer with the fine-tuned model
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=128,
                                early_stopping=True
                            )
                        
                        # Decode the generated answer
                        outputs_str = tokenizer.batch_decode(
                            generated_ids, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=True
                        )[0].strip()

                        logger.debug(f"Target: '{target_answer}' -> Cleaned: '{cleaned_target}' | Outputs: '{outputs_str}' | "
                                    f"avg_log_likelihood: {avg_log_likelihood:.3f} ({likelihood_type}) | "
                                    f"num_tokens: {num_real_tokens}")
                        # Define CSV file path
                        csv_file = "outputs.csv"
                        
                        # Check if file exists to determine if we need to write headers
                        file_exists = os.path.exists(csv_file)
                        
                        # Open file in append mode
                        with open(csv_file, "a", newline='') as f:
                            writer = csv.writer(f)
                            
                            # Write headers if file is new
                            if not file_exists:
                                writer.writerow([
                                    "Target",
                                    "Cleaned",
                                    "Outputs",
                                    "Average Log Likelihood",
                                    "Likelihood Type",
                                    "Number of Tokens"
                                ])
                            
                            # Write data row
                            writer.writerow([
                                target_answer,
                                cleaned_target,
                                outputs_str,
                                f"{avg_log_likelihood:.3f}",
                                likelihood_type,
                                num_real_tokens
                            ])
                        
                        # Count number of examples (excluding header)
                        with open(csv_file, "r") as f:
                            count = sum(1 for _ in f) - 1  # Subtract 1 for header

                        # stop properly if count > 1000
                        if count > 1000:
                            import sys
                            sys.exit()
                            

                    return total_log_likelihood, avg_log_likelihood
                else:
                    raise ValueError("No valid tokens found")
                    
            except Exception as e:
                logger.debug(f"Exact likelihood calculation failed: {e}, falling back to generation")
                
                # Fallback: Generate and compare
                generated_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=3,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    early_stopping=True
                )
                
                # Get the generated sequence
                generated_ids = generated_outputs.sequences[0]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
                
                # Check for exact match
                if generated_text == cleaned_target:
                    # Try to extract likelihood from generation scores
                    if hasattr(generated_outputs, 'scores') and generated_outputs.scores:
                        log_probs = []
                        # Find where actual tokens start (after special tokens)
                        start_idx = 1 if generated_ids[0] in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] else 0
                        
                        for i, score in enumerate(generated_outputs.scores):
                            if i + start_idx < len(generated_ids):
                                token_id = generated_ids[i + start_idx]
                                if token_id < score.shape[-1]:
                                    log_prob = F.log_softmax(score[0], dim=-1)[token_id].item()
                                    log_probs.append(log_prob)
                        
                        if log_probs:
                            total_log_likelihood = sum(log_probs)
                            avg_log_likelihood = total_log_likelihood / len(log_probs)
                            likelihood_type = "GENERATION_EXACT"
                            
                            logger.debug(f"Target: '{target_answer}' -> Cleaned: '{cleaned_target}' | "
                                        f"Model answer: '{generated_text}' | "
                                        f"avg_log_likelihood: {avg_log_likelihood:.3f} ({likelihood_type})")
                            
                            return total_log_likelihood, avg_log_likelihood
                
                # If we get here, use similarity-based scoring
                similarity = difflib.SequenceMatcher(None, generated_text, cleaned_target).ratio()
                
                # Convert similarity to pseudo log-likelihood
                answer_length = len(cleaned_target.split())
                if answer_length == 1:
                    avg_log_likelihood = -0.5 - (1.0 - similarity) * 10.0
                elif answer_length <= 5:
                    avg_log_likelihood = -1.0 - (1.0 - similarity) * 12.0
                else:
                    avg_log_likelihood = -2.0 - (1.0 - similarity) * 8.0
                
                total_log_likelihood = avg_log_likelihood * max(answer_length, 1)
                likelihood_type = "SIMILARITY"
                
                logger.debug(f"Target: '{target_answer}' -> Cleaned: '{cleaned_target}' | "
                            f"Model answer: '{generated_text}' | "
                            f"avg_log_likelihood: {avg_log_likelihood:.3f} ({likelihood_type})")
                
                return total_log_likelihood, avg_log_likelihood
                
    except Exception as e:
        logger.error(f"Likelihood calculation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return float('-inf'), float('-inf')


def should_keep_example(avg_log_likelihood, answer_complexity):
    """
    Determine if an example should be kept based on likelihood and complexity
    
    Args:
        avg_log_likelihood: Average log likelihood of the answer
        answer_complexity: Measure of answer complexity (e.g., number of values)
    
    Returns:
        keep: Boolean indicating if example should be kept
        reason: String explaining the decision
    """
    
    threshold_high = -1.04
    threshold_low = -7.33
    
    # # Dynamic thresholds based on complexity
    # if answer_complexity == 1:
    #     # Single value answers
    #     threshold_high = -1.0
    #     threshold_low = -8.0
    # elif answer_complexity <= 5:
    #     # Multi-value answers (2-5 values)
    #     threshold_high = -2.0
    #     threshold_low = -12.0
    # else:
    #     # Complex answers (>5 values)
    #     threshold_high = -3.0
    #     threshold_low = -15.0
    
    if avg_log_likelihood > threshold_high:
        return False, f"Too easy (likelihood {avg_log_likelihood:.3f} > {threshold_high})"
    elif avg_log_likelihood < threshold_low:
        return False, f"Too difficult (likelihood {avg_log_likelihood:.3f} < {threshold_low})"
    else:
        return True, f"Good difficulty (likelihood {avg_log_likelihood:.3f})"


def get_answer_complexity(sql_result_str):
    """
    Calculate the complexity of an SQL answer
    
    Args:
        sql_result_str: String representation of SQL result
        
    Returns:
        complexity: Integer representing answer complexity
    """
    import ast
    
    try:
        parsed = ast.literal_eval(sql_result_str)
        if isinstance(parsed, list):
            # Count total number of values
            total_values = 0
            for row in parsed:
                if isinstance(row, tuple):
                    total_values += len(row)
                else:
                    total_values += 1
            return total_values
        return 1
    except:
        # If parsing fails, assume it's a simple answer
        return 1


def validate_and_fix_tokens(token_ids, vocab_size, tokenizer):
    """
    Validate and fix token IDs to prevent CUDA indexing errors
    """
    valid_tokens = []
    for token_id in token_ids:
        if token_id != tokenizer.pad_token_id and token_id != -100:
            if token_id < vocab_size:
                valid_tokens.append(token_id)
            else:
                logger.warning(f"Skipping invalid token ID: {token_id} (vocab_size: {vocab_size})")
    return valid_tokens   

def run_pipeline_step(prompt_manager, 
                      agents,
                      tools, 
                      fine_tuned_model,
                      tokenizer, 
                      is_use_fine_tuned_model_loaded=False,
                      reason_previous_skip=None) -> Optional[Dict[str, Any]]:  # Examples with avg log likelihood < this are too hard

    table_id = prompt_manager.table_manager.current_table_id

    print("Generating a new entry...")

    try:
        # Generate a question
        question_prompt = prompt_manager.get_question_prompt()
        if reason_previous_skip:
            question_prompt += f"\n\nPrevious execution failed with the following reason and question and answer: {reason_previous_skip}"
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
            
            # Ensure dirty_table is a pandas DataFrame
            if not isinstance(dirty_table, pd.DataFrame):
                try:
                    if isinstance(dirty_table, dict) and 'header' in dirty_table and 'rows' in dirty_table:
                        dirty_table = pd.DataFrame(dirty_table['rows'], columns=dirty_table['header'])
                    else:
                        dirty_table = pd.DataFrame(dirty_table)
                except Exception as e:
                    logger.error(f"Failed to convert table to DataFrame: {e}")
                    return None
            
            # Preprocess the input
            model_inputs = preprocess_for_verification(
                table=dirty_table,
                question=question,
                tokenizer=tokenizer,
                max_source_length=1024
            )
            
            # Move inputs to device
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)
            
            try:
                # Calculate answer complexity
                answer_complexity = get_answer_complexity(answer_llm)
                logger.debug(f"Answer complexity: {answer_complexity}")
                
                # Calculate likelihood of the correct answer
                log_likelihood, avg_log_likelihood = calculate_answer_likelihood(
                    model=fine_tuned_model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_answer=answer_llm,
                    device=device
                )
                
                # Determine if we should keep this example
                should_keep, reason = should_keep_example(avg_log_likelihood, answer_complexity)
                
                if not should_keep:
                    print(f"Example {reason} - SKIPPING")
                    logger.debug(f"Full details - Question: {question[:50]}... | Answer: {answer_llm[:50]}...")
                    #return None
                    # return the question and the answer 
                    return {
                        "question": question,
                        "answer": answer_llm, 
                        'reason': reason
                    }
                else:
                    print(f"Example {reason} - KEEPING")
                    
            except Exception as e:
                logger.error(f"Error in likelihood filtering: {e}")
                # If filtering fails, you might want to keep the example anyway
                print("Likelihood filtering failed, keeping example by default")
                # Or skip it: return None

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