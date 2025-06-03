def calculate_answer_likelihood(model, tokenizer, input_ids, attention_mask, target_answer, device):
    """
    Calculate how likely the model is to generate the target answer
    Improved version with better handling of complex SQL results
    """
    import torch.nn.functional as F
    import difflib
    
    try:
        # Use the improved cleaning function with all values
        cleaned_target = format_sql_result_for_tapex(str(target_answer), include_all_values=True).lower().strip()
        if not cleaned_target:
            return float('-inf'), float('-inf')
        
        # Get vocabulary size for validation
        vocab_size = model.config.vocab_size
        logger.debug(f"Model vocab_size: {vocab_size}")
        
        with torch.no_grad():
            # First, try forced decoding to get exact likelihood
            try:
                # For TAPEX/BART models, we need to handle tokenization carefully
                with tokenizer.as_target_tokenizer():
                    target_encoding = tokenizer(
                        answer = cleaned_target,  # FIX: Direct string, not answer parameter
                        max_length=128,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"  
                    )
                    logger.debug(f"target_encoding keys: {target_encoding.keys()}")
                
                # FIX: Proper dictionary access and move to device
                target_ids_tensor = target_encoding["input_ids"][0].to(device)
                logger.debug(f"Raw target_ids_tensor: {target_ids_tensor}")
                
                # FIX: Process tokens properly - filter out padding and validate
                target_ids = []
                for token_id in target_ids_tensor.cpu().tolist():
                    if token_id != tokenizer.pad_token_id and token_id != -100:
                        # CRITICAL FIX: Validate token ID is within vocabulary range
                        if token_id < vocab_size:
                            target_ids.append(token_id)
                        else:
                            logger.warning(f"Skipping invalid token ID: {token_id} (vocab_size: {vocab_size})")
                
                if not target_ids:
                    logger.warning("No valid target tokens found, falling back to generation")
                    raise ValueError("No valid target tokens")

                logger.debug(f"Valid target_ids: {target_ids}")

                # Get the decoder start token id
                decoder_start_token_id = model.config.decoder_start_token_id

                if decoder_start_token_id is None:
                    decoder_start_token_id = tokenizer.bos_token_id
                if decoder_start_token_id is None:
                    decoder_start_token_id = tokenizer.pad_token_id

                # FIX: Validate decoder start token
                if decoder_start_token_id >= vocab_size:
                    logger.warning(f"Invalid decoder start token {decoder_start_token_id}, using 0")
                    decoder_start_token_id = 0

                logger.debug(f"decoder_start_token_id: {decoder_start_token_id}")

                # Create decoder input ids
                decoder_input_ids = torch.tensor([[decoder_start_token_id] + target_ids], device=device)
                
                # FIX: Final safety check
                if torch.any(decoder_input_ids >= vocab_size):
                    logger.error(f"Token IDs still out of range. Max ID: {torch.max(decoder_input_ids)}, vocab_size: {vocab_size}")
                    raise ValueError("Token IDs out of range after validation")
                
                logger.debug(f"decoder_input_ids: {decoder_input_ids}")

                # Create labels (shift decoder input ids by one position)
                labels = decoder_input_ids[:, 1:].contiguous()
                decoder_input_ids = decoder_input_ids[:, :-1].contiguous()
                
                logger.debug(f"Final decoder_input_ids: {decoder_input_ids}")
                logger.debug(f"Final labels: {labels}")

                # Get model outputs with teacher forcing
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_dict=True
                )
                
                # Faire un model pour voir les outputs
                outputs_str = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                logger.debug(f"outputs_str: {outputs_str}")
                logger.debug(f"Model forward pass successful")
                
                # Calculate average log likelihood from loss
                # Loss is negative log likelihood, so negate it
                avg_log_likelihood = -outputs.loss.item()
                logger.debug(f"avg_log_likelihood: {avg_log_likelihood}")
                log_likelihood = avg_log_likelihood * len(target_ids)
                likelihood_type = "FORCED"
                
                logger.debug(f"Target: '{target_answer}' -> Cleaned: '{cleaned_target}' | "
                            f"avg_log_likelihood: {avg_log_likelihood:.3f} ({likelihood_type})")
                
                return log_likelihood, avg_log_likelihood
                
            except Exception as e:
                logger.debug(f"Forced decoding failed: {e}, falling back to generation")
                
                # Fallback: Generate and compare
                generated_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=64,
                    num_beams=3,  
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                )
                
                # Get the generated sequence
                generated_ids = generated_outputs.sequences[0]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
                
                # Check for exact match FIRST
                if generated_text == cleaned_target:
                    # EXACT MATCH: Calculate real likelihood from generation scores
                    if hasattr(generated_outputs, 'scores') and generated_outputs.scores:
                        log_probs = []
                        # Skip the decoder start token
                        start_idx = 1
                        generated_tokens = generated_ids[start_idx:]
                        
                        for i, score in enumerate(generated_outputs.scores):
                            if i < len(generated_tokens):
                                token_id = generated_tokens[i]
                                # Ensure token_id is valid
                                if token_id < score.shape[-1]:
                                    log_prob = F.log_softmax(score[0], dim=-1)[token_id].item()
                                    log_probs.append(log_prob)
                        
                        if log_probs:
                            log_likelihood = sum(log_probs)
                            avg_log_likelihood = log_likelihood / len(log_probs)
                            likelihood_type = "EXACT_MATCH"
                        else:
                            # Exact match but no valid scores - use high confidence
                            avg_log_likelihood = -0.5
                            log_likelihood = avg_log_likelihood * len(cleaned_target.split())
                            likelihood_type = "EXACT_MATCH_DEFAULT"
                    else:
                        # Exact match but no scores - use high confidence
                        avg_log_likelihood = -0.5
                        log_likelihood = avg_log_likelihood * len(cleaned_target.split())
                        likelihood_type = "EXACT_MATCH_DEFAULT"
                else:
                    # NOT exact match: Calculate similarity-based likelihood
                    similarity = difflib.SequenceMatcher(None, generated_text, cleaned_target).ratio()
                    
                    # Adjust the likelihood scale based on answer complexity
                    answer_length = len(cleaned_target.split())
                    if answer_length == 1:
                        # Single token answers: stricter scoring
                        avg_log_likelihood = -0.5 - (1.0 - similarity) * 10.0
                    elif answer_length <= 5:
                        # Short answers: moderate scoring
                        avg_log_likelihood = -1.0 - (1.0 - similarity) * 12.0
                    else:
                        # Long answers: more lenient scoring
                        avg_log_likelihood = -2.0 - (1.0 - similarity) * 8.0
                    
                    log_likelihood = avg_log_likelihood * max(answer_length, 1)
                    likelihood_type = "SIMILARITY"
        
        logger.debug(f"Target: '{target_answer}' -> Cleaned: '{cleaned_target}' | "
                    f"Model answer: '{generated_text}' | "
                    f"avg_log_likelihood: {avg_log_likelihood:.3f} ({likelihood_type})")
        
        return log_likelihood, avg_log_likelihood
        
    except Exception as e:
        logger.error(f"Likelihood calculation failed: {str(e)}")
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
    # Dynamic thresholds based on complexity
    if answer_complexity == 1:
        # Single value answers
        threshold_high = -1.0
        threshold_low = -8.0
    elif answer_complexity <= 5:
        # Multi-value answers (2-5 values)
        threshold_high = -2.0
        threshold_low = -12.0
    else:
        # Complex answers (>5 values)
        threshold_high = -3.0
        threshold_low = -15.0
    
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
    