from typing import Dict, Any, Optional
from uuid import uuid4
from langchain_core.documents import Document
from source.step1.verification import preprocess_for_verification, evaluate_example, verify_sql_answer
import torch
import pandas as pd
#from source.tools.sql_executor import SQLExecutor
from source.prepare_data.sql_executor import SQLExecutor

# add logger
import logging
logger = logging.getLogger(__name__)

# print logger debug
logger.setLevel(logging.DEBUG)



def run_pipeline_step(prompt_manager, 
                      agents,
                      tools, 
                      fine_tuned_model,
                      tokenizer, 
                      is_use_fine_tuned_model_loaded=False) -> Optional[Dict[str, Any]]:

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

        ##### Check if the model already knows the answer

        # Get the dirty table from prompt manager
        dirty_table = prompt_manager.table_manager.get_durty_table() # get_dirty_table ?
        
        # Execute SQL query to get the answer / Use SQLExecutor class
        # Initialize SQLExecutor with the connection from prompt_manager
        # Following the pattern in columnwise_row_permuter.py and sql_executor.py in prepare_data
        conn = prompt_manager.table_manager.conn # or table_manager.connect_to_database() ??
        sql_executor = SQLExecutor(conn=conn)
        sql_result = sql_executor.forward(sql_query)
        
        answer_llm = str(sql_result).strip()
        print(f"SQL execution result: {answer_llm}")
        
        logger.debug(f"is_use_fine_tuned_model_loaded: {is_use_fine_tuned_model_loaded}")
        if is_use_fine_tuned_model_loaded:
            # Verify the answer using the fine-tuned model
        
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
            
            logger.debug(f"dirty_table Done")
            logger.debug(f"Table type: {type(dirty_table)}")
            logger.debug(f"Table columns: {dirty_table.columns.tolist()}")
            logger.debug(f"Table shape: {dirty_table.shape}")
            
            # Preprocess the input for verification
            model_inputs = preprocess_for_verification(
                table=dirty_table,
                question=question,
                tokenizer=tokenizer,
                max_source_length=1024
            )
            
            logger.debug(f"model_inputs Done")

            # Move inputs to device
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)
            
            # Generate answer with the fine-tuned model
            with torch.no_grad():
                generated_ids = fine_tuned_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode the generated answer
            generated_answer = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0].strip().lower()
            
            # Compare answers using denotation accuracy
            is_equivalent = evaluate_example(answer_llm.lower(), generated_answer)
            
            if  is_equivalent:
                print("Model can answer the question")
                return None
            
            
            
            # Log the results
            logger.debug("@@@@@@@"*5)
            logger.debug(f"generated_answer: {generated_answer}")
            logger.debug(f"answer_llm: {answer_llm}")
            logger.debug(f"question: {question}")
            logger.debug(f"sql_query: {sql_query}")


        # Generate entry and variations
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


        