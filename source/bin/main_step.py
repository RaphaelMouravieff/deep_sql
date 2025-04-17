from typing import Dict, Any, Optional


from uuid import uuid4
from langchain_core.documents import Document


def run_pipeline_step(prompt_manager, agents,
                            retriever_tool, execute_sql, max_attempts: int = 5,)-> Optional[Dict[str, Any]]:

    table_id = prompt_manager.table_manager.current_table_id

    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1}/{max_attempts} to generate a valid entry...")
      

        question_prompt = prompt_manager.get_question_prompt()
        question = agents["question_generator"].run(question_prompt)
        print(f"Generated question: {question}")

        
        sql_prompt = prompt_manager.get_traductor_prompt(question)
        sql_query = agents["sql_translator"].run(sql_prompt)
        print(f"Generated SQL: {sql_query}")

        try :
            validation_result=execute_sql.execute_it(sql_query)
        except Exception as e:
            validation_result='Error executing SQL"'
            
            print(f"Error executing SQL: {str(e)} for the query: {sql_query}")
            continue
        
        # 3. Validate the SQL query

        if "Error executing SQL" in str(validation_result) or len(str(validation_result))<3:
            print(f"Dose not work! Found {str(validation_result)[:100]} results.")
            continue
        
        print(f"SQL validation successful! Found {str(validation_result)[:100]} results.")
        
        # 5. Generate question variations
        entry = []
        diversity_prompt = prompt_manager.get_extra_prompt_divers(question, sql_query)
        variations = agents["question_diversity"].run(diversity_prompt)

        print(f"Generated variations: {variations}")

        vector_id=str(uuid4())
        retriever_tool.vectordb.add_documents(documents=[Document(question)], ids=[vector_id])
        entry.append({
                "vector_id":vector_id,
                "tables_id": table_id,
                "question": question,
                "sql": sql_query,
                "result": validation_result,
                "orginal":True
            })
        try:
            for varaition in variations:
                vector_id=str(uuid4())
                retriever_tool.vectordb.add_documents(documents=[Document(str(varaition["question"]))], ids=[vector_id])
                entry.append({
                    "vector_id":vector_id,
                    "tables_id": table_id,
                    "question": varaition["question"],
                    "sql": varaition["sql"],
                    "result": validation_result, #validation_result["results"],
                    "orginal":False
                })
        except Exception as e:
            
            
            print(f"Error executing SQL: {str(e)} ")
            continue

        return entry
    
    # If we've exhausted all attempts
    print("Failed to generate a valid entry after multiple attempts.")
    return None