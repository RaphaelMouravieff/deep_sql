from typing import Dict, Any, Optional
from uuid import uuid4
from langchain_core.documents import Document

def run_pipeline_step(prompt_manager, 
                      agents,
                      tools) -> Optional[Dict[str, Any]]:

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

        # Validate SQL
        validation_result = tools["execute_sql"].forward(sql_query)

        if "Error executing SQL" in str(validation_result) or len(str(validation_result)) < 3:
            print(f"Query failed or returned empty: {validation_result}")
            return None

        print(f"SQL validation successful! Found {str(validation_result)[:100]} results.")

        # Generate entry and variations
        entry = []
        vector_id = str(uuid4())
        tools["retriever_tool"].vectordb.add_documents(
            documents=[Document(question)], ids=[vector_id]
        )
        entry.append({
            "vector_id": vector_id,
            "tables_id": table_id,
            "question": question.lower(),
            "sql": sql_query,
            "result": validation_result,
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
                "vector_id": vector_id,
                "tables_id": table_id,
                "question": variation["question"].lower(),
                "sql": variation["sql"],
                "result": validation_result,
                "orginal": False
            })

        return entry

    except Exception as e:
        print(f"Pipeline step failed: {e}")
        return None