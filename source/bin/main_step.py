from typing import Dict, Any, Optional
from source.data_modules.sql_executor import SQLExecutor


def run_pipeline_step(prompt_manager, 
                      agents,
                      answer_checker) -> Optional[Dict[str, Any]]:

    table_id = prompt_manager.table_manager.current_table_id
    inside = (True, "question", "difficulty", 0)
    print("Generating a new entry...")

    try:
        # Generate a question
        question_prompt = prompt_manager.get_question_prompt()
        question = agents["question_generator"].run(question_prompt)
        print(f"Generated question: {question}")

        # Translate to SQL
        sql_prompt = prompt_manager.get_traductor_prompt(question)
        sql_query = agents["sql_translator"].run(sql_prompt)



        executor = SQLExecutor(prompt_manager.table_manager.conn)
        expected_answer = executor.forward(sql_query)
        
        if isinstance(expected_answer, str) and expected_answer.startswith("Error executing SQL"):
            return None, inside


        expected_answer = [str(cell) for row in expected_answer for cell in row if cell is not None]



        if answer_checker is not None:
            print('checking answer...')
            print('Table ID:', table_id)
            print('Question:', question)
            table = prompt_manager.table_manager.get_durty_table()
            print('Table:', table)
            model_answer, inside = answer_checker.check_answer(table, question, expected_answer)
            print('Model answer:', model_answer)
            print('Expected answer:', expected_answer)  
            print('Log likelihood:', inside[3])
            print('Inside:', inside[0])
            print('Inside question:', inside[1])
            print('Inside difficulty:', inside[2])

            if not inside[0]:
                print('Answer is either too similar or not valid, skipping entry generation.')
                return None, inside
            

        # Generate entry and variations
        entry = []

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

            entry.append({

                "tables_id": table_id,
                "question": variation["question"].lower(),
                "sql": sql_query,
                "orginal": False
            })

        return entry, inside

    except Exception as e:
        print(f"Pipeline step failed: {e}")
        return None, inside