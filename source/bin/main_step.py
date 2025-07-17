from typing import Dict, Any, Optional
from source.data_modules.sql_executor import SQLExecutor
from source.utils.logger import setup_logger
logger = setup_logger(__name__)



def run_pipeline_step(prompt_manager, 
                      agents,
                      answer_checker) -> Optional[Dict[str, Any]]:

    table_id = prompt_manager.table_manager.current_table_id
    inside = (True, "question", "difficulty", 0)
    logger.info("Generating a new entry...")

    try:
        # Generate a question
        question_prompt = prompt_manager.get_question_prompt()
        question = agents["question_generator"].run(question_prompt)
        logger.info(f"Generated question: {question}")

        # Translate to SQL
        sql_prompt = prompt_manager.get_traductor_prompt(question)
        sql_query = agents["sql_translator"].run(sql_prompt)



        executor = SQLExecutor(prompt_manager.table_manager.conn)
        expected_answer = executor.forward(sql_query)
        
        if isinstance(expected_answer, str) and expected_answer.startswith("Error executing SQL"):
            return None, inside


        expected_answer = [str(cell) for row in expected_answer for cell in row if cell is not None]



        if answer_checker is not None:
            logger.info('checking answer...')
            logger.info('Table ID: %s', table_id)
            logger.info('Question: %s', question)
            table = prompt_manager.table_manager.get_durty_table()
            logger.info('Table: %s', table)
            model_answer, inside = answer_checker.check_answer(table, question, expected_answer)
            logger.info('Model answer: %s', model_answer)
            logger.info('Expected answer: %s', expected_answer)
            logger.info('Log likelihood: %s', inside[3])
            logger.info('Inside: %s', inside[0])
            logger.info('Inside question: %s', inside[1])
            logger.info('Inside difficulty: %s', inside[2])

            if not inside[0]:
                logger.info('Answer is either too similar or not valid, skipping entry generation.')
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
        logger.info(f"Generated variations: {variations}")

        for variation in variations:

            entry.append({

                "tables_id": table_id,
                "question": variation["question"].lower(),
                "sql": sql_query,
                "orginal": False
            })

        return entry, inside

    except Exception as e:
        logger.info(f"Pipeline step failed: {e}")
        return None, inside