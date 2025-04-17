
import yaml
from jinja2 import Template

class PromptManager:

    def __init__(self, data_args, table_manager, library):

        self.table_manager = table_manager
        tables_info, table_samples = table_manager.get_table()
        self.library = library
        self.table_info_str = ', '.join(tables_info['tables'])
        self.table_schema_str = self.table_schema(tables_info)
        self.sample_data_str = self.sample_data(table_samples)
        self.get_examples_str = self.get_examples()
        self.prompt_templates = self.load_prompt_templates(data_args.base_prompt_path)


    def table_schema(self, tables_info):

        prompt = ""
        for table, columns in tables_info['schemas'].items():
            prompt += f"\nTable: {table}\n"
            for column in columns:
                prompt += f"  - {column}\n"

        return prompt
    

    def sample_data(self, table_samples):
        prompt = "\nSAMPLE DATA:\n"
        for table, rows in table_samples.items():
            prompt += f"\nTable: {table} (showing {len(rows)} rows)\n"
            for i, row in enumerate(rows):
                prompt += f"  Row {i+1}: {row}\n"
        return prompt


    def get_examples(self):
        prompt = ""
        for i in range(min(3, len(self.library))):
            idx = len(self.library) - i - 1
            prompt += f"\n- {self.library[idx]['question']}"

        return prompt

    def load_prompt_templates(self, prompt_path):
        with open(prompt_path, "r") as file:
            return yaml.safe_load(file)
        

    def base_prompt(self):

        template_str = self.prompt_templates["base_prompt"]["template"]
        template = Template(template_str)

        return template.render(
            table_info=self.table_info_str,
            table_schema=self.table_schema_str,
            sample_data=self.sample_data_str
        ).strip()
    

    def get_question_prompt(self):
 
        template_str = self.prompt_templates["get_question_prompt"]["template"]
        template = Template(template_str)

        base_prompt = self.base_prompt()

        return template.render(
            base_prompt=base_prompt,
            library=self.library,
            library_size=len(self.library),
            example_questions=self.get_examples_str
        ).strip()
    



    def get_traductor_prompt(self, question):

        template_str = self.prompt_templates["get_traductor_prompt"]["template"]
        template = Template(template_str)

        base_prompt = self.base_prompt()

        return template.render(
            base_prompt=base_prompt,
            question=question
        ).strip()


    def get_extra_prompt_divers(self, question, sql_question):

        template_str = self.prompt_templates["get_extra_prompt_divers"]["template"]
        template = Template(template_str)

        return template.render(
            question=question,
            sql_question=sql_question,
            table_schema=self.table_schema_str
        ).strip()
    
