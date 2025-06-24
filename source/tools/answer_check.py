import torch 

class AnswerChecker:
    def __init__(self, model, tokenizer, data_args, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data_args = data_args

    def check_answer(self, question, table, expected_answer):


        model_inputs = self.tokenizer(
                table=table,
                query=question,
                answer=expected_answer,
                max_length=self.data_args.max_source_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                truncation=True,
                return_tensors="pt",
            )

        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                answer=[", ".join(answer) for answer in expected_answer],
                max_length=self.data_args.max_target_length,
                padding=self.data_args.pad_to_max_length,
                truncation=True,
                return_tensors="pt"
            )

        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        labels = labels["input_ids"].to(self.device) 
                

        model_answer = None
        if self.data_args.output_generation:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.data_args.max_target_length,
                num_beams=5,
            )
            model_answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        loss = outputs.loss  # This is already averaged over non-masked tokens
        log_likelihood = -loss.item()


        return model_answer, log_likelihood
