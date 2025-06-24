import torch 
import warnings



class AnswerChecker:
    def __init__(self, model, tokenizer, data_args, lower_thresh=None, upper_thresh=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data_args = data_args
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        
        self.model.eval()
    

    def check_answer(self, table, question, expected_answer):

        model_inputs = self.tokenizer(
                table=table,
                query=question,
                answer=expected_answer,
                max_length=self.data_args.max_source_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                truncation=True,
                return_tensors="pt",
            )
        

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    answer=[", ".join(expected_answer)],
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
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.data_args.max_target_length,
                    num_beams=5,
                )
            model_answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

        loss = outputs.loss  # This is already averaged over non-masked tokens
        log_likelihood = -loss.item()

        if self.lower_thresh is not None:
            inside = True
            if log_likelihood < self.lower_thresh:
                inside = False
            elif log_likelihood > self.upper_thresh:
                inside = False

        return model_answer, log_likelihood, inside