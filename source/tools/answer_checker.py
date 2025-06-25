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
        label_ids = labels["input_ids"].squeeze(0).to(self.device)  # shape: [seq_len]
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        label_ids = label_ids.unsqueeze(0)


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
                labels=label_ids,
                return_dict=True
            )

        loss = outputs.loss  # This is already averaged over non-masked tokens
        log_likelihood = -loss.item()

        inside = (True, "question", "", 0)
        print('Lower threshold:', self.lower_thresh)
        print('Upper threshold:', self.upper_thresh)
        if self.lower_thresh is not None:
            print('Log likelihood:', log_likelihood)

            if log_likelihood < self.lower_thresh:
                print('lower threshold exceeded')
                inside = (False, question, "too complex – exceeds typical question depth", log_likelihood)

            elif log_likelihood > self.upper_thresh:
                print('upper threshold exceeded')
                inside = (False, question, "too simple – lacks challenge", log_likelihood)

            else:
                inside = (True, question, "Good Difficulty Range", log_likelihood)

        return model_answer, inside