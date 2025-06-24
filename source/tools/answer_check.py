from difflib import SequenceMatcher

class AnswerChecker:
    def __init__(self, model, tokenizer, metric, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.metric = metric
        self.device = device

    def check_answer(self, question: str, expected_answer: str) -> tuple[bool, str, float]:

        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)

        try:
            output_ids = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            model_answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            denotation_accuracy = self.metric(model_answer, expected_answer)

            return denotation_accuracy

        except Exception as e:
            return False, f"Error generating answer: {str(e)}"