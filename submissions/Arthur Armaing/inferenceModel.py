from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import re

class MultiTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_ids, device):
        self.stop_ids = stop_ids
        self.stop_len = len(stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) >= self.stop_len:
            last_tokens = input_ids[0][-self.stop_len:].tolist()
            return last_tokens == self.stop_ids
        return False


def get_model_inference(event_message):
    model_name = "HIT-TMG/EviOmni-nq_train-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #formulate prompt and fill in stuff
    prompt = open("eviomni_prompt", "r").read()
    passages = event_message
    
    question = ""
    with open("prompt.txt", 'r', encoding='utf-8') as file:
        question = file.read()

    instruction = prompt.format(question=question, passages=passages)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]

    stop_token = "</extract>\n\n"
    stop_ids = tokenizer.encode(stop_token, add_special_tokens=False)

    stopping_criteria = StoppingCriteriaList([
        MultiTokenStoppingCriteria(stop_ids, model.device)
    ])

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    match = re.search(r"<extract>(.*?)</extract>", response, re.DOTALL)
    evidence = match.group(1).strip()
    return evidence


if __name__ == "__main__":
    with open("passage.txt", "r") as file:
        msg = file.read()
    print(get_model_inference(msg))

