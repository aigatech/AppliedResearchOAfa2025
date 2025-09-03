import matplotlib.pyplot as plt
from transformers import pipeline

topic = input("\nEnter the topic you want to hear a debate about: ")
rounds = int(input("How many rounds should the two sides debate for (Enter an integer)? : "))

debaters = pipeline("text-generation", model="gpt2-large", pad_token_id = 50256)
stance_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sides = {"Pro": debaters, "Con": debaters}
history, scores = [], {"Pro": [], "Con": []}

for rnd in range(1, rounds + 1):
    print(f"\n=== Round {rnd} ===")
    question = input("Enter a question to ask the debaters: ")

    for side, model in sides.items():
        # Simple prompt without file reading
        with open ("prompt.txt", "r") as file:
            directions = file.read()
        prompt = (f"Directions: {directions}\n"
                  f"Debate Topic: {topic}\n"
                  f"Side: {side}\n"
                  f"Question: {question}\n"
                  f"Give a brief {side} argument (1-2 sentences):")

        response = model(
            prompt,
            max_new_tokens=50,  
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )[0]["generated_text"]

        response = response.replace(prompt, "").strip()
        
        sentences = response.split('.')
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'

        align = stance_clf(response, candidate_labels=[side])["scores"][0]
        pers = align * sentiment(response)[0]["score"]

        scores[side].append((align, pers))
        history.append(f"{side}: {response}")

        print(f"\n{side} says: {response}\n[align={align:.2f}, pers={pers:.2f}]")

user_vote = input("\nWho do you think won the debate? (Pro/Con): ")
print(f"You voted: {user_vote}")

judge_history = "\n".join(history[-10:])  # only last 10 entries
with open ("judge_prompt.txt", "r") as file:
    judge_directions = file.read()
judge_prompt = f"""Debate transcript:\n{judge_history}\n
                Directions: {judge_directions}\n
                Who won the debate (Pro/Con)? Explain why."""

judge = debaters(
    judge_prompt,
    max_new_tokens=150,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
)[0]["generated_text"]

judge = judge.replace(judge_prompt, "").strip()
print("\nAI Judge Verdict:", judge)

for side in sides:
    pers_scores = [p for (_, p) in scores[side]]  # Fixed variable name
    plt.plot(range(1, len(pers_scores)+1), pers_scores, marker="o", label=side)

plt.xlabel("Round")
plt.ylabel("Persuasiveness Score (Alignment × Sentiment)")
plt.title(f"Debate Persuasiveness: {topic}")
plt.ylim(0,1)  # optional: fix y-axis scale
plt.legend()
plt.show()