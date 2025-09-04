from transformers import pipeline

def classify_text(text):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["Event", "Meeting", "Workshop", "Spam", "Advertisement", "Update"]
    res = classifier(
        text[:1000],
        candidate_labels=candidate_labels
    )
    print(res['scores'])

    max_score = max(res['scores'])
    for i in range(len(res['scores'])):
        if res['scores'][i] == max_score:
            return candidate_labels[i]
    return candidate_labels[0]


'''
message = ""
with open("passage.txt", "r") as file:
    message = file.read()

print(classify_text(message))
'''

