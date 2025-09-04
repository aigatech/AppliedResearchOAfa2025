from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_emojis = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😱",
    "love": "❤️",
    "surprise": "😲"
}

past_reviews = []

def do_one_review(text):
    result = emotion_classifier(text)[0]
    best = max(result, key=lambda x: x['score'])
    emotion = emotion_emojis.get(best['label'], "")
    print("review:", text)
    print("emotion:", best['label'], emotion, round(best['score'], 2))
    past_reviews.append((text, best['label'], emotion, best['score']))

def many_reviews():
    print("type reviews (say done to stop):")
    while True:
        txt = input("> ")
        if txt.lower() == "done":
            break
        do_one_review(txt)

def see_reviews():
    if len(past_reviews) == 0:
        print("no reviews yet")
    else:
        num = 1
        for r in past_reviews:
            print(str(num) + ".", r[0], "->", r[1], r[2], round(r[3], 2))
            num = num + 1

def main():
    print("welcome to the movie/tv show review analyzer!!")
    while True:
        print("1. write one review")
        print("2. write many reviews")
        print("3. see past reviews")
        print("4. quit")
        pick = input("choose: ")
        if pick == "1":
            t = input("enter review: ")
            do_one_review(t)
        elif pick == "2":
            many_reviews()
        elif pick == "3":
            see_reviews()
        elif pick == "4":
            break
        else:
            print("not valid")

main()
