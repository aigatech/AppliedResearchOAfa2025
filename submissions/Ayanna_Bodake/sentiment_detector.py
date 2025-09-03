from transformers import pipeline

#different headlines from different sources
data = {
    "CNN": [
        "Florida plans to end vaccine mandates statewide, including for schoolchildren",
        "Epstein survivor's message to Trump: 'This is no hoax'",
        "Historic California gold mining town overrun by fast-moving wildfire",
        "Travis Kelce is 'giddy' about his engagement to Taylor Swift"

    ],
    "Fox News": [
        "Trump admin axes Army program after majority of officers refused to participate",
        "Putin extends invitation to Zelenskyy as former CIA station chief warns of danger", 
        "Feds bust massive drug shipment from China in 'undeclared war against America'",
        "Cardiologist reveals the 'magic sauce' for living longer that's 'free and accessible'"
    ],
    "BBC": [
        "Hundreds of women with brooms join protests as Indonesia leader flies to China",
        "Australia-Israel relations have hit a low point. Behind the scenes, it's business as usual",
        "Israel intensifies Gaza City attacks as UN warns of 'horrific' consequences for displaced families",
        "Agyemang: 'Winning Euros was so surreal that I cried'"
    ]
}

#load in huggingface sentiment pipline
sentiment_pipeline = pipeline("sentiment-analysis")

#analysis
results = {}
for source, texts in data.items():
    predictions = sentiment_pipeline(texts)
    #how many positive headlines from source
    pos = 0
    for p in predictions:
        if p['label'] == 'POSITIVE':
            pos += 1
    #how many negative headlines from source
    neg = 0
    for p in predictions:
        if p['label'] == 'NEGATIVE':
            neg += 1
    #save number of pos/neg for each source
    results[source] = {"Positive": pos, "Negative": neg}

# Print results
print("Sentiment Summary by Source:")
for source, counts in results.items():
    print(f"{source}: {counts}")
