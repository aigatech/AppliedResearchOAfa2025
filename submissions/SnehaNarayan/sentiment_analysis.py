#Sneha Narayan (GTID: 904122073) AIGT Applied Research OA Fall 2025
#gatech email: snarayan65@gatech.edu
#personal email: emailsnehan@gmail.com

from transformers import pipeline

print ("Welcome to the sentiment analyzer. This program will compare the sentiment of 2 texts and indicate which one is in a more positive tone.")
user_text1 = input("Please enter the first piece of text you would like to analyze: ")
user_text2 = input("Please enter the second piece of text you would like to analyze: ")
model = pipeline("sentiment-analysis")

sentiment_score1 = model(user_text1)
label1 = sentiment_score1[0]['label']
score1 = sentiment_score1[0]['score']

sentiment_score2 = model(user_text2)
label2 = sentiment_score2[0]['label']
score2 = sentiment_score2[0]['score']

#comparing the sentiment of the two texts
if (label1 == 'POSITIVE' and label2 == 'POSITIVE'):
    if (score1 > score2):
        print("Both texts have a positive connotation :), but the first text is more positive than the second text.")
    elif (score2 > score1):
        print("Both texts have a positive connotation :), but the second text is more positive than the first text.")
    else:
        print("Both texts have the same level of positivity!")
elif (label1 == 'NEGATIVE' and label2 == 'NEGATIVE'):
    if (score1 > score2):
        print("Both texts have a negative connotation :(, but the first text is more negative than the second text.")
    elif (score2 > score1):
        print("Both texts have a negative connotation :(, but the second text is more negative than the first text.")
    else:
        print("Both texts have the same level of negativity.")
elif (label1 == 'POSITIVE' and label2 == 'NEGATIVE'):
    print("The first text has a positive connotation while the second text has a negative connotation.")
elif (label1 == 'NEGATIVE' and label2 == 'POSITIVE'):
    print("The first text has a negative connotation while the second text has a positive connotation.")