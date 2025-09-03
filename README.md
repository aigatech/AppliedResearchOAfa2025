# AI Topic Debater 
This program is intended to simulate two debaters arguing for a Pro/Con side for a user prompted topic. The debate will last X amount of rounds specified by the user. For each round, you will ask a question to each of the debaters who will then provide a response. 

For each round, an alignment, sentiment, and persuasiveness score will be assigned to each response. 

Alignment: measure of 0 to 1 tracking how close the produced response aligns with the side (Pro/Con) being represented
- Ex: 1 for the Pro means its argument really aligns with the Pro side argument
- Calculated by stance_clf using a zero-shot-classification approach

Sentiment: measure of 0 to 1 showing how emotionally intense the response was
- Ex: 0 for Pro means the response was very monotone and standard
- Calculated by sentiment

Persuasiveness: a custom metric calculated by persuasiveness = alignment * sentiment
- used for plotting quality of responses with matplotlib

After the debate has concluded, the entire transcript will be fed to a third "judge" AI that reads through the responses and decides and winner with explanation. The user will also be prompted to give their opinion.

Finally, a line plot will demonstrates the quality of persuasiveness for each AI response.

# Running the Code
Just run the file "VinhPham.py" and the program will begin in the terminal. First, type a statement you would like the AI to debate about. For example, "Phones should be allowed in school." Second, specify how many rounds (responses) each AI should provide. Once the debate begins, type out a question that you would like the AI to answer. For example, "How do phones affect learning?" This repeats until all rounds are finished. You have the chance to input which side you believe won and a third AI judge will also give its input.

A final line plot will display the persuasiveness value of each response from the Pro and Con side.

# Disclaimer
The code functions correctly but the AI models seem to be too small to generate actual responses with adequate token size. The AI often hallucinates and sometimes just repeats random questions. I cut myself off around after some time elapsed, so I couldn't go back to fix the models from hallucinating. Hopefully this is a fine proof of concept 🙏.

