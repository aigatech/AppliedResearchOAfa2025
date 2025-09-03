# Sneha Narayan's Project: Comparing Text Sentiments
Sneha Narayan's AI@GT Applied Research Online Assessment

---

##  What does it do?
My project compares the sentiment of 2 texts that the user inputs. It will indicate which of the 2 texts has a more positive connotation. 
   
##  How to set it up
1. In line 10, enter the following as the model name: "distilbert-base-uncased-finetuned-sst-2-english"
Line 10 should look like this now: model = pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")
2. Install transformers and datasets as listed below:
transformers==4.56.0
datasets==4.0.0

##  How to run it
1. Open the sentiment_analysis.py file.
2. Follow the set up instructions.
3. Run the code.
4. Input the first piece of text in the terminal and press enter.
5. Input the second piece of text in the terminal and press enter. 
