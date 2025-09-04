Wikipedia Reading-Check 

This project takes a Wikipedia topic as input and fetches the article text.

It then summarizes it using a lightweight model and generates one short reading-check question based on the summary.

Runs fully on CPU. No GPU required.

How to Run
1. Create and activate a virtual environment
python3 -m venv submissions/<your_name>>/venv
source submissions/<your_name>/venv/bin/activate

2. Install dependencies
pip install -r submissions/<your_name>>/requirements.txt

3. Run the program 
python submissions/<your_name>>/summarizer_qa.py


Then enter a topic, e.g.:

Enter a Wikipedia topic: cristiano ronaldo
or
lionel messi

Example Output
=== Summary ===
Cristiano Ronaldo dos Santos Aveiro is a Portuguese international footballer...

=== Reading Check ===
1. How many goals has Cristiano Ronaldo scored in the Champions League?

Models Used

Summarizer: sshleifer/distilbart-cnn-12-6

Question Generator: iarfmoose/t5-base-question-generator

Model Info:
First run downloads the model weights (~300MB summarizer, ~900MB QG).
After the first run, everything is cached locally in:

~/.cache/huggingface/hub/
