Canvas Content Injection Bot

This bot takes in course and assignment information from your Canvas account and answers any question regarding that information.

To run this code, first, get a Canvas personal access token. To do so, navigate to your Canvas base URL (https://gatech.instructure.com/), select Account, select Settings, click on + New Access Token, set a purpose and expiration date, and copy and the generated token. Next, install app requirements from the CLI (pip install -r requirements.txt). Finally, run the app, either from the CLI (python canvas_context_injection_bot.py) or from your IDE. You will be prompted to enter the Canvas access token generated, then you can ask any question and get a response from the HuggingFace model. Note: the LLM used by HuggingFace took approximately 1 minute to load while testing.
