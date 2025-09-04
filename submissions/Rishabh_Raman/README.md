\# NFL Play Tagger (Zero-Shot)



\*\*What it does\*\*  

Classifies short NFL play descriptions into one of:

`touchdown, field goal, turnover, penalty, pass, rush, timeout, injury`.



Uses a Hugging Face zero-shot model, so no training is needed—runs quickly on CPU. Includes a tiny heuristic (e.g., “sack”) to show how simple rules can complement ML.



\*\*Why NFL?\*\*  

As A Georgia Tech Student,  I’m interested in sports analytics and all sorts of games and fun including Clash Royale, and this project will help me for the Research Position team as it help me model games! This is NFL in honor of the season starting tomorrow and I cannot wait! Football play text is structured and event-driven, so labeling lines is a realistic first step toward drive summaries or dashboards.



---



\## How to run



1\. Install:

```bash

pip install -r requirements.txt



