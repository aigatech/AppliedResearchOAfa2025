# Investor Due-Diligence Helper

This project is my submission for **AI@GT Applied Research Fall 2025**.  
It is a Gradio app that helps investors quickly analyze short startup descriptions.  
The app evaluates a startup blurb (e.g. memo, pitch slide, or homepage copy) and provides:

- **Quick Verdict** — Promising  / Mixed  / Risky  
- **Opportunities** — labeled with  High /  Medium /  Low  
- **Risks** — labeled with  Critical / ⚠ Moderate /  Low  
- **Entities** — organizations, people, and money terms detected in the text  
- **Tone** — sentiment of the description  
- **Auto-Generated Questions** — diligence prompts based on top risks  

---
## How to Run

```bash
# Navigate into your folder
cd submissions/Miguel Tjia

# Install dependencies
pip install -r requirements.txt

# Start the app
python app.py

# Then open your browser at:
# http://localhost:7860


## Example Input:
We build a billing API for Southeast Asia SMBs. 
$120k ARR, 35% MoM growth, pilot with 3 banks, SOC2 in progress, 8-person team.



