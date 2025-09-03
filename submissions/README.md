# Food Prediction + Recipe Generator

# What it does

Upload a photo of food and the app will classify what dish it is and show its confidence. Then it will generate a recipe, outputting a title, ingredients and steps to make the dish.

# How to run it (python 3+)

1. Clone the repository & enter the project

```bash
git clone https://github.com/ani369chad/AppliedResearchOAfa2025.git
```
Then open the cloned repository folder in VS Code

```bash
cd submissions
```

2. Create & activiate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```
(Windows) .venv\Scripts\activate

3. Install dependecies

```bash
pip3 install -r requirements.txt
```

4. Run the app

```bash
python3 app.py
```
The first run will take a minute to load as it is downloading model weights. When the terminal shows something like: shows something like:
Running on local URL: http://127.0.0.1:7860, open the url and upload your image and click submit. If you want to use a different image, click clear. 