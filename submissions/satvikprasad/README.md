## Lyrical Emotion Classifier
This project is a simple emotions classifier that accepts a fuzzy text input representing a song's title, and spits out the strongest emotion triggered by the song based on solely the lyrics.

### How to Run the Lyrical Emotion Classifier
1. Ensure your current working directory is set to the project's subfolder.

2. (Optional) Create a virtual environment using 
```
python3 -m venv .venv
```
and activate it using the appropriate script on your system.

3. Install all required dependencies by running 
```
pip install -r requirements.txt
```

4. Then, simply run 
```
export TOKENIZERS_PARALLELISM=false && python main.py
```
and visit `http://127.0.0.1:5000` on your device.

