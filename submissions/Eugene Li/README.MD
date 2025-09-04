# Eugene Li AI Submission: Georgia Tech Mascot Detecter (sort-of...)

## What this project is:
This AI model was trained on 1000 images of 90 different animals. It can guess an input image with roughly -90% accuracy, and if it guesses that the image is the Georgia Tech mascot (ik it's a yellow jacket and not a bee but close enough :c), you'll get a surprise!

## How to run it:
- Enter this command into python terminal: "pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn"
- Run ModelTrainer.py
- Pick an image you want the model to guess the animal of and put it in the same directory as everything else
- Run ModelExplorer and change the file locations

- To train the model on a different dataset, just change the name of the HuggingFace dataset
