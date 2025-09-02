# AI Indian Recipe Concept Generator
This project uses a fine-tuned Hugging Face distil-gpt2 model to generate new and creative Indian food recipe concepts. By training on the "Indian Food 101" dataset from Kaggle, the AI learns the relationships between ingredients, flavors, regions, and dish names. It can then be prompted to invent novel recipe ideas that don't exist in the original dataset.

This model is designed to be a tool for culinary brainstorming, capable of creating plausible ingredient lists and metadata for dishes that a chef could then develop.

## Features
Data Preprocessing: A script to clean and format the raw dataset for training.

CPU-Friendly Training: Uses distil-gpt2, a smaller model that can be fine-tuned on a standard CPU.

Creative Recipe Generation: The make_recipe.py script allows you to write a "creative brief" for a new dish and generates a concept based on it.

Saves Output: Automatically saves the generated recipes to an output directory.

## Prerequisites
Before you begin, ensure you have the following:

Python 3.8 or higher.

The "Indian Food 101" dataset from Kaggle. Download the indian_food_101.csv file and place it in the root of your project directory. You can find it here: Kaggle Dataset Link.

The run_clm.py script from the Hugging Face Transformers examples repository. You can download it directly from this link and save it in your project directory.
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

## Important: Remember to install all dependencies for run_clm.py into your venv for the project
These include:
```
"
#     transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
```

#### For Transformers:
`pip install git+https://github.com/huggingface/transformers`
#### For the others:


## Setup & Installation
Create a virtual environment:

`python3 -m venv venv`
`source venv/bin/activate`  
### On Windows, use: venv\Scripts\activate

## Install the required packages:

`pip install transformers torch pandas datasets`

After these steps, your project directory should look like this:
```
/your-project-folder
|-- ...\indian_food_101.csv         <-- Your downloaded dataset
|-- \submissions\preprocess.py               <-- The preprocessing script
|-- \submissions\make_recipe.py              <-- The generation script
|-- \submissions\run_clm.py                  <-- The Hugging Face training script
|-- ...\venv/
```

How to Run the Project
Follow these steps in order to generate your first recipe concepts.

### Step 1: Preprocess the Data
First, you need to convert the raw .csv file into a structured text file that the model can learn from.

Run the preprocess.py script from your terminal:

`python preprocess.py`

This will read indian_food.csv, clean the data, and create a new file named indian_recipe_concepts.txt.

### Step 2: Train the AI Model
Now, use the run_clm.py script to fine-tune the distil-gpt2 model on your processed data. This is the most time-consuming step and may take a while depending on your CPU.

Run the following command in your terminal:
```
python run_clm.py \
    --model_name_or_path abhinema/distillgpt2 \
    --train_file submissions/indian_recipe_concepts.txt \
    --do_train \
    --output_dir fine_tuned_distillgpt2-concepts \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --num_train_epochs 5 \
    --block_size 256 
```

This command will create a new folder named `fine_tuned_distillgpt2-concepts` containing your custom-trained model files.

### Step 3: Generate New Recipes
With your model trained, you can now generate new recipe concepts!

Run the make_recipe.py script:

`python make_recipe.py`

This script will:

1. Load your fine-tuned model from the fine-tuned-distilgpt2-concepts folder.

2. Use the prompt inside the script to generate new recipe ideas.

3. Create a folder named recipe_outputs.

4. Save the results in a file named generated_recipes.txt inside that folder.

Customization
To generate different kinds of recipes, simply open the make_recipe.py file and edit the prompt variable. For example, to invent a spicy vegetarian snack from Gujarat, you could change the prompt to:
```
prompt = (
    "[START]\n"
    "name: Almond Idli\n"
    "diet: vegetarian\n"
    "flavor_profile: sweet\n"
    "course: dessert\n"
    "state: Gujarat\n"
    "region: West"
)
```

The model will then generate the rest of the concept for you. Experiment with different combinations to see what creative ideas the AI can come up with!