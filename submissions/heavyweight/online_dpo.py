import random
import torch
import sys
import os

from trl.trainer.online_dpo_config import OnlineDPOConfig
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer
from trl.trainer.judges import BasePairwiseJudge
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightweight.recipe_generator import print_recipe

# Defining parameters / training variables
TRAINING_CONFIG = OnlineDPOConfig(output_dir=None)
MODEL_NAME = "Ashikan/dut-recipe-generator"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE)

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/common_ingredients.csv"), "r", encoding="utf-8") as f:
    COMMON_INGREDIENTS = [line.strip() for line in f if line.strip()]

# Custom judge class to allow human to rank outputs
class HumanJudge(BasePairwiseJudge):
    def __init__(self):
        super().__init__()

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        """
        Args:
            prompt (str): The input prompt.
            completions (List[str]): Candidate completions from the model.
        Returns:
            List[dict] | None: [{"chosen": str, "rejected": str}] or None
        """
        if len(completions) < 2:
            sys.exit()

        response_a, response_b = completions[0], completions[1]
        preference_data = self.get_recipe_preference(prompts, response_a, response_b)
        return preference_data


    def get_recipe_preference(self, prompt, recipe_a, recipe_b) -> list[int]:
        """
        Presents two recipe responses for a given prompt and asks the user to choose which is better.
        Displays both recipes, handles invalid recipes, and prompts for user input.
        
        Args:
            prompt (str): The prompt for which recipes are generated.
            recipe_a (dict or str): The first recipe response.
            recipe_b (dict or str): The second recipe response.
        
        Returns:
            dict: Dictionary with keys 'chosen' and 'rejected' containing the selected and non-selected recipes.
            None: If the feedback loop is terminated by KeyboardInterrupt.
        """
        print(f"Prompt: {prompt}")

        print(f"\nResponse A:\n")
        try:
            print_recipe(recipe_a)
        except Exception:
            print("Error: Invalid Recipe")

        print(f"\nResponse B:\n")
        try:
            print_recipe(recipe_b)
        except Exception:
            print("Error: Invalid Recipe")
        
        while True:
            try:
                choice = input("Which response is better? (A or B): ").strip().upper()
                if choice == "A":
                    return [1, 0]
                elif choice == "B":
                    return [0, 1]
                else:
                    print("Invalid input. Please enter 'A' or 'B'.")
            except KeyboardInterrupt:
                print("\nFeedback loop terminated.")
                sys.exit()


JUDGE = HumanJudge()

if __name__ == "__main__":

    while True:

        _ = input("Welcome! Type E to exit, or anything else to continue with training: ")

        if _.strip().upper() == "E":
            break

        PROMPTS: list[dict] = []

        for _ in range(10):
            prompt = random.sample(COMMON_INGREDIENTS, 5)
            PROMPTS.append({"prompt": prompt, "role": "user"})

        DATASET = Dataset.from_list(PROMPTS)

        print(DATASET)

        TRAINER = OnlineDPOTrainer(
            model=MODEL, 
            judge=JUDGE, args=TRAINING_CONFIG, processing_class=TOKENIZER, train_dataset=DATASET
        )

        TRAINER.train()