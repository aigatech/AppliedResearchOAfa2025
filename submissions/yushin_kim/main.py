import random
import sys
import textwrap
import json
from game import Phase, play_phase, hr
from colors import bold, green, red, yellow, blue, magenta, cyan

WRAP = 88

def wrap(s):
    return textwrap.fill(s, width=WRAP)

def main():
    """
    Main function to run the Word Clue game.
    """
    print(bold(cyan("===== WORD CLUE =====")))

    # Load game data from JSON file
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(red("Error: data.json not found. Please make sure the data file is in the same directory."))
        sys.exit(1)
    except json.JSONDecodeError:
        print(red("Error: Could not decode data.json. Please check for syntax errors in the file."))
        sys.exit(1)

    CULPRITS = data["CULPRITS"]
    PLACES = data["PLACES"]
    WEAPONS = data["WEAPONS"]
    NOUNS = data["NOUNS"]
    NUM_CULPRIT_SAMPLE = data["NUM_CULPRIT_SAMPLE"]
    NUM_PLACE_SAMPLE = data["NUM_PLACE_SAMPLE"]
    NUM_WEAPON_SAMPLE = data["NUM_WEAPON_SAMPLE"]
    NUM_NOUN_SAMPLE = data["NUM_NOUN_SAMPLE"]

    while True:
        print()
        print(wrap("Guess the culprit, the place, and the weapon in three phases. "
                   "Each phase gives you 3 clues: choose one noun from a list. "
                   "Feedback shows the rank of the true answer among the candidates for your chosen clue."))
        print()

        # Prepare data for each phase
        culprit_data = dict(random.sample(list(CULPRITS.items()), k=NUM_CULPRIT_SAMPLE))
        place_data = dict(random.sample(list(PLACES.items()), k=NUM_PLACE_SAMPLE))
        weapon_data = dict(random.sample(list(WEAPONS.items()), k=NUM_WEAPON_SAMPLE))

        phase_list = [
            ("Culprits", culprit_data),
            ("Places", place_data),
            ("Weapons", weapon_data),
        ]

        phases = [
            Phase(name, p_data, random.choice(list(p_data.keys())))
            for name, p_data in phase_list
        ]

        game_won = True
        for phase in phases:
            ok = play_phase(phase, nouns_pool=NOUNS, num_noun_sample=NUM_NOUN_SAMPLE)
            if not ok:
                game_won = False
                break
        
        if game_won:
            hr()
            print(bold(green("🎉 Congratulations! You've solved the mystery! 🎉")))

        hr()
        print(bold(yellow("ANSWER:")))
        for phase in phases:
            print(f"  - {phase.name[:-1]}: {bold(phase.answer)}")
        hr()
        print(cyan("Thanks for playing!"))

        hr()
        again = input(bold("Play again? (y/n) ")).strip().lower()
        if again not in ["y", "yes"]:
            break
        print("\n" * 2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as e:
        print(red(f"\nAn unexpected error occurred: {e}"))
        sys.exit(1)
