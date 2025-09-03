import random
from dataclasses import dataclass

import numpy as np

from embedding import embed, centroid, cosine
from colors import bold, green, red, yellow, cyan, magenta

WRAP = 88
def hr(): print("-" * WRAP)

@dataclass
class Phase:
    name: str
    candidates: dict[str, list[str]]
    answer: str

def play_phase(phase: Phase, nouns_pool: list[str], num_noun_sample: int, turns: int = 3) -> bool:
    """
    Manages the logic for a single phase of the game (Culprit, Place, or Weapon).
    Return: ok (is_correct)
    """
    hr()
    print(bold(cyan(f"{phase.name.upper()} PHASE")))
    hr()
    keys = list(phase.candidates.keys())
    print(bold(yellow("Candidates:")), ", ".join(keys))

    # sample nouns
    nouns = random.sample(nouns_pool, k=num_noun_sample)
    print(bold("\nPick a clue by choosing one noun from the list below."))
    print("Example: 'book' or type an index like '1'.")
    print(bold(yellow("\nNouns:")))
    for i, n in enumerate(nouns, 1):
        print(f"  {i}. {n}")
    print(f"\nYou have {bold(str(turns))} clue questions before the final guess.")

    # precompute centroids
    cents: dict[str, np.ndarray] = {k: centroid(v) for k, v in phase.candidates.items()}

    asked = 0
    while asked < turns:
        q = input(bold("> ")).strip().lower()
        if not q:
            continue
        if q.isdigit():
            i = int(q) - 1
            if not (0 <= i < len(nouns)):
                print(red("Pick a valid noun index."))
                continue
            clue = nouns[i]
        else:
            if q not in nouns:
                print(red("Pick a valid noun from the list."))
                continue
            clue = q

        # embed clue and score against all candidates
        qv = embed([clue])[0]
        sims = []
        for name, c in cents.items():
            s = cosine(qv, c)
            sims.append((name, s))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)

        # Find rank of the true answer
        rank = -1
        for i, (name, _) in enumerate(sims_sorted):
            if name == phase.answer:
                rank = i + 1
                break

        asked += 1

        # feedback
        singular_name = phase.name.lower()[:-1]
        print(f"Clue '{yellow(clue)}':")
        rank_str = bold(f"Rank {rank} / {len(sims_sorted)}")
        print(f"  True {singular_name}'s position: {green(rank_str) if rank == 1 else red(rank_str)}")
        if rank > 1:
            top1, _ = sims_sorted[0]
            print(f"  Closest {singular_name}: {magenta(top1)}")

        remaining = turns - asked
        if remaining:
            print(f"{bold(str(remaining))} clue question(s) left.")
        else:
            print(red("No clues left."))
        print()

    # final guess
    while True:
        guess = input(bold("Your final guess: ")).strip().title()
        if guess not in phase.candidates:
            print(red("Pick a valid candidate name."))
            continue
        ok = (guess == phase.answer)
        if ok:
            print(green("✅ Correct!"))
        else:
            print(red(f"❌ Not {guess}."))
        print("Answer:", bold(green(phase.answer)))
        return ok
