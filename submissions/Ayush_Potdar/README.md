# Clash Royale Counter Deck Creator

---

## What it does
This is a small model that learns card embeddings from past matches dating from 2020 to 2021 and uses that information to generate 3 counter-decks to a deck the user inputs along with a learned probability that each deck will beat the inputted deck. Training it, as described below, generates a candidates.csv file which is a pool of the most common decks with success - these are the decks measured up against the inputted deck come time of usage.

## How to use
1. Setup a clean python environment 
2. To your project folder add the following files: counter_deck.py, CardKey.csv (this contains the mapping key for the numerical ids attached to cards in the large dataset), and download the dataset titled Clash_Royale_Prediction by Grandediw from Hugging face as a csv.
3. From the project root run the following command, replacing HUGGINGFACECSV with the actual name of the file you download from hugging face:

    ```bash
    python counter_deck.py --mode train \
  --data_files "./HUGGINGFACECSV" \
  --card_map "./CardKey.csv" \
  --out_dir out \
  --limit_rows 200000 \
  --max_cards 256 \
  --top_candidates 5000 \
  --batch_size 256 \
  --max_steps 600 \
  --d_model 16
4. Once all outputs are generated from the previous command and the model is trained, run the following command replacing the cards besides deck with the cards in the deck you want to counter:

    ```bash
    python counter_nn_tf.py --mode query \
  --model out/model.weights.h5 \
  --vocab out/id2idx.json \
  --candidates out/candidates.csv \
  --card_map "./HUGGINGFACECSV.csv" \
  --deck "Knight, Archers, Fireball, Cannon, Ice Spirit, Musketeer, The Log, Hog Rider" \
  --top 3
5. You'll see the target deck (the one you're trying to counter) and the top 3 counter decks with the probability of those decks winning, ordered from best to worst in your command line if everything was successful.

#### Note: I included the CardKey.csv in my commit since that information wasn't readily available on hugging face and required some digging on the internet. Luckily, the 26 million to 28 million id system in the hugging face dataset is a standardized classified for clash royale cards.