Project: WhyWrong (Akshaj Nadimpalli)
Generates deceptive, counterfactual flashcards from any article or local text. (currently changes a specific word of the phrase and provide 3 choices, of which one is true).

0) Setup
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

1) Build tiny dataset
python make_synth_data.py --sources sample_sources.txt \
  --out data/train.jsonl \
  --k 30 --per_sentence_max 5 --max_ranked 500 --target_total 1000

2) Train LoRA
python train_lora.py \
  --train_path data/train.jsonl \
  --output_dir adapters/whywrong-lora \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

3) Run generator on new article (replace link with any wikipedia article)
Teacher (no training, just do this directly after the setup): python app.py --source "https://en.wikipedia.org/wiki/Psychology" \   
  --k 12 --mode teacher --outdir outputs
Student (with trained model): python app.py --source "https://en.wikipedia.org/wiki/Psychology" --k 10 --mode student --adapters adapters/whywrong-lora --outdir outputs
