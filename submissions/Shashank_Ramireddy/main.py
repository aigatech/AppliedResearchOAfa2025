import argparse
from haiku import load_model, generate_haiku

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--topic', required=True, help='topic for the haiku')
    # default to FLAN-T5 for better coherence
    p.add_argument('--model', default='google/flan-t5-small',
                   help='HF model id (e.g., google/flan-t5-small, google/flan-t5-base, distilgpt2)')
    return p.parse_args()

def main():
    args = parse_args()
    print("[main] loading model ...")
    tok, mdl, model_type = load_model(args.model)
    print("[main] generating haiku ...")
    poem = generate_haiku(args.topic, tok, mdl, model_type)
    print("\n" + poem + "\n")

if __name__ == '__main__':
    main()