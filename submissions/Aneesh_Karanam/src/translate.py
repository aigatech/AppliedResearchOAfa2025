import argparse
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import emoji as emoji_lib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
from src.lexicon import load_lexicon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

semantic_model = None

def load_semantic_model():
    """Load sentence transformer model for semantic similarity"""
    global semantic_model
    if semantic_model is None:
        try:
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic similarity model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            semantic_model = None
    return semantic_model

def enhanced_tokenize(text: str) -> List[str]:
    """Enhanced tokenization with better handling of contractions and phrases"""
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove punctuation but preserve word boundaries
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split and clean
    tokens = [t.lower().strip() for t in text.split() if t.strip()]
    return tokens

def extract_emojis(s: str) -> List[str]:
    """Extract emoji characters from string"""
    return [e['emoji'] for e in emoji_lib.emoji_list(s)]

def semantic_lexicon_match(word: str, lexicon: dict, threshold: float = 0.7) -> Optional[str]:
    """Find semantically similar words in lexicon using sentence transformers"""
    model = load_semantic_model()
    if model is None or not lexicon:
        return None
    
    try:
        word_embedding = model.encode([word])
        
        lexicon_words = list(lexicon.keys())
        lexicon_embeddings = model.encode(lexicon_words)
        
        similarities = np.dot(word_embedding, lexicon_embeddings.T).flatten()
        
        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= threshold:
            best_word = lexicon_words[best_idx]
            logger.debug(f"Semantic match: '{word}' -> '{best_word}' (score: {similarities[best_idx]:.3f})")
            return lexicon[best_word]
    
    except Exception as e:
        logger.debug(f"Semantic matching failed for '{word}': {e}")
    
    return None

def improved_generator_emojis(model, tokenizer, text: str, max_emojis: int = 8, temperature: float = 0.7) -> List[str]:
    """Improved emoji generation with better prompting and sampling"""
    
    # Enhanced prompt with few-shot examples
    examples = [
        "Text: I love pizza -> 😍🍕",
        "Text: The cat is sleeping -> 🐱😴",
        "Text: Happy birthday celebration -> 🎉🎂🎈"
    ]
    
    prompt = (
        "Translate text to emojis that capture emotions, actions, and key objects.\n\n"
        + "\n".join(examples) + "\n"
        f"Text: {text} -> "
    )
    
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=temperature,
                do_sample=temperature > 0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after the arrow)
        if "->" in generated_text:
            emoji_part = generated_text.split("->")[-1].strip()
        else:
            emoji_part = generated_text[len(prompt):].strip()
        
        emojis = extract_emojis(emoji_part)
        
        # Fallback: if no emojis generated, try simpler approach
        if not emojis:
            simple_prompt = f"Convert to emojis: {text}"
            simple_inputs = tokenizer(simple_prompt, return_tensors='pt', truncation=True, max_length=256)
            simple_inputs = {k: v.to(device) for k, v in simple_inputs.items()}
            
            with torch.no_grad():
                simple_outputs = model.generate(
                    **simple_inputs,
                    max_new_tokens=20,
                    temperature=0.8,
                    do_sample=True
                )
            
            simple_text = tokenizer.decode(simple_outputs[0], skip_special_tokens=True)
            emojis = extract_emojis(simple_text)
        
        return emojis[:max_emojis]
        
    except Exception as e:
        logger.error(f"Error in emoji generation: {e}")
        return []

def calculate_translation_confidence(text: str, emojis: List[str], lexicon: dict) -> float:
    """Calculate confidence score for translation quality"""
    if not emojis or not text.strip():
        return 0.0
    
    tokens = enhanced_tokenize(text)
    if not tokens:
        return 0.0
    
    # Calculate lexicon coverage
    lexicon_matches = sum(1 for token in tokens if token in lexicon)
    lexicon_coverage = lexicon_matches / len(tokens)
    
    # Calculate emoji density (emojis per word ratio)
    emoji_density = len(emojis) / len(tokens)
    optimal_density = 0.5  # Assume 1 emoji per 2 words is optimal
    density_score = 1.0 - abs(emoji_density - optimal_density)
    density_score = max(0.0, density_score)
    
    # Combine scores
    confidence = (lexicon_coverage * 0.6) + (density_score * 0.4)
    return min(1.0, confidence)

def hybrid_translate(text: str, lexicon: dict, model=None, tokenizer=None, 
                    max_emojis: int = 12, temperature: float = 0.7) -> List[str]:
    """Enhanced hybrid translation with semantic matching and improved generation"""
    
    if not text.strip():
        return []
    
    tokens = enhanced_tokenize(text)
    lex_emojis = []
    unmapped_words = []
    
    for token in tokens:
        if token in lexicon:
            lex_emojis.append(lexicon[token])
        else:
            unmapped_words.append(token)
    
    semantic_emojis = []
    still_unmapped = []
    
    for word in unmapped_words:
        semantic_match = semantic_lexicon_match(word, lexicon)
        if semantic_match:
            semantic_emojis.append(semantic_match)
        else:
            still_unmapped.append(word)
    
    gen_emojis = []
    if still_unmapped and model is not None and tokenizer is not None:
        remaining_text = " ".join(still_unmapped)
        remaining_slots = max_emojis - len(lex_emojis) - len(semantic_emojis)
        
        if remaining_slots > 0:
            gen_emojis = improved_generator_emojis(
                model, tokenizer, remaining_text, 
                max_emojis=remaining_slots, temperature=temperature
            )
    
    all_emojis = lex_emojis + semantic_emojis + gen_emojis
    
    seen = set()
    unique_emojis = []
    for emoji in all_emojis:
        if emoji not in seen:
            seen.add(emoji)
            unique_emojis.append(emoji)
    
    final_emojis = unique_emojis[:max_emojis]
    
    logger.debug(f"Translation summary: {len(lex_emojis)} lexicon, {len(semantic_emojis)} semantic, {len(gen_emojis)} generated")
    
    return final_emojis

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Translate text to emojis using hybrid approach")
    parser.add_argument('--text', required=True, help='Text to translate')
    parser.add_argument('--max_emojis', type=int, default=12, help='Maximum number of emojis')
    parser.add_argument('--temperature', type=float, default=0.7, help='Creativity level (0.1-1.0)')
    parser.add_argument('--use_model', action='store_true', help='Use AI model for generation')
    parser.add_argument('--confidence', action='store_true', help='Show confidence score')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    lexicon = load_lexicon()
    
    model = tokenizer = None
    if args.use_model:
        print("Loading flan-t5-small model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Continuing with lexicon-only translation...")
    
    emojis = hybrid_translate(
        args.text, lexicon, 
        model=model, tokenizer=tokenizer, 
        max_emojis=args.max_emojis, temperature=args.temperature
    )
    
    result = " ".join(emojis) if emojis else "❌ No emojis generated"
    print(result)
    
    if args.confidence:
        conf_score = calculate_translation_confidence(args.text, emojis, lexicon)
        print(f"Confidence: {conf_score:.2f}")

if __name__ == "__main__":
    main()