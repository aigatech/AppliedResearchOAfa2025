from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Initialize the model
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading sentence transformer: {e}")
    model = None

def pick_best_headline(headlines, goal="increase signups for a local service"):
    """Pick the best headline based on semantic similarity to goal."""
    if not model or not headlines:
        return headlines[0] if headlines else ""
    
    try:
        # Encode headlines and goal
        headline_embeddings = model.encode(headlines, convert_to_tensor=True)
        goal_embedding = model.encode([goal], convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(goal_embedding, headline_embeddings)[0]
        
        # Return the headline with highest similarity
        best_index = torch.argmax(similarities).item()
        return headlines[best_index]
        
    except Exception as e:
        print(f"Error in pick_best_headline: {e}")
        return headlines[0] if headlines else ""

def score_variants(variants, goal):
    """Score variants and return sorted list of (variant, score) tuples."""
    if not model or not variants:
        return [(v, 0.0) for v in variants]
    
    try:
        # Encode variants and goal
        variant_embeddings = model.encode(variants, convert_to_tensor=True)
        goal_embedding = model.encode([goal], convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(goal_embedding, variant_embeddings)[0]
        
        # Return sorted list of (variant, score) tuples
        scored_variants = [(variants[i], similarities[i].item()) for i in range(len(variants))]
        return sorted(scored_variants, key=lambda x: x[1], reverse=True)
        
    except Exception as e:
        print(f"Error in score_variants: {e}")
        return [(v, 0.0) for v in variants]

def pick_best_cta(ctas, goal="drive action"):
    """Pick the best call-to-action based on goal."""
    return pick_best_headline(ctas, goal)

def rank_copy_elements(copy_dict, goal="increase engagement"):
    """Rank all copy elements and return the best ones."""
    if not copy_dict:
        return copy_dict
    
    result = copy_dict.copy()
    
    # Rank headlines if multiple
    if isinstance(copy_dict.get("headline"), list):
        result["headline"] = pick_best_headline(copy_dict["headline"], goal)
    
    # Rank CTAs if multiple
    if isinstance(copy_dict.get("ctas"), list) and len(copy_dict["ctas"]) > 1:
        result["ctas"] = [pick_best_cta(copy_dict["ctas"], goal)]
    
    return result
