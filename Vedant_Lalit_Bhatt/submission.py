import gradio as gr
from transformers import pipeline
import wikipediaapi
import spacy
import torch


nlp = spacy.load("en_core_web_md")



wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="VedantBhattFactChecker/1.0 (https://github.com/vedantlbhatt)"
)

def extract_entity_wikidata(claim):
    doc = nlp(claim)
    named_entities = [ent.text for ent in doc.ents]
    claim_lower = claim.lower()
    

    if any(phrase in claim_lower for phrase in ['lived in', 'existed in', 'lived during', 'existed during']): # VERY common facts faced
        words = claim.split()
        for i, word in enumerate(words):
            if word.lower() in ['lived', 'existed'] and i > 0:
                subject = ' '.join(words[:i])
                
                for ent in named_entities:
                    if subject.lower() == ent.lower():
                        return ent

                #using noun chunks
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                for chunk in noun_chunks:
                    if subject.lower() in chunk.lower() or chunk.lower() in subject.lower():
                        return chunk.title()
                
                #use this returned subject for search
                return subject.title()
    
#common case of location
    if ' is in ' in claim_lower or ' is located in ' in claim_lower:
        words = claim.split()
        for i, word in enumerate(words):
            if word.lower() in ['is', 'are'] and i > 0:
                subject = ' '.join(words[:i])
                for ent in named_entities:
                    if subject.lower() == ent.lower():
                        return ent
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                for chunk in noun_chunks:
                    if subject.lower() in chunk.lower() or chunk.lower() in subject.lower():
                        return chunk.title()
                capitalized_subject = subject.title()
                for ent in named_entities:
                    if capitalized_subject.lower() in ent.lower() or ent.lower() in capitalized_subject.lower():
                        return ent
                return capitalized_subject
    
    # common case of "in"
    if ' is in the ' in claim_lower:
        words = claim.split()
        for i, word in enumerate(words):
            if word.lower() == 'is' and i > 0:
                subject = ' '.join(words[:i])
                for ent in named_entities:
                    if subject.lower() == ent.lower():
                        return ent
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                for chunk in noun_chunks:
                    if subject.lower() in chunk.lower() or chunk.lower() in subject.lower():
                        return chunk.title()
                return subject.title()
    
    #resort to original logic of finding subjects
    if named_entities:
        return sorted(named_entities, key=len, reverse=True)[0]
    
    #use noun chunks if else
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    if noun_chunks:
        main_nouns = []
        for chunk in noun_chunks:
            parts = chunk.split()
            if parts:
                main_nouns.append(parts[0])
        if main_nouns:
            return main_nouns[0].title()
        return noun_chunks[0].title()
    
    #last resort: just use the claim (unlikely :(   )
    return claim.title()

# Use GPU if available for faster inference
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

def get_wikipedia_summary(query):

    page = wiki.page(query)
    if page.exists():
        if "most commonly refers to" in page.summary or "may refer to" in page.summary:
            #differentiation between us state/countries abroad
            if query.lower() in ['georgia', 'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming']:
                page = wiki.page(f"{query} (U.S. state)")
                if page.exists():
                    return page.summary
        else:
            return page.summary
    

    if query.lower() in ['georgia', 'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming']:
        # Try with "U.S. state" suffix
        page = wiki.page(f"{query} (U.S. state)")
        if page.exists():
            return page.summary
    
    #use common variations (variations with capitilization to find correct page)
    page = wiki.page(query.capitalize())
    if page.exists():
        return page.summary
    page = wiki.page(query.lower())
    if page.exists():
        return page.summary
    
    return "No Wikipedia page found." # :(

#actual fact checking
def fact_check(claim):
    entity = extract_entity_wikidata(claim)
    context = get_wikipedia_summary(entity)
    context = context[:1000]
    
    #see if claim key words appear in the wiki
    claim_lower = claim.lower()
    context_lower = context.lower()
    
    words = claim_lower.split()
    key_terms = []

    #remove these redundant, meaningless words (will mess up the checking) :(
    common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'lived', 'existed', 'lives', 'exists'}
    for word in words:
        if word not in common_words and len(word) > 2:
            key_terms.append(word)
    
    found_terms = 0
    for term in key_terms:
        if term in context_lower:
            found_terms += 1
    
    if len(key_terms) > 0:
        overlap_ratio = found_terms / len(key_terms)
        
        if overlap_ratio >= 0.8:  
            return {'true': 0.85, 'uncertain': 0.10, 'false': 0.05}, context
        elif overlap_ratio >= 0.6:  
            return {'true': 0.70, 'uncertain': 0.20, 'false': 0.10}, context
        elif overlap_ratio >= 0.4:  
            return {'true': 0.50, 'uncertain': 0.35, 'false': 0.15}, context
        else: 
            return {'true': 0.20, 'uncertain': 0.60, 'false': 0.20}, context
    

    candidate_labels = ['true', 'false', 'uncertain']
    result = classifier(
        context,
        candidate_labels,
        hypothesis_template=f'The statement \'{claim}\' is {{}}.'
    )
    
    scores = {result['labels'][i]: round(result['scores'][i], 3) for i in range(len(result['labels']))}
    return scores, context

#simple ui
iface = gr.Interface(
    fn=fact_check,
    inputs=gr.Textbox(lines=2, placeholder="Enter a claim (e.g. The Eiffel Tower is in Paris)", label="Claim"),
    outputs=[
        gr.Label(num_top_classes=3, label="Fact Check Result"),
        gr.Textbox(lines=6, label="Wikipedia Evidence", interactive=False)
    ],
    title="AI Fact Checker with Wikipedia",
    description="Enter a claim. The app fetches evidence from Wikipedia and checks if the claim is true, false, or uncertain."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True, share=True)
