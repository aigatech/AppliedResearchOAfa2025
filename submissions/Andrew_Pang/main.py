
from utils import clean_text, read_txt

import torch
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer, util
import warnings



# Suppress transformers logs
hf_logging.set_verbosity_error()   # options: debug, info, warning, error, critical
warnings.filterwarnings("ignore")



def load_semantic_model():

    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    return semantic_model



def get_semantic_similarity(semantic_model, queries, corpus):

    query_embedding = semantic_model.encode(queries, convert_to_tensor=True)
    corpus_embeddings = semantic_model.encode(corpus, convert_to_tensor=True)

    return util.cos_sim(query_embedding, corpus_embeddings)



def load_qa_model():

    qa_model_name = "deepset/roberta-base-squad2"  # or your downloaded model path
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

    return {'tokenizer': qa_tokenizer, 'model': qa_model}



def answer_question(qa_model, question: str, context: str) -> tuple[str, float]:

    model = qa_model['model']
    tokenizer = qa_model['tokenizer']

    # Tokenize inputs
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Apply softmax to get probabilities
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)

    # Get most probable start and end positions
    start_idx = torch.argmax(start_probs)
    end_idx = torch.argmax(end_probs) + 1  # Include end token

    # Get confidence score as product of start and end probabilities
    confidence = (start_probs[0][start_idx] * end_probs[0][end_idx - 1]).item()

    # Convert IDs to string
    answer_ids = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_ids))

    return answer.strip(), confidence



def clean_entities(raw_entities):

    def normalize(entity):
        return entity.strip().lower()
    
    clean_set = {normalize(ent['word']) for ent in raw_entities if ent['word'].strip() and ent['score'] > 0.8 and ent['entity_group'] in ['ORG', 'MISC']}
    clean_set = {e for e in  clean_set if len(e) > 1 and "##" not in e}

    return clean_set



def match_entities(entities1, entities2):

    def normalize(entity):
        return entity.strip().lower()

    # Clean and normalize entity lists
    set1 = {normalize(ent['word']) for ent in entities1 if ent['word'].strip() and ent['score'] > 0.8}
    set2 = {normalize(ent['word']) for ent in entities2 if ent['word'].strip() and ent['score'] > 0.8}

    # Remove single-character or incomplete fragments
    set1 = {e for e in set1 if len(e) > 1 and "##" not in e}
    set2 = {e for e in set2 if len(e) > 1 and "##" not in e}

    # Compute overlaps and differences
    common = set1.intersection(set2)
    missing_from_1 = set2 - set1
    missing_from_2 = set1 - set2

    return {
        "common": list(common),
        "missing_from_first": list(missing_from_1),
        "missing_from_second": list(missing_from_2),
    }



def semantic_match(key_entities, search_entities, semantic_model):

    key_dict = {i: [] for i in key_entities}
    search_dict = {i: [] for i in search_entities}

    for key_entity in key_entities:
        for search_entity in search_entities:
            similarity = get_semantic_similarity(semantic_model, key_entity, search_entity)
            sim_val = similarity.item()
            
            key_dict[key_entity].append(sim_val)
            search_dict[search_entity].append(sim_val)

    # Average similarities per entity
    key_dict = {k: max(v) if v else 0 for k, v in key_dict.items()}
    search_dict = {k: max(v) if v else 0 for k, v in search_dict.items()}

    # Rank (sort) by values, highest first
    key_ranked = dict(sorted(key_dict.items(), key=lambda x: x[1], reverse=True))
    search_ranked = dict(sorted(search_dict.items(), key=lambda x: x[1], reverse=True))

    return key_ranked, search_ranked



def retrieve_qa_context(semantic_model, tokenizer, prompt, context, max_length):

    # Clean and split context into sentences
    sentences = clean_text(context)
    sentence_scores = {}

    # Compute similarity for each sentence
    for sentence in sentences:
        similarity = get_semantic_similarity(semantic_model, prompt, sentence)
        sim_val = similarity.item() if hasattr(similarity, "item") else float(similarity)
        sentence_scores[sentence] = sim_val

    # Sort sentences by similarity (descending order)
    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    extracted_context = ""
    total_length = 0

    # Iteratively add sentences until max_length is exceeded
    for sentence, score in ranked_sentences:
        sent_len = len(tokenizer(sentence, return_tensors="pt")['input_ids'][0])

        if total_length + sent_len > max_length:
            break  # stop once we exceed the max token length

        extracted_context += sentence.strip() + " "
        total_length += sent_len

    return extracted_context.strip()



def resume_job_evaluator(resume_doc, job_doc):

    semantic_model = load_semantic_model()
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

    similarity = get_semantic_similarity(semantic_model, resume_doc, job_doc)
    resume_entities = clean_entities(ner_pipeline(resume_doc))
    job_entities = clean_entities(ner_pipeline(job_doc))

    key_dict, search_dict = semantic_match(job_entities, resume_entities, semantic_model)
    matching = [k for k,v in key_dict.items() if v > 0.5]
    missing = [k for k,v in key_dict.items() if v < 0.5]
    top_5 = list(search_dict.keys())[:5]

    print(f'relevance: {int(similarity.item()*100)}%')
    print(f'matching skills: {matching}')
    print(f'missing skills: {missing}')
    print(f'my top 5 skills: {top_5}')



def job_qa_agent(question, job_doc):

    semantic_model = load_semantic_model()
    qa_model = load_qa_model()
    tokenizer = qa_model['tokenizer']
    answer = answer_question(qa_model, question, retrieve_qa_context(semantic_model, tokenizer, question, job_doc, 350))

    print(f'question: {question}')
    if '<s>' not in answer:
        print(f'answer: {answer[0]}')
        print(f'confidence: {answer[1]:.2f}')
    else:
        print('answer: unable to answer')



if __name__ == '__main__':

    resume_path = 'docs/resume.txt'
    job_path = 'docs/gen_ai_data_engineer.txt' # change to any .txt job description file
    question = 'What are the qualifications for this job?' # try asking different questions based on the job description

    resume_doc = read_txt(resume_path)
    job_doc = read_txt(job_path)
    resume_job_evaluator(resume_doc, job_doc)
    job_qa_agent(question, job_doc)