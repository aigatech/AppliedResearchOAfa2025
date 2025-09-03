import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load Models
def load_answer_model():
    return pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)

answer_model = load_answer_model()
ner_model = load_ner_model()

# Functions
def get_answer(paragraph, question):
    if not answer_model:
        return "Answer model not loaded.", ""
    if len(paragraph.strip()) == 0 or len(question.strip()) == 0:
        return "Please provide both paragraph and question.", ""
    result = answer_model(question=question, context=paragraph)
    answer = result['answer']
    score = round(result['score'], 4)
    return answer, score

def run_ner(paragraph):
    if not ner_model:
        return "NER model not loaded.", "NER model not loaded."
    if len(paragraph.strip()) == 0:
        return "Please provide a paragraph.", "Please provide a paragraph."
    result = ner_model(paragraph)
    entities = process_ner_results(result)
    entities_str = "### Keywords:\n"+format_entities(entities)
    highlighted_md = highlight_entities(paragraph, result)
    return entities_str, highlighted_md

def process_ner_results(results):
    people = []
    location = []
    organization = []
    misc = []

    for entity in results:
        entity['word'] = entity['word'].replace("##", "")
        match entity['entity']:
            case 'B-PER':
                people.append(entity['word'])
            case 'I-PER':
                people[-1] += " " + entity['word']
            case 'B-LOC':
                location.append(entity['word'])
            case 'I-LOC':
                location[-1] += " " + entity['word']
            case 'B-ORG':
                organization.append(entity['word'])
            case 'I-ORG':
                organization[-1] += " " + entity['word']
            case 'B-MISC':
                misc.append(entity['word'])
            case 'I-MISC':
                misc[-1] += " " + entity['word']
    
    return list(set(people)), list(set(location)), list(set(organization)), list(set(misc))

def format_entities(entities):
    people, location, organization, misc = entities
    result = []
    if people:
        result.append("**Person**")
        for i, person in enumerate(people, start=1):
            result.append(f"{i}. {person}")
    if location:
        result.append("\n**Location**")
        for i, loc in enumerate(location, start=1):
            result.append(f"{i}. {loc}")
    if organization:
        result.append("\n**Organization**")
        for i, org in enumerate(organization, start=1):
            result.append(f"{i}. {org}")
    if misc:
        result.append("\n**Misc**")
        for i, item in enumerate(misc, start=1):
            result.append(f"{i}. {item}")
    return "\n".join(result)

def highlight_entities(paragraph, entities):
    highlighted = paragraph
    offset = 0

    # Merge B- and I- entities together
    merged_entities = []
    for entity in entities:
        if merged_entities and entity['entity'].startswith('I-') and merged_entities[-1]['entity'][2:] == entity['entity'][2:]:
            merged_entities[-1]['end'] = entity['end']
            merged_entities[-1]['word'] += entity['word']
        else:
            merged_entities.append(entity)
    entities = merged_entities

    for entity in entities:
        start = entity['start'] + offset
        end = entity['end'] + offset
        tag = entity['entity'].split('-')[-1]
        open_tag = f"<span style='background-color: yellow;' title='{tag}'>"
        close_tag = "</span>"
        highlighted = highlighted[:start] + open_tag + highlighted[start:end] + close_tag + highlighted[end:]
        offset += len(open_tag) + len(close_tag)
    return highlighted

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📖 History Helper")

    with gr.Row():
        paragraph = gr.Textbox(label="Context", lines=10, placeholder="Enter your context here...", scale=5)
        with gr.Column(scale=5):
            with gr.Row(equal_height=True):
                question = gr.Textbox(label="Question", placeholder="Enter your question here...", scale=2)
                question_btn = gr.Button("Submit", scale=0)
            with gr.Row():
                answer_output = gr.Textbox(label="Answer", scale=7)
                answer_score = gr.Textbox(label="Score", scale=3)
            
    with gr.Row():
        ner_md = gr.Markdown("Highlighted keywords will appear here.")
        ner_output = gr.Markdown("Keywords will appear here.")

    question_btn.click(get_answer, [paragraph, question], [answer_output, answer_score])
    question_btn.click(run_ner, paragraph, [ner_output, ner_md])

demo.launch(server_port = 8080)
