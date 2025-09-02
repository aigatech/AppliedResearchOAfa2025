from transformers import pipeline
from collections import defaultdict
import gradio as gr

ABBREVIATIONS = {
    "pt": "patient",
    "c/o": "complains of",
    "hx": "history of",
    "r/o": "rule out"
}

def expand_abbreviations(text):
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    return text

def merge_subwords(entities):
    merged = []
    current = None
    for ent in entities:
        word = ent["word"].replace("##", "")
        if current and ent["entity_group"] == current["entity_group"]:
            current["word"] += word
        else:
            if current:
                merged.append(current)
            current = {"word": word, "entity_group": ent["entity_group"]}
    if current:
        merged.append(current)
    return merged

ENTITY_MAP = {
    "Sign_symptom": "Symptoms",
    "Disease_disorder": "Diseases",
    "Medication": "Medications",
    "Therapeutic_procedure": "Procedures",
    "Biological_structure": "Body_Structures"
}

def group_entities(entities):
    grouped = defaultdict(list)
    for ent in entities:
        category = ENTITY_MAP.get(ent["entity_group"], ent["entity_group"])
        grouped[category].append(ent["word"])
    return dict(grouped)

def generate_summary(standardized):
    parts = []
    if "Symptoms" in standardized:
        parts.append("Patient complains of " + ", ".join(standardized["Symptoms"]))
    if "Diseases" in standardized:
        parts.append("History of " + ", ".join(standardized["Diseases"]))
    if "Medications" in standardized:
        parts.append("Currently on " + ", ".join(standardized["Medications"]))
    if "Procedures" in standardized:
        parts.append("Planned procedures: " + ", ".join(standardized["Procedures"]))
    return "; ".join(parts) + "."

ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

def process_note_gui(note):
    note_lower = note.lower()
    note_expanded = expand_abbreviations(note_lower)
    entities = ner_pipeline(note_expanded)
    entities_merged = merge_subwords(entities)
    standardized = group_entities(entities_merged)
    summary = generate_summary(standardized)
    return standardized, summary

iface = gr.Interface(
    fn=process_note_gui,
    inputs=gr.Textbox(lines=10, placeholder="Paste doctor note here..."),
    outputs=[gr.JSON(label="Standardized Entities"), gr.Textbox(label="Summary Sentence")],
    title="Doctor Note Standardizer",
    description="Paste a clinical note to extract standardized entities and generate a summary."
)

if __name__ == "__main__":
    iface.launch()
