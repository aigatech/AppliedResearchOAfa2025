
import spacy



def clean_text(text):

    nlp = spacy.load("en_core_web_md")

    # Remove leading and trailing whitespace
    cleaned_text = text.strip()

    # # Replace new lines with spaces
    # cleaned_text = cleaned_text.replace('\n', ' ')

    # Replace multiple spaces with a single space
    cleaned_text = ' '.join(cleaned_text.split())

    data = nlp(cleaned_text)

    data_sentences = list(map(str, list(data.sents)))

    return data_sentences



def read_txt(path):

    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text



def write_txt(path, text):

    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)