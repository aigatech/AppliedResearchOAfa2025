import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import random


st.set_page_config(
    page_title="StudyCircle - Your Study Hub",
    page_icon="🌀",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Poppins', sans-serif !important;
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background-color: #002a4e;
        padding: 1rem;
        border-radius: 10px;
    }


    

    .stApp, .stApp * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTextArea > div > div > textarea {
        font-family: 'Poppins', sans-serif !important;
    }
    

    .stMarkdown, .stMarkdown * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stSelectbox > div > div {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stSlider > div > div {
        font-family: 'Poppins', sans-serif !important;
    }
    

    .stButton > button {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
        border-radius: 0.75rem !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 0 1px #307fed !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        background-color: rgba(48, 127, 237, 0.2) !important;
        transform: translateY(0) !important;
    }
    

    .stButton > button p {
        font-weight: 600 !important;
        color: #ffffff !important;
        margin: 0 !important;
    }
    

    .stApp {
        background-color: #002a4e !important;
    }
    

    .stAppHeader {
        background-color: #002a4e !important;
    }
    
    .stAppHeader > div {
        background-color: #002a4e !important;
    }
    

    .stAppHeader > div > div > div {
        background-color: #002a4e !important;
    }
    
    .main .block-container {
        background-color: #002a4e !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    

    .stMarkdown, .stMarkdown * {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #002a4e !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
    }
    
    .stTextInput > div > div > input:focus {
        background-color: #002a4e !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
        box-shadow: 0 0 0 2px rgba(48, 127, 237, 0.3) !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #002a4e !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        background-color: #002a4e !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
        box-shadow: 0 0 0 2px rgba(48, 127, 237, 0.3) !important;
    }
    
    .stSelectbox > div > div {
        background-color: #002a4e !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
    }
    
    .stSelectbox > div > div:focus {
        background-color: #002a4e !important;
        color: #ffffff !important;
        border: 1px solid #307fed !important;
        box-shadow: 0 0 0 2px rgba(48, 127, 237, 0.3) !important;
    }
    
    .stSlider > div > div {
        color: #ffffff !important;
    }
    

    .stContainer, .stContainer * {
        background-color: #002a4e !important;
    }
    

    .stAlert {
        background-color: rgba(48, 127, 237, 0.1) !important;
        border: 1px solid #307fed !important;
        color: #ffffff !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1) !important;
        border: 1px solid #ffc107 !important;
        color: #ffffff !important;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1) !important;
        border: 1px solid #dc3545 !important;
        color: #ffffff !important;
    }
    
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border: 1px solid #28a745 !important;
        color: #ffffff !important;
    }

    div[data-testid="stButton"] > button[data-testid="stButton-secondary"] {
        background-color: #f44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():

    models = {}
    
    try:
        with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):

            models['summarizer'] = pipeline("summarization", model="facebook/bart-large-cnn")
            

            models['qg_tokenizer'] = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
            models['qg_model'] = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
            

            models['translator'] = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
            

            models['embedding_tokenizer'] = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            models['embedding_model'] = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        return models
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

def get_embeddings(text, tokenizer, model):

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def find_synonyms(word, tokenizer, model, top_k=5):


    common_words = [
        "happy", "sad", "big", "small", "fast", "slow", "good", "bad", "beautiful", "ugly",
        "smart", "dumb", "strong", "weak", "hot", "cold", "new", "old", "young", "rich",
        "poor", "easy", "hard", "simple", "complex", "clean", "dirty", "safe", "dangerous",
        "important", "useless", "interesting", "boring", "funny", "serious", "loud", "quiet"
    ]
    
    word_embedding = get_embeddings(word, tokenizer, model)
    
    similarities = []
    for common_word in common_words:
        if common_word.lower() != word.lower():
            common_embedding = get_embeddings(common_word, tokenizer, model)
            similarity = cosine_similarity(word_embedding, common_embedding).item()
            similarities.append((common_word, similarity))
    

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def generate_questions(text, tokenizer, model, num_questions=3):

    questions = []
    

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    

    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))
    
    for sentence in selected_sentences:
        if len(sentence) > 10:
            input_text = f"generate question: {sentence}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )
            
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            

            if question.lower().startswith("generate question:"):
                question = question[17:].strip()
            elif question.lower().startswith("question:"):
                question = question[9:].strip()
            

            if len(question) > 5 and question.endswith('?'):
                questions.append(question)
    
    return questions

def main():

    st.markdown('<h1 class="main-header">🌀 StudyCircle</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">Your AI-Powered Study Hub</p>', unsafe_allow_html=True)
    

    models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check your internet connection and try again.")
        return
    

    st.markdown('<div style="text-align: center; margin-bottom: 3rem;">', unsafe_allow_html=True)
    

    st.markdown("""
    <script>
    setTimeout(function() {
        const buttons = document.querySelectorAll('button[kind="secondary"]');
        buttons.forEach(button => {
            const text = button.textContent.trim();
            if (text === 'Study Summarizer') {
                button.style.backgroundColor = '#ffefd7';
                button.style.color = '#000000';
                button.style.border = '1px solid #ffefd7';
            } else if (text === 'Quiz Helper') {
                button.style.backgroundColor = '#a900ff';
                button.style.color = '#ffffff';
                button.style.border = '1px solid #a900ff';
            } else if (text === 'Language Assistant') {
                button.style.backgroundColor = '#92d1ae';
                button.style.color = '#000000';
                button.style.border = '1px solid #92d1ae';
            } else if (text === 'Synonym Finder') {
                button.style.backgroundColor = '#ffcc00';
                button.style.color = '#000000';
                button.style.border = '1px solid #ffcc00';
            }
        });
    }, 100);
    </script>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Study Summarizer", key="summarizer_btn", use_container_width=True):
            st.session_state.selected_tool = "Study Summarizer"
    
    with col2:
        if st.button("Quiz Helper", key="quiz_btn", use_container_width=True):
            st.session_state.selected_tool = "Quiz Helper"
    
    with col3:
        if st.button("Language Assistant", key="language_btn", use_container_width=True):
            st.session_state.selected_tool = "Language Assistant"
    
    with col4:
        if st.button("Synonym Finder", key="synonym_btn", use_container_width=True):
            st.session_state.selected_tool = "Synonym Finder"
    
    st.markdown('</div>', unsafe_allow_html=True)
    

    if 'selected_tool' not in st.session_state:
        st.session_state.selected_tool = None
    

    tool = st.session_state.selected_tool
    

    if tool is None:
        st.markdown("""
        <div style="text-align: center; margin-top: 4rem; padding: 3rem; background-color: #002a4e; border-radius: 15px; border: 2px dashed #307fed;">
            <h3 style="color: #ffffff; margin-bottom: 1rem;">🎯 Choose a Study Tool Above</h3>
            <p style="color: #ffffff; font-size: 1.1rem; margin: 0;">
                Select any of the four AI-powered study tools to get started with your learning journey!
            </p>
        </div>
        """, unsafe_allow_html=True)
    

    elif tool == "Study Summarizer":

        st.header("📝 Study Summarizer")
        st.write("Upload your study material and get AI-generated summaries!")
        
        text_input = st.text_area(
            "Enter your study material:",
            height=200,
            placeholder="Paste your notes, textbook content, or any study material here..."
        )
        
        if st.button("Generate Summary", type="primary"):
            if text_input:
                if len(text_input) < 100:
                    st.warning("Please enter at least 100 characters for a meaningful summary.")
                else:
                    with st.spinner("Generating summary..."):
                        try:
                            summary = models['summarizer'](text_input, max_length=150, min_length=50, do_sample=False)

                            st.subheader("📋 Summary:")
                            st.write(summary[0]['summary_text'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
            else:
                st.warning("Please enter some text to summarize.")
        st.markdown('</div>', unsafe_allow_html=True)
    

    elif tool == "Quiz Helper":

        st.header("❓ Quiz Helper")
        st.write("Generate practice questions from your study material!")
        
        quiz_text = st.text_area(
            "Enter content to generate questions from:",
            height=200,
            placeholder="Paste your study material here to generate practice questions..."
        )
        
        num_questions = st.slider("Number of questions to generate:", 1, 5, 3)
        
        if st.button("Generate Questions", type="primary"):
            if quiz_text:
                if len(quiz_text) < 50:
                    st.warning("Please enter at least 50 characters for question generation.")
                else:
                    with st.spinner("Generating questions..."):
                        try:
                            questions = generate_questions(
                                quiz_text, 
                                models['qg_tokenizer'], 
                                models['qg_model'], 
                                num_questions
                            )
                            

                            st.subheader("🎯 Practice Questions:")
                            for i, question in enumerate(questions, 1):
                                st.write(f"**{i}.** {question}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating questions: {str(e)}")
            else:
                st.warning("Please enter some text to generate questions from.")
        st.markdown('</div>', unsafe_allow_html=True)
    

    elif tool == "Language Assistant":

        st.header("🌍 Language Assistant")
        st.write("Practice Spanish with AI-powered translation!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("English → Spanish")
            english_text = st.text_area(
                "Enter English text:",
                height=150,
                placeholder="Type your English text here..."
            )
            
            if st.button("Translate to Spanish", type="primary"):
                if english_text:
                    with st.spinner("Translating..."):
                        try:
                            translation = models['translator'](english_text)

                            st.subheader("🇪🇸 Spanish Translation:")
                            st.write(translation[0]['translation_text'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error translating: {str(e)}")
                else:
                    st.warning("Please enter some English text to translate.")
        
        with col2:
            st.subheader("💡 Study Tips")
            st.info("""
            **Language Learning Tips:**
            - Practice translating short sentences daily
            - Try to understand the grammar structure
            - Use the translations to build vocabulary
            - Practice speaking the translated sentences aloud
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    

    elif tool == "Synonym Finder":

        st.header("🔍 Synonym Finder")
        st.write("Find similar words to expand your vocabulary!")
        
        word = st.text_input(
            "Enter a word to find synonyms:",
            placeholder="Type any word here..."
        )
        
        num_synonyms = st.slider("Number of synonyms to find:", 3, 10, 5)
        
        if st.button("Find Synonyms", type="primary"):
            if word:
                if len(word.split()) > 1:
                    st.warning("Please enter only a single word.")
                else:
                    with st.spinner("Finding synonyms..."):
                        try:
                            synonyms = find_synonyms(
                                word, 
                                models['embedding_tokenizer'], 
                                models['embedding_model'], 
                                num_synonyms
                            )
                            

                            st.subheader(f"🔍 Synonyms for '{word}':")
                            for i, (synonym, similarity) in enumerate(synonyms, 1):
                                similarity_percent = similarity * 100
                                st.write(f"**{i}.** {synonym} (similarity: {similarity_percent:.1f}%)")
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error finding synonyms: {str(e)}")
            else:
                st.warning("Please enter a word to find synonyms for.")
        st.markdown('</div>', unsafe_allow_html=True)
    

    st.markdown("---")
    st.markdown(
        "© 2025 StudyCircle | Guilherme Luvielmo - AI@GT Fall 2025"
    )

if __name__ == "__main__":
    main()
