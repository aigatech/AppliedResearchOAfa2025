import streamlit as st
import time

# Page congif
st.set_page_config(
    page_title="Simple Personality Quiz",
    page_icon="🧠",
    layout="wide"
)

QUESTIONS = [
    {
        "question": "I enjoy trying new experiences and activities",
        "trait": "openness",
        "positive": True
    },
    {
        "question": "I prefer to work without a plan",
        "trait": "conscientiousness", 
        "positive": False
    },
    {
        "question": "I feel energized after spending time with people",
        "trait": "extraversion",
        "positive": True
    },
    {
        "question": "I find it easy to forgive others",
        "trait": "agreeableness",
        "positive": True
    },
    {
        "question": "I worry about things",
        "trait": "neuroticism",
        "positive": False
    },
    {
        "question": "I like to keep things organized",
        "trait": "conscientiousness",
        "positive": True
    },
    {
        "question": "I prefer quiet, peaceful activities",
        "trait": "extraversion",
        "positive": False
    },
    {
        "question": "I can be critical of others",
        "trait": "agreeableness",
        "positive": False
    },
    {
        "question": "I handle stress well",
        "trait": "neuroticism",
        "positive": True
    },
    {
        "question": "I find abstract thinking stimulating",
        "trait": "openness",
        "positive": True
    }
]

TRAIT_DESCRIPTIONS = {
    "openness": {
        "name": "Openness to Experience",
        "description": "How open you are to new ideas, experiences, and creative thinking",
        "high": "You are creative, imaginative, and open to new experiences. You enjoy abstract thinking and exploring new ideas.",
        "low": "You prefer practical, concrete approaches and value tradition and routine.",
        "example": "High: Enjoys art, philosophy, trying new foods. Low: Prefers familiar routines and concrete tasks."
    },
    "conscientiousness": {
        "name": "Conscientiousness", 
        "description": "How organized, responsible, and goal-directed you are",
        "high": "You are organized, responsible, and goal-directed. You plan ahead and pay attention to details.",
        "low": "You are spontaneous and flexible, preferring to go with the flow rather than stick to strict plans.",
        "example": "High: Makes to-do lists, meets deadlines. Low: Prefers spontaneity, less structured approach."
    },
    "extraversion": {
        "name": "Extraversion",
        "description": "How outgoing, energetic, and socially engaged you are",
        "high": "You are outgoing, energetic, and socially confident. You gain energy from social interactions.",
        "low": "You are reserved and introspective, gaining energy from quiet activities and alone time.",
        "example": "High: Enjoys parties, public speaking. Low: Prefers small groups, needs alone time."
    },
    "agreeableness": {
        "name": "Agreeableness",
        "description": "How cooperative, trusting, and helpful you are toward others",
        "high": "You are cooperative, trusting, and helpful. You value harmony and get along well with others.",
        "low": "You are direct and competitive, prioritizing your own interests and being straightforward with others.",
        "example": "High: Avoids conflicts, helps others. Low: Direct communication, competitive nature."
    },
    "neuroticism": {
        "name": "Neuroticism",
        "description": "How sensitive you are to stress and negative emotions",
        "high": "You are sensitive and may experience more emotional ups and downs. You're in touch with your feelings.",
        "low": "You are emotionally stable and calm, handling stress well and maintaining consistent moods.",
        "example": "High: Worries easily, sensitive to criticism. Low: Calm under pressure, emotionally stable."
    }
}

def calculate_scores(responses):
    # Calculate personality scores
    scores = {trait: 0 for trait in TRAIT_DESCRIPTIONS.keys()}
    counts = {trait: 0 for trait in TRAIT_DESCRIPTIONS.keys()}
    
    for i, response in enumerate(responses):
        if response is not None: 
            question = QUESTIONS[i]
            trait = question["trait"]
            positive = question["positive"]
            
            # Score: 1 for positive response to positive question, or negative response to negative question
            if (positive and response == "agree") or (not positive and response == "disagree"):
                scores[trait] += 1
            elif (positive and response == "disagree") or (not positive and response == "agree"):
                scores[trait] -= 1
            
            counts[trait] += 1
    
    # Calculate percentages
    percentages = {}
    for trait in scores:
        if counts[trait] > 0:
            raw_score = scores[trait]
            percentage = max(0, min(100, 50 + (raw_score * 25)))
            percentages[trait] = percentage
        else:
            percentages[trait] = 50 
    
    return percentages

def get_personality_summary(scores):
    # personality summary
    summary = []
    
    for trait, score in scores.items():
        trait_info = TRAIT_DESCRIPTIONS[trait]
        
        if score >= 70:
            level = "high"
            description = trait_info["high"]
        elif score <= 30:
            level = "low" 
            description = trait_info["low"]
        else:
            level = "moderate"
            description = f"You show balanced characteristics in {trait_info['name'].lower()}."
        
        summary.append({
            "trait": trait_info["name"],
            "score": score,
            "level": level,
            "description": description,
            "example": trait_info["example"]
        })
    
    return summary

def main():
    st.title("Simple Personality Quiz")
    st.markdown("Discover your personality traits in just 10 questions!")
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.responses = [None] * len(QUESTIONS)
        st.session_state.quiz_complete = False
    
    if not st.session_state.quiz_complete:
        st.header(f"Question {st.session_state.current_question + 1} of {len(QUESTIONS)}")
        
        question = QUESTIONS[st.session_state.current_question]
        st.markdown(f"**{question['question']}**")
        st.markdown(f"*Trait: {TRAIT_DESCRIPTIONS[question['trait']]['name']}*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Disagree", use_container_width=True):
                st.session_state.responses[st.session_state.current_question] = "disagree"
                next_question()
                
        with col2:
            if st.button("Neutral", use_container_width=True):
                st.session_state.responses[st.session_state.current_question] = "neutral"
                next_question()
                
        with col3:
            if st.button("Agree", use_container_width=True):
                st.session_state.responses[st.session_state.current_question] = "agree"
                next_question()
        
        progress = (st.session_state.current_question + 1) / len(QUESTIONS)
        st.progress(progress)
        
        if st.session_state.current_question > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_question -= 1
                st.rerun()
    
    else:
        st.header("Your Personality Results")
        
        scores = calculate_scores(st.session_state.responses)
        summary = get_personality_summary(scores)
        
        for trait_summary in summary:
            with st.expander(f"{trait_summary['trait']} - {trait_summary['score']:.0f}/100"):
                st.markdown(f"**Level:** {trait_summary['level'].title()}")
                st.markdown(f"**Description:** {trait_summary['description']}")
                st.markdown(f"**Example:** {trait_summary['example']}")
        
        dominant_trait = max(scores.items(), key=lambda x: x[1])
        st.success(f"**Your dominant trait:** {TRAIT_DESCRIPTIONS[dominant_trait[0]]['name']}")
        
        if st.button("Take Quiz Again"):
            st.session_state.current_question = 0
            st.session_state.responses = [None] * len(QUESTIONS)
            st.session_state.quiz_complete = False
            st.rerun()

def next_question():
    # Move to next question
    if st.session_state.current_question < len(QUESTIONS) - 1:
        st.session_state.current_question += 1
    else:
        st.session_state.quiz_complete = True
    st.rerun()

if __name__ == "__main__":
    main()