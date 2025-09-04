#ashwin vijayakumar's submission for BDBI

import streamlit as st
from sentence_transformers import SentenceTransformer
import torch

#streamlist ui setup
st.title("Semantic FAQ Matcher")
st.write("Ask a question and get the most relevant FAQ answer!")

#load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

#faq dataset (used chatgpt to generate faq and answer responses. hardcoded for now)
faqs = [
    {"question": "How do I reset my password?", "answer": "Go to settings and click on 'Reset Password'."},
    {"question": "What is the refund policy?", "answer": "Refunds are allowed within 30 days of purchase."},
    {"question": "How can I contact support?", "answer": "Email us at support@example.com."},
    {"question": "Where can I find my invoices?", "answer": "Invoices are under 'Billing' in your account."},
    {"question": "Can I change my subscription plan?", "answer": "Yes, go to your account settings and select 'Change Plan'."},
    {"question": "How do I update my billing information?", "answer": "Go to 'Billing' settings and click on 'Update Payment Method'."},
    {"question": "Why was my payment declined?", "answer": "Payments may be declined due to insufficient funds, expired cards, or bank restrictions."},
    {"question": "Can I pause my subscription?", "answer": "Yes, you can pause your subscription for up to 3 months in your account settings."},
    {"question": "How do I delete my account?", "answer": "Navigate to 'Account Settings' and click 'Delete Account'. This action is permanent."},
    {"question": "How do I update my email address?", "answer": "Go to 'Profile Settings' and select 'Change Email'."},
    {"question": "Can I transfer my subscription to another account?", "answer": "Subscriptions are tied to your account and cannot be transferred."},
    {"question": "How do I change my shipping address?", "answer": "Go to 'Shipping Details' under your account to update your address."},
    {"question": "Do you offer student discounts?", "answer": "Yes, we offer 20% off for verified students."},
    {"question": "How do I cancel my subscription?", "answer": "You can cancel anytime under 'Subscription Settings' in your account."},
    {"question": "How long does delivery take?", "answer": "Standard shipping takes 5–7 business days, express shipping takes 2–3 days."},
    {"question": "Do you ship internationally?", "answer": "Yes, we ship to most countries worldwide. Additional fees may apply."},
    {"question": "Where can I track my order?", "answer": "You can track your order in 'Orders' under your account."},
    {"question": "How do I apply a promo code?", "answer": "Enter your promo code at checkout before completing your order."},
    {"question": "Can I get a replacement for a defective product?", "answer": "Yes, defective products are eligible for free replacement within 14 days."},
    {"question": "What payment methods do you accept?", "answer": "We accept credit cards, debit cards, PayPal, and Apple Pay."},
    {"question": "How do I download my purchase receipt?", "answer": "Receipts are available in 'Billing History' under your account."},
    {"question": "What should I do if I forgot my username?", "answer": "Click 'Forgot Username' on the login page and follow the instructions."},
    {"question": "Is my data secure?", "answer": "Yes, we use industry-standard encryption and do not share your personal data."},
    {"question": "Do you offer gift cards?", "answer": "Yes, digital gift cards are available for purchase on our website."},
    {"question": "How do I enable two-factor authentication?", "answer": "Go to 'Security Settings' and toggle on two-factor authentication."}
]


faq_questions = [faq['question'] for faq in faqs]

#embeddings for all FAQ questions
@st.cache_resource
def compute_embeddings(questions):
    return model.encode(questions, convert_to_tensor=True)

faq_embeddings = compute_embeddings(faq_questions)

#find similar FAQ based on user input
def find_most_similar_faq(user_query, faqs, faq_embeddings, model, threshold=0.5):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, faq_embeddings)
    best_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_idx].item()
    
    if best_score < threshold:
        return None, None, best_score
    else:
        return faqs[best_idx]['question'], faqs[best_idx]['answer'], best_score


#streamlit for ui because i am too lazy to use flask lol
user_query = st.text_input("Enter your question here:")
if user_query:
    matched_question, answer, score = find_most_similar_faq(user_query, faqs, faq_embeddings, model)
    
    if matched_question:
        st.subheader("Matched FAQ:")
        st.write(f"**Question:** {matched_question}")
        st.write(f"**Answer:** {answer}")
        st.write(f"**Similarity Score:** {score:.4f}")
    else:
        st.write("Sorry, no matching FAQ found. Try rephrasing your question.")
