---
title: AI Weather Analyst
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
---

# 🌐 AI Weather Analyst

## Overview
**AI Weather Analyst**  is a lightweight demo that transforms raw weather conditions into a professional, structured weather report using Hugging Face models and Gradio.

The app generates:
- 📰 A **headline** (5 worods, news-style)
- ✍️ A **summary** (one concise news-style sentence)
- 💡 An **insight** (a short practical tip)
- 🌡️ Structured **details** (condition, temperature, wind)

## Features
- Uses **FLAN-T5** from Hugging Face for natural  language generation
- Deterministic outputs (no random text generation for consistency)
- Supports curated forecasts for select cities (Atlanta, New York, Chicago, San Francisco)
- Fallback simulation for any other city with randomized conditions
- Interactive **Gradio UI** for easy use

## ▶️ How to Run  
1. Clone this repository and navigate into your submission folder:  
   ```bash
   git clone https://github.com/ashleyaechang/AppliedResearchOAfa2025.git
   cd AppliedResearchOAfa2025/submissions/Ashley_Chang
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the app:
   ```bash
   python app.py