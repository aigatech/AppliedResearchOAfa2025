#!/usr/bin/env python3

from scrape import Scraper
from summarizer import Summarizer

link = "https://arxiv.org/pdf/2505.17612" #CHANGE ME TO ANY RESEARCH PAPER OR TECHNICAL BLOG! 
text = Scraper(link).extract()

summarizer = Summarizer()

print("Loading summary...\n")
summary = summarizer.summarize(text)
print("=== QUICK SUMMARY ===\n")
print(summary["summary"])   

print("\nLoading questions...\n")
questions = summarizer.questions_for_sections(text)

print("=== SECTION QUESTIONS ===")
for item in questions:
    print(f"\n[{item['section']}] (chunk {item['chunk_index']})")
    print("Q:", item["question"])
