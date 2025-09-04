Analysis of the Hybrid Adversarial Evolver

Introduction

The Hybrid Adversarial Evolver project explores how well a language model performs on a small Q&A dataset, built specifically for a 1.5-hour AI club interview demo. The evaluation pipeline is powered by Hugging Face’s transformers library and runs the microsoft/phi-2 model (~2.7B parameters) on data/seed.jsonl. The setup is optimized for an ASUS ROG Zephyrus G14 with an RTX 4070 GPU (8GB VRAM, 32GB RAM, Windows, CUDA 12.4).

Methodology

The script loads microsoft/phi-2, reads a JSONL dataset containing about a dozen factual Q&A pairs, and scores responses using both keyword checks and similarity matching (≥80% threshold with SequenceMatcher). Earlier, the larger openai/gpt-oss-20b (20B parameters) was tested, but its Mixture of Experts design and reliance on Triton caused compatibility issues on Windows. Switching to microsoft/phi-2 provided a better fit for 8GB VRAM, using bfloat16 precision and Hugging Face’s Dataset API for efficient batching.

Results

Out of 12 test questions, the model answered 9 correctly, giving a base accuracy of 0.75. This was an improvement over the initial 0.50 score, thanks to the added keyword normalization (e.g., mapping “four” to “4”) and similarity scoring. Correct answers included math (“2+2=4”), geography (“Paris” as the capital of France), and biology (“the pancreas produces insulin”). Errors included predicting “Pluto” instead of “Neptune,” returning no answer for “Iron,” and confusing “Kazakhstan” with “Kosovo.” These mistakes likely reflect Phi-2’s smaller size and limited factual recall.

Challenges and Workarounds

The main hurdles were limited GPU memory, Triton’s lack of Windows support, and overly strict substring matching. These were addressed by moving to microsoft/phi-2, filtering out noisy logs, and introducing more flexible keyword/similarity scoring. Using the datasets library also sped up data handling.

Conclusion

Overall, the pipeline demonstrated that microsoft/phi-2 can achieve 0.75 accuracy on the demo dataset, making it a practical choice for the AI club showcase. Future improvements could include fine-tuning the model, expanding the dataset, or integrating more advanced evaluation metrics such as BLEU scores for greater robustness.