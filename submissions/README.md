# Translation Effects on Emotional Expression Across LanguagesRetry

## What This Project Does
Takes English text, translates it to 4 languages (French, Spanish, German, Hindi), then runs an English emotion classifier on all versions to see how translation affects emotional perception. \n

Setup: Load translation pipelines for 4 languages + English emotion classifier \n
Process: Take user input → translate to 4 languages → run emotion analysis on each \n
Output: Display translations with emotion scores, create comparison table showing percentage differences across languages \n

## Purpose
The aim of this project was to see how direct translations can differ in emotional intention. The same phrase "directly" translated leads to very different sentiments across languages. \n

## Instructions to Run File
1. Install dependencies (listed at the top of the code document) \n
2. Run the file directly! \n

## Example Output
English text: Hello World!!
         French Spanish  German   Hindi \n
anger      4.56%   8.06%  10.40%  18.80% \n
disgust    2.01%   2.38%  12.07%   1.14% \n
fear       1.08%   3.52%   4.29%   8.42% \n
joy       51.27%  62.37%   7.08%   9.37% \n
neutral   20.47%  16.36%  33.56%  38.54% \n
sadness    2.58%   2.73%   7.58%   2.34% \n
surprise  18.02%   4.57%  25.03%  21.40% \n



