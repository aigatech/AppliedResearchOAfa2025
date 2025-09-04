This application works as a simple classifier between fake news and real news, trained on a news dataset from Hugging Face (using the distillbert base model).
Given a URL input from the user, it parses the contents at the URL and predicts whether the news is real or fake based on the context in the dataset.
The model is not entirely acccurate for newer articles, since it lacks the grounding data to determine the truth of the information.
The libraries required are as follows:
- datasets
- transformers
- bs4 (URL text parser)
- scikit-learn
- pandas
- requests
