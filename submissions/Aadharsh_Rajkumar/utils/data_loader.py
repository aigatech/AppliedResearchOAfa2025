from datasets import load_dataset

def load_imdb_dataset(split = "train", limit = None):
    """
    Load our IMDB dataset from HuggingFace
    """
    dataset = load_dataset("imdb", split=split)
    if limit:
        dataset = dataset.select(range(limit))
    return dataset