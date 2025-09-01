from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from emotions import emotions as e

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmotionAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        """ Emotion phrases generated using Claude Sonnet 4"""
        self.emotions = e        
        self.emotion_embeddings = self._compute_emotions()

    def get_embeddings(self, phrases):
        encoded_input = self.tokenizer(phrases, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        return F.normalize(mean_pooling(model_output, encoded_input['attention_mask']), p=2, dim=1)

    def _compute_emotions(self): 
        emotion_embeddings = {}

        for emotion, phrases in self.emotions.items():
            emotion_embeddings[emotion] = torch.mean(self.get_embeddings(phrases), dim=0)

        return emotion_embeddings

    def most_similar(self, lyrics: list[str]) -> str:
        if not lyrics or len(lyrics) == 0:
            raise ValueError("No lyrics provided")

        """ Returns a 2D tensor of the embeddings of each phrase in the song """
        lyrics_embeddings = self.get_embeddings(lyrics)

        emotion_scores = {}
        row_emotions = {}

        for row in range(lyrics_embeddings.shape[0]):
            lyric_embedding = lyrics_embeddings[row, :]

            for emotion, embedding in self.emotion_embeddings.items():
                """ Dot product of unit vectors gives us the (cosine of) angle between them"""
                emotion_scores[emotion] = torch.dot(lyric_embedding, embedding).item()
                row_emotions[row] = max(emotion_scores.items(), key=lambda x : x[1])[0]

        for emotion, _ in self.emotion_embeddings.items():
            emotion_scores[emotion] /= lyrics_embeddings.shape[0]

        return max(emotion_scores.items(), key=lambda x : x[1]), row_emotions


analyser = EmotionAnalyzer()
