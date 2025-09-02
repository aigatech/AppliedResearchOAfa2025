import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()

class PreferenceNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize the model, optimizer, and loss function
MODEL = PreferenceNet(EMBED_DIM)
OPTIM = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
LOSS_FN = nn.BCELoss()

def score_recipes(recipes):
    embeddings = torch.tensor(EMBED_MODEL.encode(recipes, convert_to_numpy=True), dtype=torch.float32)
    with torch.no_grad():
        scores = MODEL(embeddings)
    return scores.detach().numpy().flatten()


def update_model(chosen_recipe, rejected_recipe):
    recipes = [chosen_recipe, rejected_recipe]
    labels = torch.tensor([[1.0], [0.0]])
    embeddings = torch.tensor(EMBED_MODEL.encode(recipes, convert_to_numpy=True), dtype=torch.float32)
    
    MODEL.train()
    OPTIM.zero_grad()
    outputs = MODEL(embeddings)
    loss = LOSS_FN(outputs, labels)
    loss.backward()
    OPTIM.step()
    MODEL.eval()
    
    return loss.item()