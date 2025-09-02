from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import pandas as pd
from datasets import load_dataset
import requests
from io import BytesIO



# --------------------------------- Pokémon Prediction ------------------------------------------ #
# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and image processor
model_id = "skshmjn/Pokemon-classifier-gen9-1025"
model = ViTForImageClassification.from_pretrained(model_id).to(device)
image_processor = ViTImageProcessor.from_pretrained(model_id)

# Load and process an image
img = Image.open('squirtle.jpg').convert("RGB")
inputs = image_processor(images=img, return_tensors='pt').to(device)

# Make predictions
outputs = model(**inputs)
predicted_id = outputs.logits.argmax(-1).item()
predicted_pokemon = model.config.id2label[predicted_id]

# Print predicted class
print(f"Predicted Pokémon Pokédex number: {predicted_id+1}")
print(f"Predicted Pokémon: {predicted_pokemon}")

# --------------------------------- Pokemon Prediction ------------------------------------------ #



df = pd.read_csv("pokemon.csv")
input_pokemon_type1 = "" #Input pokemon type #1
input_pokemon_type2 = "" #Input pokemon type #2

#Finding Inputted pokemon's type
for i in range (0,len(df['Name'])):
    if (df['Name'][i] == predicted_pokemon):
        input_pokemon_type1 = df['Type 1'][i]
        input_pokemon_type2 = df['Type 2'][i]
        print("Type 1: " + str(input_pokemon_type1))
        print("Type 2: " + str(input_pokemon_type2))

print('''
      
''')

#Setting up type effectiveness strength according to imported csv
df = pd.read_csv("typing_chart.csv")
effective_dic_1 = {}
effective_dic_2 = {}
combined_effectiveness = {}
#Setting default dictionary type in case pokemon has no second type
for i in range(0,len(df['Types'])):
    effective_dic_2[df['Types'][i]] = 1.0

for i in range (0,len(df[input_pokemon_type1])):
    if pd.isnull(df[input_pokemon_type1][i]):
        effective_dic_1[df['Types'][i]] = 1.0
    else:
        effective_dic_1[df['Types'][i]] = (df[input_pokemon_type1][i])
    
if pd.isnull(input_pokemon_type2) == False:
    for i in range(0,len(df[input_pokemon_type2])):
        if pd.isnull(df[input_pokemon_type2][i]):
            effective_dic_2[df['Types'][i]] = 1.0
        else:
            effective_dic_2[df['Types'][i]] = (df[input_pokemon_type2][i])
    

#Combining both type effectivenesses
for key in effective_dic_1:
    combined_effectiveness[key] = effective_dic_1[key] * effective_dic_2[key]

print("Effectiveness Chart against " + predicted_pokemon + ": ")
print(combined_effectiveness)
print('''
      
''')

#Find Max Effective Type against input pokemon
max_effective_types = []
max_strength = 0 
for key in combined_effectiveness:
    if combined_effectiveness[key] >= max_strength:
        max_strength = combined_effectiveness[key]

for key in combined_effectiveness:
    if combined_effectiveness[key] == max_strength:
        max_effective_types.append(key)


effective_selection_dic = {} #Dictionary of all pokemon with best super effective type and their total strength stat
df = pd.read_csv("pokemon.csv")
#Finding Best Pokemon with Max Effective Type 
for i in range(1,len(df['Name'])):
    if df['Type 1'][i] in (max_effective_types) or df['Type 2'][i] in (max_effective_types):
        #Use Total Strength Stat
        effective_selection_dic[df['Name'][i]] = df['Total'][i]


# Sorting by descending values and replacing effective_selection_dic with sorted values
effective_selection_dic = dict(sorted(effective_selection_dic.items(), key=lambda item: item[1], reverse=True))
print("List of Best Pokemon to Counter " + predicted_pokemon + ": ")
print(effective_selection_dic)
print('''
      
''')

best_pokemon = list(effective_selection_dic.items())[0][0]
print(best_pokemon + " is the best pokemon to counter " + predicted_pokemon)

#Pull Image of Best Counter from PokeAPI
best_pokemon_number = 0
for i in range (0,len(df['#'])):
    if (df['Name'][i] == best_pokemon):
        best_pokemon_number = df['#'][i]
url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{best_pokemon_number}.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.show()


    









        


