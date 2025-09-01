from flask import Flask, request, render_template
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
import base64
from io import BytesIO 
from io import BytesIO
# output is a PIL.Image object

app=Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/ask',methods=['POST'])
def ask():
    client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
    )
    user_input=request.form['user_input']
    full_prompt=f"Context: You are thrown into a medieval world. You have become a knight in a fantastical world. You still have access to modern things though. Generate an image of: {user_input}"
    image = client.text_to_image(
    full_prompt,
    model="black-forest-labs/FLUX.1-dev",
    )
    buffered=BytesIO()
    image.save(buffered,format="PNG")
    img_str=base64.b64encode(buffered.getvalue()).decode("utf-8")
    answer=img_str
    try:
        pass
    except Exception as e:
        answer=f"Error:{str(e)}"
    return render_template('index.html',response=answer)
if __name__=='__main__':
    app.run(debug=True)

