Project title - Color Suggestion according to the Skin Tone

What it does
This project analyzes a face photo to estimate skin color. It crops the central area of the face, calculates color statistics and an average hex color, then asks a small text model to suggest matching colors. It shows a cropped face image, a color swatch image, and simple text details.

How to run it
1) Open a terminal in this folder.

2) Create and activate a virtual environment.
macOS or Linux:
python3 -m venv .venv
source .venv/bin/activate
Windows:
py -m venv .venv
.venv\Scripts\activate

3) Install dependencies.
pip install -r requirements.txt

4) Run the web app.
python app.py
Open the link shown in the terminal. Upload a face photo and click Analyze Image.

5) Run by command line (alternative).
python main.py --image path/to/your_image.jpg
The program prints suggested colors and saves a swatch image to palette_preview.png.
