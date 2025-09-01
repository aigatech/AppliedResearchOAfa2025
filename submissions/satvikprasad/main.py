from flask import Flask, render_template, request

from analyzer import EmotionAnalyzer
from scraper import Scraper

app = Flask(__name__)

emotion_analyzer = EmotionAnalyzer()

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/api/similarity", methods=['GET'])
def similarity():
    query = request.args.get('phrase')
    
    lyrics_scraper = Scraper()

    title, phrases = lyrics_scraper.get_lyrics(search_query=query)

    try:
        emotion, row_emotions = emotion_analyzer.most_similar(phrases)

        lyric_fragment = "<div class=\"lyric_preview\">"

        for row, emotion in row_emotions.items():
            lyric_fragment += f'''
            <div class="lyric-row-item">
                <p class="lyric-row-lyric">{phrases[row]}</p>
                <p class="lyric-row-emotion"><b>{emotion}</b></p>
            </div>
            '''

        lyric_fragment += "</div>"

        return f'''
            <div id="summary" hx-swap-oob="true">
                <p>The primary emotion in your song is <span class="capitalize"><b>{emotion}</b></span>.</p>
                <p class="center">The phrases we analysed:</p>
                {lyric_fragment}
            </div>
            <div id="summary-list" hx-swap-oob="beforeend">
                <div class="summary-list-item">
                    <h2>{title}</h2>
                    <p>Emotion: <span class="capitalize"><b>{emotion}</b></span></p>
                </div>
            </div>
        '''
    except ValueError:
        return '''
            <div id="summary" hx-swap-oob="true">
                <p>An error occurred. Maybe your song contains no lyrics?</p>
            <div>
        '''




if __name__ == "__main__":
    app.run(debug = True)
