from lyricsgenius import Genius

class Scraper:
    def __init__(self):
        """ Access token is public here so I don't have to commit a .env to the repo, secrets in production will definitely not be exposed I promise :) """
        self.genius = Genius("NCEotz6MUTqQNDfzlq6_cXTtnre6Of33ehtocWGeBleSEwemZJy-wFr1klwBjoci")

    def get_lyrics(self, search_query):
        if search_query == None:
            return []
        
        """ Don't know why this package doesn't automatically escape problematic characters but oh well """
        song = self.genius.search_song(title=search_query.replace("'", "\'"))

        if song == None:
            return "", []

        return song.full_title, [x for x in song.lyrics.splitlines() if len(x) > 0 and x[0] != "["]

