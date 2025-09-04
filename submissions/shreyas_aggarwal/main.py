import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import requests
import random
import time

class TVShowRecommender:
    def __init__(self):
        # Load Hugging Face model
        print("Loading Hugging Face model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tvmaze_base_url = "https://api.tvmaze.com"
        self.shows_cache = []
        self.embeddings_cache = []
        self.current_page = 0
        self.page_size = 20
        self.user_data = {
            "liked": [],
            "disliked": [],
            "current_index": 0,
            "seen_show_ids": set()
        }
        # Load initial shows
        print("Fetching initial TV shows...")
        self.load_more_shows(initial_load=True)
        self.load_user_data()

    # Fetch shows from TVmaze API
    def fetch_shows_from_tmdb(self, page=1):
        try:
            url = f"{self.tvmaze_base_url}/shows"
            params = {"page": page - 1}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            shows = []
            for tv_show in data:
                if tv_show.get('summary') and tv_show.get('name'):
                    show = {
                        "id": tv_show['id'],
                        "title": tv_show['name'],
                        "genre": " / ".join(tv_show.get('genres', [])) or "General",
                        "year": tv_show.get('premiered', 'Unknown')[:4] if tv_show.get('premiered') else 'Unknown',
                        "description": tv_show['summary'].replace("<p>", "").replace("</p>", "").replace("<b>", "").replace("</b>", ""),
                        "rating": tv_show.get('rating', {}).get('average', 0),
                        "popularity": tv_show.get('weight', 0)
                    }
                    shows.append(show)
            return shows
        except Exception as e:
            print(f"API error: {e}")
            return []

    # Load more shows and add to cache
    def load_more_shows(self, initial_load=False):
        if initial_load:
            for page in range(1, 4):
                new_shows = self.fetch_shows_from_tmdb(page)
                self.add_shows_to_cache(new_shows)
                time.sleep(0.1)
        else:
            new_shows = self.fetch_shows_from_tmdb(self.current_page)
            self.add_shows_to_cache(new_shows)
            self.current_page += 1

    # Add new shows to cache and compute embeddings
    def add_shows_to_cache(self, new_shows):
        if not new_shows:
            return
        filtered_shows = []
        for show in new_shows:
            if show['id'] not in self.user_data['seen_show_ids']:
                filtered_shows.append(show)
                self.user_data['seen_show_ids'].add(show['id'])
        if not filtered_shows:
            return
        self.shows_cache.extend(filtered_shows)
        descriptions = [show['description'] for show in filtered_shows]
        if descriptions:
            new_embeddings = self.model.encode(descriptions)
            if len(self.embeddings_cache) == 0:
                self.embeddings_cache = new_embeddings
            else:
                self.embeddings_cache = np.vstack([self.embeddings_cache, new_embeddings])

    # Save user preferences
    def save_user_data(self):
        data_to_save = self.user_data.copy()
        data_to_save['seen_show_ids'] = list(self.user_data['seen_show_ids'])
        with open('user_preferences.json', 'w') as f:
            json.dump(data_to_save, f, indent=2)

    # Load user preferences
    def load_user_data(self):
        if os.path.exists('user_preferences.json'):
            with open('user_preferences.json', 'r') as f:
                loaded_data = json.load(f)
                self.user_data.update(loaded_data)
                self.user_data['seen_show_ids'] = set(self.user_data.get('seen_show_ids', []))

    # Get current show to rate
    def get_current_show(self):
        if self.user_data["current_index"] >= len(self.shows_cache) - 5:
            print("Loading more shows...")
            self.load_more_shows()
        if self.user_data["current_index"] >= len(self.shows_cache):
            self.user_data["current_index"] = random.randint(0, len(self.shows_cache) - 1)
        return self.shows_cache[self.user_data["current_index"]] if self.shows_cache else None

    # User likes a show
    def swipe_right(self):
        current_show = self.get_current_show()
        if current_show:
            self.user_data["liked"].append(current_show)
            self.user_data["current_index"] += 1
            self.save_user_data()
            print(f"Liked: {current_show['title']}")

    # User dislikes a show
    def swipe_left(self):
        current_show = self.get_current_show()
        if current_show:
            self.user_data["disliked"].append(current_show)
            self.user_data["current_index"] += 1
            self.save_user_data()
            print(f"Disliked: {current_show['title']}")

    # Generate recommendations using cached shows
    def generate_recommendations(self, top_k=10):
        if len(self.user_data["liked"]) == 0:
            print("No liked shows yet. Please rate some shows first!")
            return []

        liked_descriptions = [show["description"] for show in self.user_data["liked"]]
        liked_embeddings = self.model.encode(liked_descriptions)
        user_profile = np.mean(liked_embeddings, axis=0)

        rated_ids = set([show["id"] for show in self.user_data["liked"]] +
                        [show["id"] for show in self.user_data["disliked"]])

        unrated_shows = [show for show in self.shows_cache if show["id"] not in rated_ids]
        if not unrated_shows:
            print("Loading more shows for recommendations...")
            self.load_more_shows()
            unrated_shows = [show for show in self.shows_cache if show["id"] not in rated_ids]
        if not unrated_shows:
            return []

        unrated_indices = [self.shows_cache.index(show) for show in unrated_shows]
        unrated_embeddings = self.embeddings_cache[unrated_indices]
        similarities = cosine_similarity([user_profile], unrated_embeddings)[0]

        recommendations = []
        for i, show in enumerate(unrated_shows):
            recommendations.append({"show": show, "similarity_score": similarities[i]})
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations[:top_k]

    # Display a single show
    def display_show(self, show):
        print(f"\n{'='*60}")
        print(f"{show['title']} ({show['year']})")
        print(f"Genre: {show['genre']}")
        if 'rating' in show and show['rating']:
            print(f"Rating: {show['rating']}/10")
        print(f"{show['description']}")
        print(f"{'='*60}")

    # Display top recommendations
    def display_recommendations(self):
        recommendations = self.generate_recommendations()
        if not recommendations:
            return
        print(f"\nTOP {len(recommendations)} RECOMMENDATIONS")
        print("="*70)
        for i, rec in enumerate(recommendations, 1):
            show = rec["show"]
            score = rec["similarity_score"]
            print(f"{i}. {show['title']} ({show['year']})")
            print(f"   Genre: {show['genre']}")
            print(f"   Match Score: {score:.3f}")
            if 'rating' in show and show['rating']:
                print(f"   Rating: {show['rating']}/10")
            print(f"   Description: {show['description'][:100]}...")
            print()

    # Display user stats
    def display_stats(self):
        total_rated = len(self.user_data['liked']) + len(self.user_data['disliked'])
        print(f"\nYOUR STATS")
        print(f"Total Shows Rated: {total_rated}")
        print(f"Liked: {len(self.user_data['liked'])} shows")
        print(f"Disliked: {len(self.user_data['disliked'])} shows")
        print(f"Shows in Cache: {len(self.shows_cache)}")
        print(f"Current Position: Show #{self.user_data['current_index'] + 1}")
        if self.user_data['liked']:
            print(f"\nRECENT LIKED SHOWS:")
            for show in self.user_data['liked'][-5:]:
                print(f"  • {show['title']}")

    # Main loop
    def run(self):
        print("Welcome to Infinite TV Show Recommender!")
        print("Swipe Right (R) = Like, Swipe Left (L) = Dislike, skip = Skip")
        print("Type 'recs' for recommendations, 'stats' for statistics, 'quit' to exit")
        while True:
            try:
                current_show = self.get_current_show()
                if current_show is None:
                    print("Loading more shows...")
                    self.load_more_shows()
                    continue
                self.display_show(current_show)
                action = input(f"\nAction (Right=Like/Left=Dislike/skip/recs/stats/quit) [Show {self.user_data['current_index'] + 1}]: ").strip().lower()
                if action in ['r', 'right']:
                    self.swipe_right()
                elif action in ['l', 'left']:
                    self.swipe_left()
                elif action in ['skip', 's']:
                    self.user_data["current_index"] += 1
                    print("Skipped!")
                elif action == 'recs':
                    self.display_recommendations()
                elif action == 'stats':
                    self.display_stats()
                elif action == 'quit':
                    print("Thanks for using Infinite TV Show Recommender!")
                    break
                else:
                    print("Invalid action. Use R, L, skip, recs, stats, or quit")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Loading more content...")
                self.load_more_shows()


if __name__ == "__main__":
    print("Starting Infinite TV Show Recommender")
    recommender = TVShowRecommender()
    recommender.run()