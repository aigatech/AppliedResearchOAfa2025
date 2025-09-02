import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class TrieNode:
    """Trie node for efficient prefix matching in lexicon"""
    def __init__(self):
        self.children = {}
        self.emoji = None
        self.is_word = False

class EmojiTrie:
    """Trie data structure for efficient emoji lexicon lookup"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, emoji: str):
        """Insert word-emoji pair into trie"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_word = True
        node.emoji = emoji
    
    def search(self, word: str) -> Optional[str]:
        """Search for exact word match"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node.emoji if node.is_word else None
    
    def get_prefix_matches(self, prefix: str) -> List[tuple]:
        """Get all words that start with given prefix"""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        matches = []
        self._collect_words(node, prefix, matches)
        return matches
    
    def _collect_words(self, node: TrieNode, prefix: str, matches: List[tuple]):
        """Recursively collect all words from a node"""
        if node.is_word:
            matches.append((prefix, node.emoji))
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, prefix + char, matches)

class EnhancedLexicon:
    """Enhanced lexicon with multiple lookup strategies and caching"""
    
    def __init__(self, lexicon_data: Dict[str, str]):
        self.data = lexicon_data
        self.trie = EmojiTrie()
        self.category_map = defaultdict(list)
        self.emoji_to_words = defaultdict(list)
        
        # Build optimized data structures
        self._build_trie()
        self._build_category_map()
        self._build_reverse_map()
        
        logger.info(f"Enhanced lexicon loaded with {len(lexicon_data)} entries")
    
    def _build_trie(self):
        """Build trie for efficient prefix matching"""
        for word, emoji in self.data.items():
            self.trie.insert(word, emoji)
    
    def _build_category_map(self):
        """Build category mappings (basic emotion/object categorization)"""
        emotion_words = {'happy', 'sad', 'angry', 'love', 'hate', 'excited', 'worried', 'scared'}
        food_words = {'pizza', 'burger', 'cake', 'coffee', 'beer', 'wine', 'apple', 'banana'}
        animal_words = {'cat', 'dog', 'bird', 'fish', 'elephant', 'lion', 'tiger', 'bear'}
        
        for word, emoji in self.data.items():
            if word in emotion_words:
                self.category_map['emotions'].append((word, emoji))
            elif word in food_words:
                self.category_map['food'].append((word, emoji))
            elif word in animal_words:
                self.category_map['animals'].append((word, emoji))
            else:
                self.category_map['other'].append((word, emoji))
    
    def _build_reverse_map(self):
        """Build reverse mapping from emojis to words"""
        for word, emoji in self.data.items():
            self.emoji_to_words[emoji].append(word)
    
    def get(self, word: str) -> Optional[str]:
        """Get emoji for exact word match"""
        return self.data.get(word.lower())
    
    def search_trie(self, word: str) -> Optional[str]:
        """Search using trie structure"""
        return self.trie.search(word)
    
    def get_prefix_matches(self, prefix: str, limit: int = 5) -> List[tuple]:
        """Get words starting with prefix"""
        matches = self.trie.get_prefix_matches(prefix)
        return matches[:limit]
    
    def get_category_words(self, category: str) -> List[tuple]:
        """Get words from specific category"""
        return self.category_map.get(category, [])
    
    def get_similar_emojis(self, emoji: str) -> List[str]:
        """Get words that map to the same emoji"""
        return self.emoji_to_words.get(emoji, [])
    
    def get_stats(self) -> Dict[str, int]:
        """Get lexicon statistics"""
        unique_emojis = len(set(self.data.values()))
        return {
            'total_words': len(self.data),
            'unique_emojis': unique_emojis,
            'avg_words_per_emoji': len(self.data) / unique_emojis if unique_emojis > 0 else 0,
            'categories': len(self.category_map)
        }

def load_lexicon(path: str = "data/emoji_lexicon.json") -> Dict[str, str]:
    """Load emoji lexicon with error handling and fallback"""
    lexicon_path = Path(path)
    
    cache_path = lexicon_path.with_suffix('.pkl')
    if cache_path.exists() and cache_path.stat().st_mtime > lexicon_path.stat().st_mtime:
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded lexicon from cache: {len(cached_data)} entries")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cached lexicon: {e}")
    
    if not lexicon_path.exists():
        logger.warning(f"Lexicon file not found: {lexicon_path}")
        return create_default_lexicon()
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            logger.error("Invalid lexicon format - expected dictionary")
            return create_default_lexicon()
        
        cleaned_data = {}
        for word, emoji in data.items():
            if isinstance(word, str) and isinstance(emoji, str) and word.strip() and emoji.strip():
                cleaned_data[word.lower().strip()] = emoji.strip()
        
        logger.info(f"Loaded lexicon: {len(cleaned_data)} entries")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cleaned_data, f)
            logger.debug("Saved lexicon to cache")
        except Exception as e:
            logger.warning(f"Failed to save lexicon cache: {e}")
        
        return cleaned_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in lexicon file: {e}")
        return create_default_lexicon()
    except Exception as e:
        logger.error(f"Error loading lexicon: {e}")
        return create_default_lexicon()

def create_default_lexicon() -> Dict[str, str]:
    """Create a basic default lexicon if main file is missing"""
    logger.info("Creating default lexicon")
    
    default_lexicon = {
        # Emotions
        "happy": "😊", "sad": "😢", "love": "❤️", "angry": "😠",
        "excited": "🤗", "worried": "😟", "scared": "😨", "surprised": "😲",
        
        # Actions
        "run": "🏃", "walk": "🚶", "dance": "💃", "sing": "🎤",
        "sleep": "😴", "eat": "🍽️", "drink": "🥤", "work": "💼",
        
        # Objects
        "car": "🚗", "house": "🏠", "phone": "📱", "computer": "💻",
        "book": "📚", "music": "🎵", "movie": "🎬", "game": "🎮",
        
        # Animals
        "cat": "🐱", "dog": "🐶", "bird": "🐦", "fish": "🐟",
        "elephant": "🐘", "lion": "🦁", "tiger": "🐅", "bear": "🐻",
        
        # Food
        "pizza": "🍕", "burger": "🍔", "cake": "🎂", "coffee": "☕",
        "beer": "🍺", "wine": "🍷", "apple": "🍎", "banana": "🍌",
        
        # Weather
        "sun": "☀️", "rain": "🌧️", "snow": "❄️", "wind": "💨",
        "hot": "🔥", "cold": "🧊", "cloudy": "☁️", "storm": "⛈️",
        
        # Time
        "morning": "🌅", "night": "🌙", "day": "☀️", "evening": "🌇",
        
        # Nature
        "tree": "🌳", "flower": "🌸", "mountain": "⛰️", "ocean": "🌊",
        "fire": "🔥", "water": "💧", "earth": "🌍", "star": "⭐"
    }
    
    return default_lexicon

def load_enhanced_lexicon(path: str = "data/emoji_lexicon.json") -> EnhancedLexicon:
    """Load enhanced lexicon with optimized data structures"""
    lexicon_data = load_lexicon(path)
    return EnhancedLexicon(lexicon_data)

def validate_lexicon(lexicon: Dict[str, str]) -> tuple[bool, List[str]]:
    """Validate lexicon data and return issues found"""
    issues = []
    
    if not isinstance(lexicon, dict):
        return False, ["Lexicon must be a dictionary"]
    
    if not lexicon:
        issues.append("Lexicon is empty")
    
    for word, emoji in lexicon.items():
        if not isinstance(word, str) or not word.strip():
            issues.append(f"Invalid word: {repr(word)}")
        
        if not isinstance(emoji, str) or not emoji.strip():
            issues.append(f"Invalid emoji for word '{word}': {repr(emoji)}")
        
        if emoji and not any(e['emoji'] for e in __import__('emoji').emoji_list(emoji)):
            issues.append(f"'{emoji}' for word '{word}' may not be a valid emoji")
    
    return len(issues) == 0, issues

def merge_lexicons(*lexicon_paths: str) -> Dict[str, str]:
    """Merge multiple lexicon files, with later files taking precedence"""
    merged = {}
    
    for path in lexicon_paths:
        try:
            lexicon = load_lexicon(path)
            merged.update(lexicon)
            logger.info(f"Merged {len(lexicon)} entries from {path}")
        except Exception as e:
            logger.warning(f"Failed to load lexicon from {path}: {e}")
    
    logger.info(f"Final merged lexicon: {len(merged)} entries")
    return merged

def export_lexicon(lexicon: Dict[str, str], output_path: str, format: str = 'json'):
    """Export lexicon to different formats"""
    output_path = Path(output_path)
    
    if format.lower() == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    elif format.lower() == 'csv':
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'emoji'])
            for word, emoji in sorted(lexicon.items()):
                writer.writerow([word, emoji])
    
    elif format.lower() == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for word, emoji in sorted(lexicon.items()):
                f.write(f"{word}: {emoji}\n")
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Exported {len(lexicon)} entries to {output_path}")

if __name__ == "__main__":
    """Command line interface for lexicon utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emoji lexicon utilities")
    parser.add_argument('--validate', type=str, help='Validate lexicon file')
    parser.add_argument('--stats', type=str, help='Show lexicon statistics')
    parser.add_argument('--export', type=str, help='Export lexicon to different format')
    parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json', help='Export format')
    parser.add_argument('--merge', nargs='+', help='Merge multiple lexicon files')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.validate:
        lexicon = load_lexicon(args.validate)
        is_valid, issues = validate_lexicon(lexicon)
        
        if is_valid:
            print(f"Lexicon is valid ({len(lexicon)} entries)")
        else:
            print("Lexicon has issues:")
            for issue in issues[:10]: 
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
    
    elif args.stats:
        enhanced_lex = load_enhanced_lexicon(args.stats)
        stats = enhanced_lex.get_stats()
        
        print("📊 Lexicon Statistics:")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Unique emojis: {stats['unique_emojis']}")
        print(f"  Avg words per emoji: {stats['avg_words_per_emoji']:.2f}")
        print(f"  Categories: {stats['categories']}")
        
        print("\n📂 Category breakdown:")
        for category, items in enhanced_lex.category_map.items():
            print(f"  {category}: {len(items)} items")
    
    elif args.merge:
        merged = merge_lexicons(*args.merge)
        
        if args.output:
            export_lexicon(merged, args.output, args.format)
        else:
            print(f"Merged {len(merged)} entries")
            for word, emoji in list(merged.items())[:5]:
                print(f"  {word}: {emoji}")
            if len(merged) > 5:
                print(f"  ... and {len(merged) - 5} more")
    
    elif args.export:
        lexicon = load_lexicon(args.export)
        
        if not args.output:
            print("Error: --output required for export")
        else:
            export_lexicon(lexicon, args.output, args.format)
    
    else:
        lexicon = load_lexicon()
        print(f"Loaded lexicon with {len(lexicon)} entries")
        
        sample_items = list(lexicon.items())[:5]
        print("Sample entries:")
        for word, emoji in sample_items:
            print(f"  {word}: {emoji}")