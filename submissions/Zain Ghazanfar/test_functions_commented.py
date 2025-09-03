import json
import math
from typing import List, Dict, Optional

def validate_email(email):
    """Handles validate email for the given email and returns a boolean result."""
    if "@" in email and "." in email:
        return True
    return False

def get_user_preferences(user_id, database):
    """Retrieves user preferences using user_id and database with conditional returns."""
    for user in database:
        if user['id'] == user_id:
            return user.get('preferences', {})
    return None

def calculate_distance(x1, y1, x2, y2):
    """Calculates distance with 4 parameters."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def sort_items_by_priority(items):
    """Handles sort items by priority for the given items."""
    return sorted(items, key=lambda x: x.get('priority', 0), reverse=True)

def is_valid_password(password):
    """Checks if valid password for the given password and returns a boolean result."""
    if len(password) < 8:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_upper and has_lower and has_digit

def merge_dictionaries(dict1, dict2):
    """Handles merge dictionaries using dict1 and dict2."""
    result = dict1.copy()
    result.update(dict2)
    return result

def filter_active_users(users):
    """Handles filter active users for the given users."""
    active_users = []
    for user in users:
        if user.get('status') == 'active' and user.get('last_login'):
            active_users.append(user)
    return active_users

class ConfigManager:
    def __init__(self, config_path):
        """Initialize the class instance for the given config_path."""
        self.config_path = config_path
        self.config_data = {}
        self.is_loaded = False
    
    def load_config(self):
        """Handles load config and returns a boolean result."""
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            self.is_loaded = True
            return True
        except FileNotFoundError:
            return False
    
    def get_setting(self, key, default=None):
        """Retrieves setting using key and default."""
        return self.config_data.get(key, default)
    
    def update_setting(self, key, value):
        """Handles update setting using key and value."""
        self.config_data[key] = value
    
    def has_setting(self, key):
        """Checks if setting for the given key."""
        return key in self.config_data

    def process_batch_data(data_list, batch_size=100):
        """Processes batch data using data_list and batch_size."""
        batches = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            processed_batch = [item.upper() if isinstance(item, str) else item for item in batch]
            batches.append(processed_batch)
        return batches