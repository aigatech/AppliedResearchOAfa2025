class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Standard Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright Colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

# Helper functions for convenience
def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"

def bold(text):
    return colorize(text, Colors.BOLD)

def green(text):
    return colorize(text, Colors.GREEN)

def red(text):
    return colorize(text, Colors.RED)

def yellow(text):
    return colorize(text, Colors.YELLOW)

def blue(text):
    return colorize(text, Colors.BLUE)

def magenta(text):
    return colorize(text, Colors.MAGENTA)

def cyan(text):
    return colorize(text, Colors.CYAN)
