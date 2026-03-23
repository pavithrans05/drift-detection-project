import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = {
    "gas": os.path.join(BASE_DIR, "data", "gas"),
    "intel": os.path.join(BASE_DIR, "data", "intel"),
    "nasa": os.path.join(BASE_DIR, "data", "nasa"),
    "swat": os.path.join(BASE_DIR, "data", "swat"),
}