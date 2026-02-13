import sqlite3
import json
import hashlib

# Create database connection
conn = sqlite3.connect("cache.db", check_same_thread=False)
cursor = conn.cursor()

# ---------------------------
# Create Tables (if not exist)
# ---------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS embedding_cache (
    hash TEXT PRIMARY KEY,
    modality TEXT,
    vector TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS answer_cache (
    hash TEXT PRIMARY KEY,
    response TEXT,
    image_base64 TEXT
)
""")

conn.commit()


# ---------------------------
# Utility: Hash Function
# ---------------------------

def _hash(text: str, modality: str = ""):
    return hashlib.sha256((text + modality).encode()).hexdigest()


# ---------------------------
# Embedding Cache
# ---------------------------

def get_cached_embedding(text: str, modality: str):
    key = _hash(text, modality)

    cursor.execute(
        "SELECT vector FROM embedding_cache WHERE hash=?",
        (key,)
    )
    row = cursor.fetchone()

    if row:
        return json.loads(row[0])
    return None


def save_embedding(text: str, vector, modality: str):
    key = _hash(text, modality)

    cursor.execute(
        "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?)",
        (key, modality, json.dumps(vector))
    )
    conn.commit()


# ---------------------------
# Answer Cache
# ---------------------------

def get_cached_answer(query: str):
    key = _hash(query)

    cursor.execute(
        "SELECT response, image_base64 FROM answer_cache WHERE hash=?",
        (key,)
    )
    row = cursor.fetchone()

    if row:
        return row[0], row[1]
    return None


def save_answer(query: str, response: str, image_base64: str):
    key = _hash(query)

    cursor.execute(
        "INSERT OR REPLACE INTO answer_cache VALUES (?, ?, ?)",
        (key, response, image_base64)
    )
    conn.commit()
