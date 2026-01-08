import re
from textblob import TextBlob
from collections import Counter

def preprocess_chat(raw_chat: str, stopword_threshold=10, short_threshold=3):
    """
    Preprocesses and annotates a chat string for lightweight analytical features.
    Keeps punctuation, removes PII, optionally removes stopwords,
    and calculates basic structural/textual metrics.
    """

    # -------------------------------
    # 1. Remove emails & phone numbers
    # -------------------------------
    clean_text = re.sub(r'\S+@\S+', '', raw_chat)
    clean_text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '', clean_text)

    # -------------------------------
    # 2. Normalize whitespace
    # -------------------------------
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # -------------------------------
    # 3. Remove timestamps like [HH:MM] or (HH:MM:SS)
    # -------------------------------
    clean_text = re.sub(r'\[?\(?\d{1,2}:\d{2}(:\d{2})?\)?\]?', '', clean_text)

    # -------------------------------
    # 4. Tokenize (keep punctuation)
    # -------------------------------
    tokens = clean_text.split()

    # -------------------------------
    # 5. Conditional stopword removal
    # -------------------------------
    stopwords = {"the", "a", "is", "to", "and", "in", "of", "it"}
    if len(tokens) >= stopword_threshold:
        tokens_filtered = [t for t in tokens if t.lower() not in stopwords]
    else:
        tokens_filtered = tokens

    # -------------------------------
    # 6. Sentiment analysis
    # -------------------------------
    sentiment_score = TextBlob(clean_text).sentiment.polarity

    # -------------------------------
    # 7. Keyword frequency
    # -------------------------------
    keyword_freq = Counter(tokens_filtered)

    # -------------------------------
    # 8. Lightweight text metrics
    # -------------------------------
    num_words = len(tokens)
    num_chars = len(clean_text)
    avg_word_length = num_chars / num_words if num_words > 0 else 0

    num_exclamations = clean_text.count('!')
    num_questions = clean_text.count('?')
    num_periods = clean_text.count('.')

    uppercase_words = sum(1 for t in tokens if t.isupper())
    percent_uppercase = uppercase_words / num_words if num_words > 0 else 0

    # -------------------------------
    # 9. Annotator flags
    # -------------------------------
    is_too_short = num_chars < short_threshold
    annotations = {
        "is_too_short": is_too_short
    }

    # -------------------------------
    # 10. Return everything
    # -------------------------------
    return {
        "clean_text": clean_text,
        "tokens": tokens_filtered,
        "keyword_frequency": dict(keyword_freq),
        "num_words": num_words,
        "num_chars": num_chars,
        "avg_word_length": avg_word_length,
        "num_exclamations": num_exclamations,
        "num_questions": num_questions,
        "num_periods": num_periods,
        "percent_uppercase": percent_uppercase,
        "annotations": annotations
    }

