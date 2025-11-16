
import textstat,re,nltk
from textblob import TextBlob



def return_semantic_flags(text: str) -> tuple:
    """
    Detect sentiment and profanity (toxicity) in a message.

    Returns tuple:
        (sentiment: str,
         sentiment_score: float,
         toxicity_flag: bool,
         toxicity_score: float)
    """
    from better_profanity import profanity

    # Load profanity words (default)
    profanity.load_censor_words()

    if not text or not text.strip():
        return ("Neutral", 0.0, False, 0.0)

    # --- Sentiment analysis ---
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # --- Toxicity / Profanity check ---
    toxicity_flag = profanity.contains_profanity(text)
    toxicity_score = 1.0 if toxicity_flag else 0.0

    # Round scores to 3 decimals
    return (sentiment, round(sentiment_score, 3), toxicity_flag, round(toxicity_score, 3))

def return_keys(text):
    import emoji
    hashtags = re.findall(r"#\w+", text)


    emojis = [c for c in text if c in emoji.EMOJI_DATA]


    def detect_casing(s: str) -> str:
        if s.isupper():
            return "ALL CAPS"
        elif s.islower():
            return "all lowercase"
        elif s.istitle():
            return "Title Case"
        elif s and s[0].isupper():
            return "Sentence case"
        else:
            return "Mixed/Other"

    casing = detect_casing(text)

    domain_keywords = {"grow", "business", "marketing", "sales"}
    words = re.findall(r"\b\w+\b", text.lower())
    matched_keywords = [w for w in words if w in domain_keywords]

    return hashtags, emojis, casing, matched_keywords

def return_reading_score(text):

    flesch_score = textstat.flesch_reading_ease(text)

    grade_level = textstat.flesch_kincaid_grade(text)
    smog_index = textstat.smog_index(text)
    gunning_fog = textstat.gunning_fog(text)

    sentence_count = textstat.sentence_count(text)
    word_count = textstat.lexicon_count(text)

    return flesch_score, grade_level, smog_index, gunning_fog, sentence_count, word_count


# --- Main function ---
def analyze_with_nltk(text: str):
    """
    Analyze a string with NLTK:
    - Lemmas
    - POS counts
    - Bigrams
    - Keyphrases (Adj+Noun / Noun+Noun)
    """
    from nltk import pos_tag, word_tokenize, bigrams
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from collections import Counter

    # --- Ensure NLTK resources are available ---
    def ensure_nltk_resources():
        resources = {
            "punkt": "tokenizers/punkt",
            "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
            "wordnet": "corpora/wordnet"
        }
        for res, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                # Download quietly
                nltk.download(res, quiet=True)

    ensure_nltk_resources()

    # --- Lemmatizer helper ---
    lemmatizer = WordNetLemmatizer()

    def nltk_pos_to_wordnet(pos):
        """Convert NLTK POS tags to WordNet POS tags for lemmatization."""
        if pos.startswith("J"):
            return wordnet.ADJ
        elif pos.startswith("V"):
            return wordnet.VERB
        elif pos.startswith("N"):
            return wordnet.NOUN
        elif pos.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN  # default fallback
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Lemmatize
    lemmas = [lemmatizer.lemmatize(word, nltk_pos_to_wordnet(pos)) for word, pos in tagged]

    # POS counts
    pos_counts = Counter([pos for _, pos in tagged])

    # Bigrams
    bigrams_list = list(bigrams(lemmas))

    # Keyphrases
    keyphrases = []
    for i in range(len(tagged) - 1):
        pos1, pos2 = tagged[i][1], tagged[i+1][1]
        if (pos1.startswith("JJ") and pos2.startswith("NN")) or (pos1.startswith("NN") and pos2.startswith("NN")):
            keyphrases.append(f"{tagged[i][0]} {tagged[i+1][0]}")

    return lemmas, pos_counts, bigrams_list, keyphrases

#########################----------------------ADVANCED!

def analyze_register(text: str, top_slang=500, ngram_range=(1,2)):
    """
    Analyze the formality, slang, and tone of a sentence.
    Returns: (Formality, Score, Slang, Tone)

    Parameters:
    - top_slang: number of slang terms/phrases to extract
    - ngram_range: tuple for multi-word slang detection (min_n, max_n)
    """

    import subprocess
    import sys
    import contextlib
    # --- Lazy import + auto-install NLTK ---
    try:
        from nltk.corpus import brown, words
        from nltk.util import ngrams
        from collections import Counter
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "nltk"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        from nltk.corpus import brown, words
        from nltk.util import ngrams
        from collections import Counter

    # --- Ensure necessary corpora are downloaded ---
    for corpus in ["brown", "words"]:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                nltk.download(corpus, quiet=True)

    # --- Cache the slang list for speed ---
    if not hasattr(analyze_register, "_cached_slang_list"):
        english_vocab = set(w.lower() for w in words.words())
        brown_words = [w.lower() for w in brown.words()]

        slang_candidates = []

        min_n, max_n = ngram_range
        for n in range(min_n, max_n + 1):
            for gram in ngrams(brown_words, n):
                if any(word not in english_vocab and word.isalpha() for word in gram):
                    slang_candidates.append(" ".join(gram))

        freq = Counter(slang_candidates)
        analyze_register._cached_slang_list = [word for word, count in freq.most_common(top_slang)]

    slang_terms = analyze_register._cached_slang_list

    # --- Tokenize input text ---
    tokens = re.findall(r"\b\w+\b", text.lower())

    # --- Detect slang (single + multi-word) ---
    found_slang = [term for term in slang_terms
                   if (" " in term and term in text.lower()) or term in tokens]

    # --- Formality scoring ---
    formal_markers = ["therefore", "henceforth", "shall", "regarding", "respectfully"]
    informal_markers = slang_terms + ["hey", "hi", "what’s up", "gonna", "wanna", "kinda"]

    score = sum(word in text.lower() for word in formal_markers) - sum(word in text.lower() for word in informal_markers)

    if score > 0:
        formality = "Formal"
    elif score < 0:
        formality = "Informal"
    else:
        formality = "Neutral"

    # --- Tone detection ---
    tone = []
    text_lower = text.lower()
    if re.search(r"(congratulations|amazing|awesome|great|well done)", text_lower):
        tone.append("motivational")
    if re.search(r"(let’s|together|team|collaborate)", text_lower):
        tone.append("collaborative")
    if re.search(r"(hey|hi|hello|thanks|cheers)", text_lower):
        tone.append("friendly")
    if not tone:
        tone.append("neutral")

    return formality, score, found_slang if found_slang else None, tone


def render_check_anomaly_flags(input_file="Output.txt"):
    """
    Reads Output.txt and predicts anomalies/sentiment shifts:
      - Mood/Sentiment differences
      - Day/Week/Month Changes vs averages

    Returns a tuple:
        final_score_percent, mood_score_percent, change_score_percent, anomalies_list, anomaly_block_str
    """
    # Optional TextBlob sentiment analysis
    try:
        from textblob import TextBlob
        HAS_TEXTBLOB = True
    except Exception:
        HAS_TEXTBLOB = False

    def _parse_pairs(s: str):
        """Extract pairs like 'label: +12.3%' into a dict."""
        if not s:
            return {}
        pairs = re.findall(r"([A-Za-z]+)\s*:\s*([+-]?\d+(?:\.\d+)?)%", s)
        return {k.lower(): float(v) for k, v in pairs}

    def _find_block(text: str, header_label: str):
        """Return the text block between the line '|  <header_label>  |' and the next divider '|-----|'."""
        hdr = rf"\|\s*{re.escape(header_label)}\s*\|"
        m = re.search(hdr, text)
        if not m:
            return None
        start = m.end()
        div = r"\|\s*-{10,}\s*\|"
        m2 = re.search(div, text[start:])
        return text[start:(start + m2.start())] if m2 else text[start:]

    anomalies = []
    mood_score = 0.0
    change_score_total = 0.0
    change_count = 0

    # ----------------------------
    # Read file
    # ----------------------------
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        return (0.0, 0.0, 0.0, [f"[DEBUG] Failed to read file: {e}"], "")

    # ----------------------------
    # (1) Mood-based anomaly check
    # ----------------------------
    try:
        mm = re.search(r"\|\s*User\(s\):\s*(.+?)\s*\[To an extent of emotion\]", text)
        am = re.search(r"\|\s*Average:\s*(.+?)\s*\[To an extent of emotion\]", text)
        if not (mm and am):
            mm = re.search(r"User\(s\):\s*(.+?)\s*\[To an extent of emotion\]", text)
            am = re.search(r"Average:\s*(.+?)\s*\[To an extent of emotion\]", text)

        if mm and am:
            user_mood = mm.group(1).strip()
            avg_mood = am.group(1).strip()
            if user_mood.lower() != avg_mood.lower():
                anomalies.append(f"⚠️ Sentiment shift: User mood '{user_mood}' vs Avg mood '{avg_mood}'")
                mood_score += 0.7
            else:
                anomalies.append("✅ User mood consistent with average")
                mood_score += 0.3
        else:
            anomalies.append("[DEBUG] Mood data not found.")
    except Exception as e:
        anomalies.append(f"[DEBUG] Mood parsing failed: {e}")

    if HAS_TEXTBLOB:
        try:
            im = re.search(r"Input \(text\) : \[(.*?)\]", text, re.S)
            if im:
                raw = im.group(1).strip("'\"")
                pol = TextBlob(raw).sentiment.polarity
                if pol <= -0.3:
                    anomalies.append(f"📉 ML Sentiment Alert: Strong negative polarity ({pol:.2f})")
                    mood_score += 0.5
                elif pol >= 0.3:
                    anomalies.append(f"📈 ML Sentiment Alert: Strong positive polarity ({pol:.2f})")
                    mood_score += 0.5
                else:
                    anomalies.append(f"😐 ML Sentiment Stable ({pol:.2f})")
                    mood_score += 0.2
        except Exception as e:
            anomalies.append(f"[DEBUG] TextBlob sentiment failed: {e}")

    # ----------------------------
    # (2) Day/Week/Month Change anomalies
    # ----------------------------
    for timeframe in ("Day", "Week", "Month"):
        block = _find_block(text, f"{timeframe} Changes")
        if not block:
            anomalies.append(f"[DEBUG] No {timeframe} Changes block found.")
            continue

        m_type = re.search(r"Type Changes:\s*([^\n|]+)", block)
        m_int  = re.search(r"Intent Changes:\s*([^\n|]+)", block)
        cur_type   = _parse_pairs(m_type.group(1)) if m_type else {}
        cur_intent = _parse_pairs(m_int.group(1))  if m_int  else {}

        m_avg_type   = re.search(r"Average Type Change:(.*?)(?:Average Intent Change:|\Z)", block, re.S)
        m_avg_intent = re.search(r"Average Intent Change:(.*)", block, re.S)
        avg_type   = _parse_pairs(m_avg_type.group(1))   if m_avg_type   else {}
        avg_intent = _parse_pairs(m_avg_intent.group(1)) if m_avg_intent else {}

        for label, val in cur_type.items():
            if label in avg_type:
                av = avg_type[label]
                diff = abs(val - av)
                if diff > 15:
                    anomalies.append(f"📊 {timeframe}: Type '{label}' change {val}% vs avg {av}% (Δ {diff:.2f}%)")
                    change_score_total += 0.7
                else:
                    anomalies.append(f"✅ {timeframe}: Type '{label}' stable ({val}% vs avg {av}%, Δ {diff:.2f}%)")
                    change_score_total += 0.3
                change_count += 1
            else:
                anomalies.append(f"[DEBUG] {timeframe}: No average for type '{label}'")

        for label, val in cur_intent.items():
            if label in avg_intent:
                av = avg_intent[label]
                diff = abs(val - av)
                if diff > 15:
                    anomalies.append(f"📊 {timeframe}: Intent '{label}' change {val}% vs avg {av}% (Δ {diff:.2f}%)")
                    change_score_total += 0.7
                else:
                    anomalies.append(f"✅ {timeframe}: Intent '{label}' stable ({val}% vs avg {av}%, Δ {diff:.2f}%)")
                    change_score_total += 0.3
                change_count += 1
            else:
                anomalies.append(f"[DEBUG] {timeframe}: No average for intent '{label}'")

    change_score = (change_score_total / change_count) if change_count else 0.0

    # ----------------------------
    # Merge 50/50 & build block
    # ----------------------------
    if mood_score == 0.0 and change_score == 0.0:
        final_score = 0.0
        anomalies = ["[DEBUG] Could not extract anomalies, defaulting to 0%."]
    else:
        final_score = round(((min(mood_score, 1.0) * 0.5) + (min(change_score, 1.0) * 0.5)) * 100, 2)

    anomaly_block = (
        f" 🔎 Final Anomaly Confidence Score: {final_score}%\n"
        f"    - Mood Score Contribution: {round(min(mood_score,1.0)*100,2)}%\n"
        f"    - Change Score Contribution: {round(min(change_score,1.0)*100,2)}%\n"
    )
    for a in anomalies:
        anomaly_block += f"| {a} |\n"

    return (
        final_score,
        round(min(mood_score, 1.0) * 100, 2),
        round(min(change_score, 1.0) * 100, 2),
        anomalies,
        anomaly_block
    )


def speech_act_analysis(text: str) -> tuple:
    """
    Advanced Speech Act Analysis using NLTK.
    Auto-downloads required resources if missing.

    Returns tuple:
        (speech_act: str,
         tense: str,
         mood: str,
         sentence_type: str,
         clause_count: int,
         token_count: int)
    """

    # --- Ensure NLTK resources ---
    resources = {
        "punkt": "tokenizers/punkt",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger"
    }
    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)

    if not text or not text.strip():
        return ("Statement", "Present", "Neutral", "Declarative", 0, 0)

    # Tokenize + POS tagging
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    # --- Detect tense (heuristic) ---
    tense = "Present"
    for word, tag in tags:
        if tag in ["VBD", "VBN"]:   # past tense verbs
            tense = "Past"
            break
        elif tag in ["MD"]:         # modal verb → future/conditional
            tense = "Future"
            break

    # --- Detect act (base rules) ---
    speech_act = "Statement"
    mood = "Neutral"

    if text.strip().endswith("?"):
        speech_act = "Question"
    elif any(w.lower() in ["please", "could", "would", "can"] for w, t in tags):
        speech_act = "Request"
        mood = "Polite"
    elif tags and tags[0][1].startswith("VB"):  # starts with verb = imperative
        speech_act = "Proposal"
        mood = "Directive"
    elif any(w.lower() in ["let's", "shall"] for w, t in tags):
        speech_act = "Encouragement"
        mood = "Inclusive"
    elif any(w.lower() in ["must", "should", "need", "have to"] for w, t in tags):
        speech_act = "Encouragement"
        mood = "Obligatory"

    # --- Sentence type classification ---
    if text.strip().endswith("?"):
        sentence_type = "Interrogative"
    elif text.strip().endswith("!"):
        sentence_type = "Exclamatory"
    elif tags and tags[0][1].startswith("VB"):
        sentence_type = "Imperative"
    else:
        sentence_type = "Declarative"

    # --- Clause estimation ---
    clause_count = len(re.split(r"(,|;| and | but | or )", text)) // 2 + 1

    return (
        speech_act,
        tense,
        mood,
        sentence_type,
        clause_count,
        len(tokens)
    )


def intent_classification(text: str) -> tuple:
    """
    Advanced but efficient intent classification.

    Returns tuple:
        (primary_intent: str, audience: str, polarity: str)
    """

    if not text or not text.strip():
        return ("Statement", "General", "Neutral")

    text_lower = text.lower()

    # --- Detect primary intent using keywords and heuristics ---
    call_to_action_keywords = ["try", "do", "submit", "click", "start", "apply", "follow", "buy", "check"]
    comparison_keywords = ["better", "worse", "than", "vs", "vs.", "compare", "compared"]
    encouragement_keywords = ["well done", "great job", "keep going", "awesome", "congratulations"]
    advice_keywords = ["should", "recommend", "advise", "suggest", "consider"]
    warning_keywords = ["be careful", "watch out", "danger", "warning", "risk"]

    primary_intent = "Statement"  # default

    if text.strip().endswith("?"):
        primary_intent = "Question"
    elif any(word in text_lower for word in call_to_action_keywords):
        primary_intent = "Call_to_Action"
    elif any(word in text_lower for word in comparison_keywords):
        primary_intent = "Comparison"
    elif any(phrase in text_lower for phrase in encouragement_keywords):
        primary_intent = "Encouragement"
    elif any(word in text_lower for word in advice_keywords):
        primary_intent = "Advice"
    elif any(phrase in text_lower for phrase in warning_keywords):
        primary_intent = "Warning"

    # --- Detect audience ---
    if re.search(r"\b(we|our|us)\b", text_lower):
        audience = "Team/Peers"
    elif re.search(r"\byou\b", text_lower):
        audience = "Individual"
    elif re.search(r"\bthey\b", text_lower):
        audience = "Third-party"
    else:
        audience = "General"

    # --- Detect polarity using TextBlob ---
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    if polarity_score > 0.1:
        polarity = "Positive"
    elif polarity_score < -0.1:
        polarity = "Negative"
    else:
        polarity = "Neutral"

    return (primary_intent, audience, polarity)


def speech_act_analysis_advanced(text: str) -> tuple:
    """
    Advanced Speech Act Analysis using NLTK + TextBlob.

    Returns tuple:
        (speech_act_type: str,
         tense: str,
         mood: str,
         clause_count: int,
         sentiment: str)

    Features:
    - Detects multiple speech acts (Request, Proposal, Encouragement, Advice, Warning, Statement, Question)
    - Tense detection (Past, Present, Future)
    - Mood / politeness heuristics
    - Clause counting
    - Sentiment overlay (Positive, Neutral, Negative)

    Auto-downloads required NLTK resources if missing.
    """

    # --- Ensure NLTK resources ---
    resources = {
        "punkt": "tokenizers/punkt",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger"
    }
    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)

    if not text or not text.strip():
        return ("Statement", "Present", "Neutral", 0, "Neutral")

    # Tokenize + POS tagging
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    text_lower = text.lower()

    # --- Detect tense ---
    tense = "Present"
    for word, tag in tags:
        if tag in ["VBD", "VBN"]:
            tense = "Past"
            break
        elif tag in ["MD"]:
            tense = "Future"
            break


    # --- Sentiment analysis ---
    polarity_score = TextBlob(text).sentiment.polarity
    if polarity_score > 0.1:
        sentiment = "Positive"
    elif polarity_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # --- Detect speech act type & mood ---
    speech_act_type = "Statement"
    mood = "Neutral"

    # Requests (polite)
    if any(w in text_lower for w in ["please", "could", "would", "can", "kindly"]):
        speech_act_type = "Request"
        mood = "Polite"
    # Proposals (directive)
    elif tags and tags[0][1].startswith("VB"):
        speech_act_type = "Proposal"
        mood = "Directive"
    # Encouragements
    elif any(w in text_lower for w in ["let's", "shall", "keep going", "well done", "great job", "awesome"]):
        speech_act_type = "Encouragement"
        mood = "Encouraging"
    # Advice
    elif any(w in text_lower for w in ["should", "recommend", "advise", "suggest", "consider"]):
        speech_act_type = "Advice"
        mood = "Directive"
    # Warning
    elif any(w in text_lower for w in ["be careful", "watch out", "danger", "risk", "warning"]):
        speech_act_type = "Warning"
        mood = "Directive"
    # Question
    elif text.strip().endswith("?"):
        speech_act_type = "Question"
        mood = "Neutral"

    return (speech_act_type, tense, mood)









