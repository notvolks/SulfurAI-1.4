from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# ----------------------------
# NEW: Extract top keyword from message text
# ----------------------------
def extract_top_keyword(data, top_n=1):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import defaultdict
    stop_words = set(stopwords.words('english'))
    word_freq = defaultdict(int)

    for text, _, _ in data:
        words = word_tokenize(text.lower())
        for word in words:
            # Filter: alphabetic, not stopword, not too short
            if word.isalpha() and word not in stop_words and len(word) > 2:
                word_freq[word] += 1

    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in most_common[:top_n]]

# ----------------------------
# Step 1: Load and parse the data.txt
# ----------------------------
def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                text, country, confidence = parts
                text = text.strip("[]'\"")
                try:
                    confidence = float(confidence)
                    data.append((text, country.strip(), confidence))
                except ValueError:
                    continue
    return data

# ----------------------------
# Step 2: Determine top 1–3 countries
# ----------------------------
def get_top_countries(data, top_n=3):
    country_counts = Counter([row[1] for row in data])
    top = country_counts.most_common(top_n)
    return [country for country, _ in top]

# ----------------------------
# Step 3: Convert country names to Google Trends codes (ISO Alpha-2)
# ----------------------------
def map_country_to_code(country_name):
    mapping = {
        "United States": "US",
        "United Kingdom": "GB",
        "England": "GB",  # England is not a country code — map to GB
        "Italy": "IT",
        "China": "CN",
        "France": "FR",
        "Germany": "DE",
        "Japan": "JP",
        "Brazil": "BR",
        "Spain": "ES",
        "Netherlands": "NL",
        "Russian Federation": "RU"
    }
    return mapping.get(country_name, '')

# ----------------------------
# Step 4: Fetch Google Trends insights
# ----------------------------
def fetch_trends_for_country(pytrends, keyword, country_code, retries=1, delay=10):
    import pandas as pd
    from pytrends.exceptions import TooManyRequestsError
    import requests
    try:
        pytrends.build_payload([keyword], geo=country_code, timeframe='today 12-m')
        over_time = pytrends.interest_over_time()
        related = pytrends.related_queries().get(keyword, {})
        trend_slope = over_time[keyword].pct_change().mean()
        trend_direction = (
            "upward" if trend_slope > 0.01 else
            "downward" if trend_slope < -0.01 else
            "flat"
        )
        return {
            "trend_direction": trend_direction,
            "top_related": related.get("top", pd.DataFrame()).head(5),
            "rising_related": related.get("rising", pd.DataFrame()).head(5)
        }

    except TooManyRequestsError:

      err_log = ["|-----------------------|","|ERROR_TOO_MANY_REQUESTS|","|-----------------------|"]
      return err_log


    except requests.exceptions.RequestException:

       err_log = ["|-----------------------|", "|ERROR_NO_CONNECTION|", "|-----------------------|"]
       return err_log


    except Exception as e:
      err_log = ["|-----------------------|", f"ERROR_UNKNOWN: {str(e)}", "|-----------------------|"]
      return err_log

# ----------------------------
# Step 5: Combine everything
# ----------------------------
def generate_combined_insight(filepath, keyword=None):
    from pytrends.request import TrendReq
    output_lines = []
    data = load_data(filepath)

    if keyword is None:
        keywords = extract_top_keyword(data)
        if not keywords:
            print("⚠️ No keyword could be extracted. Using default: 'AI' [DEBUG FILE: location_trends_s.py]")
            keyword = "AI"
        else:
            keyword = keywords[0]


    top_countries = get_top_countries(data)
    pytrends = TrendReq()

    output_lines.append(f"|--------------------------------------------------------------|")
    output_lines.append(f"|                                                               |")
    output_lines.append(f"|   Top Countries Trends Based on Userbase: {top_countries}    |")
    output_lines.append(f"|                                                               |")


    for country in top_countries:
        country_code = map_country_to_code(country)
        if not country_code:
            continue

        output_lines.append(f"|--------------------------------------------------------------|")
        output_lines.append(f"|   🌍 Country: {country} ({country_code})")
        output_lines.append(f"|--------------------------------------------------------------|")
        output_lines.append(f"|                                                               |")
        trends = fetch_trends_for_country(pytrends, keyword, country_code)
        if isinstance(trends, list) and any("ERROR_" in line for line in trends):
            return "\n".join(trends)

        if isinstance(trends, str) and trends.startswith("ERROR_"):
            return trends  # Return the error code string

        if not trends:
            return "ERROR_UNKNOWN"

        if "error" in trends:
            print(f"   ⚠️ Error fetching trends: {trends['error']}  [DEBUG FILE: location_trends_s.py]")
            continue

        output_lines.append(f"|   📈 Trend Direction: {trends['trend_direction']}")
        output_lines.append("|   🔍 Top Related Searches:")
        if not trends['top_related'].empty:
            for idx, row in trends['top_related'].iterrows():
                output_lines.append(f"|      {row['query']} ({row['value']})")
        else:
            output_lines.append("|     (No top related queries found)")

        output_lines.append("|   🚀 Rising Related Searches:")
        if not trends['rising_related'].empty:
            for idx, row in trends['rising_related'].iterrows():
                output_lines.append(f"|      {row['query']} ({row['value']})")
        else:
            output_lines.append("|     (No rising queries found)")

    output_lines.append(f"\n|                                                               |")
    output_lines.append(f"\n|--------------------------------------------------------------|")

    return "\n".join(output_lines)


# ----------------------------
# Run the whole process
# ----------------------------
def run_script():
    def get_call_file_path():
        from extra_models.Sulfur.TrainingScript.Build import call_file_path
        return call_file_path.Call()


    call = get_call_file_path()
    training_data_location = call.training_data_location()
    data = load_data(training_data_location)
    keyword_list = extract_top_keyword(data)
    keyword = keyword_list[0] if keyword_list else "AI"
    return generate_combined_insight(training_data_location, keyword=keyword)

