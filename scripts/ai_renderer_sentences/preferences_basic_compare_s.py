
from scripts.ai_renderer_sentences import error
from extra_models.Sulfur.TrainingScript.Build import call_file_path
from setup.verification.input_text import txt_data

import os
import random
import re

class prefer_compare():
    def __init__(self):
        self.call = call_file_path.Call()
        self.folder_path_training_data_sk, self.folder_path_training_data_name_sk, self.file_path_training_data_sk = self.call.training_data_sk()

        current_dir_i = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.folder_path_input = os.path.join(current_dir_i, 'data','training_data')
        self.file_name_input = 'Input.txt'  # Debugger variables
        self.file_name_attributes = "Attributes.txt"
        self.file_name_output = "Output.txt"
        self.file_path_input = os.path.join(self.folder_path_input, self.file_name_input)
        self.file_path_attributes = os.path.join(self.folder_path_input, self.file_name_attributes)
        self.file_path_output = os.path.join(self.folder_path_input, self.file_name_output)

    def read_preferences_compare_user(self, amount):
        import nltk
        from collections import Counter
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from packaging import version

        current_version = version.parse(nltk.__version__)

        if current_version < version.parse("3.8.2"):
            print("#################********DEBUG_EXCEPTION**********Your version of NLTK -> PUNKT is outdated as NLTK is under version 3.8.2. SulfurAI may have security issues as such. Upgrade NLTK now!")
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        else:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find("nltk.data.find('taggers/averaged_perceptron_tagger')")
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

        try:
            nltk.data.find("nltk.data.find('taggers/averaged_perceptron_tagger_eng')")
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("VADER lexicon not found. Downloading now...")
            nltk.download('vader_lexicon')

        preferences = []
        preferences_weight = []
        input_data, too_long, re_was_subbed = txt_data.verify_input("list")
        with open(self.file_path_training_data_sk, 'r', encoding='utf-8', errors='ignore') as file:
            training_data_sk = [line.strip().replace('\r', '').lower() for line in file.readlines()]
        string_input_data = str(input_data)
        input_data = [word for line in input_data for word in line.split()]
        words = word_tokenize(string_input_data.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        word_frequencies = Counter(filtered_words)
        most_common = word_frequencies.most_common(amount)
        most_common_words = {word: freq for word, freq in most_common}

        for line in input_data:
            index_weight = 0
            occurrences = [(i, train_line) for i, train_line in enumerate(training_data_sk)]
            if occurrences:
                for index, matching_line in occurrences:
                    if f"{line} " in matching_line:
                        if line.lower() in most_common_words:
                            index_weight += float(index) - int(len(line)) + len(matching_line)
                        else:
                            index_weight += float(index) - int(len(line))
                    else:
                        if line.lower() in most_common_words:
                            index_weight += float(index) - int(len(line)) + len(matching_line)
                        else:
                            if len(matching_line) != 0:
                                index_weight += (float(index) - int(len(line))) / len(matching_line)
                            else:
                                index_weight += (float(index) - int(len(line))) / (len(matching_line) + 1)
            preferences_weight.append(line)
            preferences_weight.append(index_weight)

        global preferences_user_weight_list
        preferences_user_weight_list = preferences_weight
        numeric_values = [value for value in preferences_weight if isinstance(value, (int, float))]
        three_highest = sorted(numeric_values, reverse=True)[:amount]
        for item in three_highest:
            index = preferences_weight.index(item)
            preferences.append(preferences_weight[index - 1])

        ############attributes

        preferences_attribute = []

        with open(self.file_path_attributes, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            for line in lines:
                if 'PREFERENCES_USER :' in line:
                    items = line.split(':', 1)[1].strip()

                    preferences_attribute = [item.strip() for item in items.split(',') if item.strip()]
                    break

        if not preferences_attribute:
            preferences_attribute = []
        else:
            if len(preferences_attribute) != amount:
                print(
                    f"Warning: Not enough preferences in the Attributes. Results may vary. Required amount: {amount}")

        if preferences_attribute:
            preferences = preferences_attribute

        file_path_OutputData_name_preferences_user = self.call.preferences_user()
        with open(file_path_OutputData_name_preferences_user, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(', '.join(preferences))

        return preferences

    def read_preferences_compare_global(self, amount):
        import nltk
        from collections import Counter
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords

        try:
            preferences = []
            preferences_weight = []
            input_data, too_long, re_was_subbed = txt_data.verify_input("list")
            with open(self.file_path_training_data_sk, 'r', encoding='utf-8', errors='ignore') as file:
                training_data_sk = [line.strip().replace('\r', '').lower() for line in file.readlines()]
            string_input_data = str(training_data_sk)
            input_data = [word for line in training_data_sk for word in line.split()]
            words = word_tokenize(string_input_data.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            word_frequencies = Counter(filtered_words)
            most_common = word_frequencies.most_common(amount)
            most_common_words = {word: freq for word, freq in most_common}

            for line in input_data:
                index_weight = 0
                occurrences = [(i, train_line) for i, train_line in enumerate(training_data_sk)]
                if occurrences:
                    for index, matching_line in occurrences:
                        if f"{line} " in matching_line:
                            if line.lower() in most_common_words:
                                index_weight += float(index) - int(len(line)) + len(matching_line)
                            else:
                                index_weight += float(index) - int(len(line))
                        else:
                            if line.lower() in most_common_words:
                                index_weight += float(index) - int(len(line)) + len(matching_line)
                            else:
                                if len(matching_line) != 0:
                                    index_weight += (float(index) - int(len(line))) / len(matching_line)
                                else:
                                    index_weight += (float(index) - int(len(line))) / (len(matching_line) + 1)
                preferences_weight.append(line)
                preferences_weight.append(index_weight)
            numeric_values = [value for value in preferences_weight if isinstance(value, (int, float))]
            three_highest = sorted(numeric_values, reverse=True)[:amount]
            for item in three_highest:
                index = preferences_weight.index(item)
                preferences.append(preferences_weight[index - 1])

            import pandas as pd
            df = pd.Series(preferences)
            preferences = df.drop_duplicates().reset_index(drop=True).tolist()

            while len(preferences) < amount:
                most_common = word_frequencies.most_common()
                found_words = set()

                for word, freq in most_common:
                    if word not in preferences and len(preferences) < amount:
                        preferences.append(word)
                        found_words.add(word)

                if not found_words:
                    break

            preferences = preferences[:amount]

            file_path_OutputData_name_preferences_global = self.call.preferences_global()
            with open(file_path_OutputData_name_preferences_global, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(', '.join(preferences))

            return preferences
        except (FileNotFoundError, UnicodeDecodeError, IndexError, TypeError, ValueError):
            error("er1", "AI_RENDERER_FILE (2_GLOBAL)", " PREFERENCES_BASIC_COMPARE_S.PY", "11")

    #######################verbs
    def get_verb_want(self, training_data_sk, preferences_user, preferences_text_global, user_or_global):
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        from collections import Counter

        preferences_text = ""
        if user_or_global == "user":
            preferences_text = str(preferences_user)
        elif user_or_global == "global":
            preferences_text = re.sub(r'[^a-zA-Z\s]', '', preferences_text_global)
        tokens = word_tokenize(preferences_text.lower())
        pos_tags = pos_tag(tokens)
        verbs = [word for word, tag in pos_tags if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
        cleaned_verbs = [pref.strip(" '[]") for pref in verbs if pref.strip(" '[]")]
        word_counts = Counter(verbs)

        def user():
            word_counts = Counter(word.lower() for word in word_tokenize(str(training_data_sk)))
            if cleaned_verbs:
                wanted_verb_most_important_user = max(cleaned_verbs, key=lambda word: word_counts.get(word, 0))
            else:
                wanted_verb_most_important_user = random.choice(cleaned_verbs) if cleaned_verbs else "None_Found"
            file_path_OutputData_Wanted_verb_most_important_user = self.call.Wanted_verb_most_important_user()
            with open(file_path_OutputData_Wanted_verb_most_important_user, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(wanted_verb_most_important_user)
            return wanted_verb_most_important_user

        def Global():
            word_counts = Counter(verbs)
            if word_counts:
                wanted_verb_most_important_global = max(word_counts, key=word_counts.get)
            else:
                wanted_verb_most_important_global = random.choice(cleaned_verbs) if word_counts else "None_Found"

            file_path_OutputData_Wanted_verb_most_important_global = self.call.Wanted_verb_most_important_global()
            with open(file_path_OutputData_Wanted_verb_most_important_global, 'w', encoding='utf-8',
                      errors='ignore') as file:
                file.write(wanted_verb_most_important_global)
            return wanted_verb_most_important_global

        if user_or_global == "user":
            wanted_verb_most_important_user = user()
            return wanted_verb_most_important_user
        elif user_or_global == "global":
            wanted_verb_most_important_global = Global()
            return wanted_verb_most_important_global

    #######################adjectives

    def get_adjective_describe(self, training_data_sk, preferences_user, preferences_text_global, user_or_global):
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        from collections import Counter

        preferences_text = ""
        if user_or_global == "user":
            preferences_text = str(preferences_user)
        elif user_or_global == "global":
            preferences_text = re.sub(r'[^a-zA-Z\s]', '', preferences_text_global)
        tokens = word_tokenize(preferences_text.lower())
        pos_tags = pos_tag(tokens)
        adjectives = [word for word, tag in pos_tags if tag in ["JJ", "JJR", "JJS"]]
        cleaned_adjectives = [pref.strip(" '[]") for pref in adjectives if pref.strip(" '[]")]
        word_counts = Counter(adjectives)

        def user():
            word_counts = Counter(word.lower() for word in word_tokenize(str(training_data_sk)))
            if cleaned_adjectives:
                adjective_describe_user = max(cleaned_adjectives, key=lambda word: word_counts.get(word, 0))
            else:
                adjective_describe_user = random.choice(cleaned_adjectives) if cleaned_adjectives else "None_Found"

            file_path_OutputData_Describe_adjective_user = self.call.Describe_adjective_most_important_user()
            with open(file_path_OutputData_Describe_adjective_user, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(adjective_describe_user)

            return adjective_describe_user

        def Global():
            word_counts = Counter(adjectives)
            if word_counts:
                adjective_describe_global = max(word_counts, key=word_counts.get)
            else:
                adjective_describe_global = random.choice(cleaned_adjectives) if word_counts else "None_Found"
            file_path_OutputData_Describe_adjective_global = self.call.Describe_adjective_most_important_global()
            with open(file_path_OutputData_Describe_adjective_global, 'w', encoding='utf-8',
                      errors='ignore') as file:
                file.write(adjective_describe_global)
            return adjective_describe_global

        if user_or_global == "user":
            adjective_describe_user = user()
            return adjective_describe_user
        elif user_or_global == "global":
            adjective_describe_global = Global()
            return adjective_describe_global

    ##################MOOD

    def get_mood(self, file_path_input, file_path_training_data_sk,
                 file_path_OutputData_mood_accuracy_user, file_path_OutputData_mood_accuracy_global):
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            with open(file_path_input, 'r', encoding='utf-8', errors='ignore') as file:
                input_data_user = file.readlines()
            with open(file_path_training_data_sk, 'r', encoding='utf-8', errors='ignore') as file:
                input_data_global = file.readlines()

            mood_user = "Not_found"
            mood_global = "Not_found"
            mood_accuracy_user = 0
            mood_accuracy_global = 0

            if str(input_data_user).strip() and str(input_data_global).strip():
                sia = SentimentIntensityAnalyzer()

                sentiment_scores_user = sia.polarity_scores(str(input_data_user))
                sentiment_scores_global = sia.polarity_scores(str(input_data_global))

                def determine_mood(scores):
                    if scores["compound"] >= 0.05:
                        return "Positive"
                    elif scores["compound"] <= -0.05:
                        return "Negative"

                    else:
                        non_compound = {k: v for k, v in scores.items() if k != 'compound'}
                        if non_compound:
                            highest = max(non_compound, key=non_compound.get)
                        else:
                            highest = "Not_found"

                        if highest == "neu":
                            return "Neutral"
                        if highest == "neg":
                            return "Negative"
                        if highest == "pos":
                            return "Positive"
                        return highest

                mood_user = determine_mood(sentiment_scores_user)
                mood_global = determine_mood(sentiment_scores_global)

                def compute_accuracy(scores):
                    mood_keys = {'pos', 'neg', 'neu'}
                    filtered_scores = {k: scores[k] for k in mood_keys if k in scores}

                    if not filtered_scores:
                        return 0.0
                    sorted_vals = sorted(filtered_scores.values(), reverse=True)

                    if len(sorted_vals) == 1:
                        return round(min(sorted_vals[0] * 100, 99.9), 2)
                    top, second = sorted_vals[0], sorted_vals[1]
                    confidence = (top - second) * 100

                    return round(min(confidence, 99.9), 2)

                mood_accuracy_user = compute_accuracy(sentiment_scores_user)
                mood_accuracy_global = compute_accuracy(sentiment_scores_global)

            else:
                mood_user = "Not_found"
                mood_global = "Not_found"
                mood_accuracy_user = 0
                mood_accuracy_global = 0

            file_path_OutputData_mood_user = self.call.mood_user()
            file_path_OutputData_mood_global = self.call.mood_global()

            with open(file_path_OutputData_mood_user, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(str(mood_user))
            with open(file_path_OutputData_mood_global, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(str(mood_global))

            with open(file_path_OutputData_mood_accuracy_user, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(str(mood_accuracy_user))
            with open(file_path_OutputData_mood_accuracy_global, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(str(mood_accuracy_global))

            return mood_user, mood_accuracy_user, mood_global, mood_accuracy_global

        except (LookupError, TypeError, UnicodeError, UnicodeDecodeError, FileNotFoundError) as e:
            error("er1", "GET_MOOD_FUNCTION", "PREFERENCES_BASIC_COMPARE_S.PY", "11")
            print(f"Exception: {e}")
            return "Not Found", "Not Found", "Not Found", "Not Found"

    ##################SCRIPT RUNNER

    def get_process(self, amount):
        def get_path_directories():
            input_data, too_long, re_was_subbed = txt_data.verify_input("list")

            with open(self.file_path_training_data_sk, 'r', encoding='utf-8', errors='ignore') as file:
                training_data_sk = [line.strip().replace('\r', '').lower() for line in file.readlines()]
            return input_data, training_data_sk

        input_data, training_data_sk = get_path_directories()

        def wanted_nouns():
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag
            from collections import Counter

            #################user

            preferences_user = self.read_preferences_compare_user(amount)
            preferences_text = str(preferences_user)
            tokens = word_tokenize(preferences_text.lower())
            pos_tags = pos_tag(tokens)
            nouns = [word for word, tag in pos_tags if tag in ["NN", "NNS", "NNP", "NNPS"]]
            cleaned_nouns = [pref.strip(" '[]") for pref in nouns if pref.strip(" '[]")]
            word_counts = Counter(nouns)

            word_counts = Counter(word.lower() for word in word_tokenize(str(training_data_sk)))
            if cleaned_nouns:
                wanted_noun_most_important_user = max(cleaned_nouns, key=lambda word: word_counts.get(word, 0))
            else:
                wanted_noun_most_important_user = random.choice(cleaned_nouns) if cleaned_nouns else "None_Found"
            file_path_OutputData_Wanted_noun_most_important_user = self.call.Wanted_noun_most_important_user()
            with open(file_path_OutputData_Wanted_noun_most_important_user, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(wanted_noun_most_important_user)

            #################global
            preferences_text_global = ' '.join(self.read_preferences_compare_global(amount))
            preferences_text = re.sub(r'[^a-zA-Z\s]', '', preferences_text_global)
            tokens = word_tokenize(preferences_text.lower())
            pos_tags = pos_tag(tokens)
            nouns = [word for word, tag in pos_tags if tag in ["NN", "NNS", "NNP", "NNPS"]]
            cleaned_nouns = [pref.strip(" '[]") for pref in nouns if pref.strip(" '[]")]

            word_counts = Counter(nouns)
            if word_counts:
                wanted_noun_most_important_global = max(word_counts, key=word_counts.get)
            else:
                wanted_noun_most_important_global = random.choice(cleaned_nouns) if word_counts else "None_Found"

            file_path_OutputData_Wanted_noun_most_important_global = self.call.Wanted_noun_most_important_global()
            with open(file_path_OutputData_Wanted_noun_most_important_global, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(wanted_noun_most_important_global)

            return wanted_noun_most_important_user, wanted_noun_most_important_global, preferences_user, preferences_text_global

        wanted_noun_most_important_user, wanted_noun_most_important_global, preferences_user, preferences_text_global = wanted_nouns()

        ########mood
        def run_mood():
            file_path_OutputData_mood_accuracy_user = self.call.mood_accuracy_user()
            file_path_OutputData_mood_accuracy_global = self.call.mood_accuracy_global()
            mood_user, mood_accuracy_user, mood_global, mood_accuracy_global = self.get_mood(self.file_path_input, self.file_path_training_data_sk, file_path_OutputData_mood_accuracy_user, file_path_OutputData_mood_accuracy_global)
            average_mood_accuracy = round((mood_accuracy_user + mood_accuracy_global) / 2, 2)
            return mood_user, mood_accuracy_user, mood_global, mood_accuracy_global, average_mood_accuracy

        adjective_describe_global = self.get_adjective_describe(training_data_sk, preferences_user, preferences_text_global, "global")
        adjective_describe_user = self.get_adjective_describe(training_data_sk, preferences_user, preferences_text_global, "user")

        wanted_verb_most_important_global = self.get_verb_want(training_data_sk, preferences_user, preferences_text_global, "global")
        wanted_verb_most_important_user = self.get_verb_want(training_data_sk, preferences_user, preferences_text_global, "user")

        mood_user, mood_accuracy_user, mood_global, mood_accuracy_global, average_mood_accuracy = run_mood()

        return wanted_noun_most_important_user, wanted_noun_most_important_global, preferences_user, preferences_text_global, wanted_verb_most_important_user, wanted_verb_most_important_global, input_data, adjective_describe_user, adjective_describe_global, mood_user, mood_global, mood_accuracy_user, mood_accuracy_global, average_mood_accuracy

