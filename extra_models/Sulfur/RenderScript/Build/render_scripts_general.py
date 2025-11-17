from datetime import datetime


def _get_call_file_path():
    """
     Retrieves the callable file path from the build script.

     This function imports the `call_file_path` module and executes
     its `Call()` method to return an object used to access specific file paths
     within the SulfurAI framework.

     Returns:
         object: A callable object containing methods to fetch important file paths.
     """
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()
call = _get_call_file_path()

def _dataprofiles_write_output(output_file_path):
    import os
    current_dir_i = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # up from Build to TrainingScript
        os.pardir,  # up to Sulfur
        os.pardir,  # up to VersionFiles
        os.pardir,
    ))
    folder_path_output_dataprofiles = os.path.join(current_dir_i, 'RETURNS', 'model_output')
    folder_path_output_dataprofiles = os.path.join(folder_path_output_dataprofiles, "output.txt")
    with open(output_file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    with open(folder_path_output_dataprofiles, "w", encoding="utf-8", errors="ignore") as file:
        file.writelines(lines)




def _rest_of_the_script(tag_trainer,return_statements,start_time,output_file_path,add_to_training_data=True,is_main=False,endpoint_custom=False):
    """
    The main logic pipeline for SulfurAI.
    Handles device classification, training data updates, sentiment analysis, and UI output.

    Args:
        tag_trainer (str): Identifier tag for training session.
        return_statements (bool): If True, returns a dictionary of outputs.
        add_to_training_data (bool): If True, adds the session input to training data.

    Returns:
        dict (optional): Summary results from the AI process (if `return_statements` is True).
    """
    # Write output

    import os
    from scripts.ai_renderer_sentences import Mean_device_s
    from scripts.ai_renderer_sentences import preferences_basic_compare_s
    from scripts.ai_renderer_2 import sentence_detectAndInfer_s
    from scripts.ai_renderer_2 import sentence_detectAndCompare_s
    from scripts.ai_renderer_sentences.sentence_location_build import location_detect_s
    from scripts.ai_renderer_sentences.sentence_location_build import location_trends_s
    from setup.verification.input_text import txt_data
    from extra_models.Sulfur.RenderScript.Build import finish_render
    from extra_models.Sulfur.RenderScript.Build import ui_add_output_data
    from extra_models.Sulfur.RenderScript.Build import call_ai
    from extra_models.Sulfur.Models.base_sulfur_drl_build.filter import filter_text


    #----------------Declare Globals
    # python
    # In your _rest_of_the_script function, update globals to include the new variables:
    global input_data, too_long, re_was_subbed, OutputDevice, Device_Accuracy, main_devices, average_accuracy, \
        mood_user, mood_global, mood_accuracy_user, mood_accuracy_global, average_mood_accuracy, \
        stype_user, sintent_user, acc_sent_user, acc_intent_user, avg_sent_types, avg_intent_types, \
        acc_sent_global, acc_intent_global, avg_accuracy_global, country, confidence, country_global, confidence_global, \
        hours, minutes, seconds, total_time_ms, primary_opp, subsidiary_opp, acc_opp_user, model, advanced_model_debug, \
        speech_act, speech_act_type, tense, mood, mood_2, sentence_type, clause_count, tokens, formality, score, \
        found_slang, tone, primary_intent, audience, polarity, final_score, mood_score, change_score, anomalies, anomaly_block, \
        keyword_frequency, percent_uppercase, avg_word_length, num_exclamations, num_words, num_chars, num_questions, num_periods, \
        hashtags, emojis, casing, matched_keywords, count_user_sessions, flesch_score, grade_level, smog_index, gunning_fog, \
        lemmas, pos_counts, bigrams_list, keyphrases, sentiment, sentiment_score, toxicity_flag, toxicity_score, sentence_count, word_count

    model = None
    advanced_model_debug = None
    #--------------------------------

    Check_device_s = call_ai._call_ai_class("CD", is_main=is_main)
    instance = Check_device_s["Ai"]()
    Check_device_s["is_grandcaller_main"](is_main)
    ai_process_cd_instance = Check_device_s["ai_process_cd"]()
    Device_Result, Device_Accuracy = ai_process_cd_instance.process_script(add_to_training_data)
    try:
        # === Load Input and Preferences ===
        preferences_input = []
        current_dir_i = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            os.pardir,  # up from Build to TrainingScript
            os.pardir,  # up to Sulfur
            os.pardir,  # up to VersionFiles
            os.pardir,
        ))
        folder_path_input = os.path.join(current_dir_i, 'data', 'training_data')
        file_name_input = 'Input.txt'
        file_name_attributes = "Attributes.txt"
        file_name_output = "Output.txt"

        file_path_input = os.path.join(folder_path_input, file_name_input)
        file_path_attributes = os.path.join(folder_path_input, file_name_attributes)

        if output_file_path == "[]":    file_path_output = os.path.join(folder_path_input, file_name_output)
        else:   file_path_output = output_file_path




        pass
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OPTIMISE
        instance_preferences = preferences_basic_compare_s.prefer_compare()
        (
            wanted_noun_most_important_user, wanted_noun_most_important_global, preferences_user,
            preferences_text_global, wanted_verb_most_important_user, wanted_verb_most_important_global,
            input_data, adjective_describe_user, adjective_describe_global, mood_user, mood_global,
            mood_accuracy_user, mood_accuracy_global, average_mood_accuracy
        ) = instance_preferences.get_process(3)
        #print("1") #debug



        pass
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OPTIMISE

        (
            stype_user, sintent_user, primary_opp, subsidiary_opp,
            acc_sent_user, acc_intent_user, acc_opp_user,
            avg_sent_types, avg_intent_types,
            acc_sent_global, acc_intent_global,
            avg_accuracy_global
        ) = sentence_detectAndInfer_s.sentence_intent_and_infer(add_to_training_data)






        # === Read Time-Based Change Settings ===
        file_path_settings_name_ui_days_ago = call.settings_ui_days_ago()
        file_path_settings_name_ui_days_apart = call.settings_ui_days_apart()
        file_path_settings_name_ui_weeks_ago = call.settings_ui_weeks_ago()
        file_path_settings_name_ui_weeks_apart = call.settings_ui_weeks_apart()
        file_path_settings_name_ui_months_ago = call.settings_ui_months_ago()
        file_path_settings_name_ui_months_apart = call.settings_ui_months_apart()
        file_path_settings_name_ui_years_ago = call.settings_ui_years_ago()
        file_path_settings_name_ui_years_apart = call.settings_ui_years_apart()

        with open(file_path_settings_name_ui_days_ago, "r", encoding="utf-8", errors="ignore") as f:
            past_d_changes = int(f.readline())
        with open(file_path_settings_name_ui_days_apart, "r", encoding="utf-8", errors="ignore") as f:
            changes_d_apart_at_leastDays = int(f.readline())

        with open(file_path_settings_name_ui_weeks_ago, "r", encoding="utf-8", errors="ignore") as f:
            past_w_changes = int(f.readline())
        with open(file_path_settings_name_ui_weeks_apart, "r", encoding="utf-8", errors="ignore") as f:
            changes_w_apart_at_leastWeek = int(f.readline())

        with open(file_path_settings_name_ui_months_ago, "r", encoding="utf-8", errors="ignore") as f:
            past_m_changes = int(f.readline())
        with open(file_path_settings_name_ui_months_apart, "r", encoding="utf-8", errors="ignore") as f:
            changes_m_apart_at_leastMonth = int(f.readline())

        with open(file_path_settings_name_ui_years_ago, "r", encoding="utf-8", errors="ignore") as f:
            past_y_changes = int(f.readline())
        with open(file_path_settings_name_ui_years_apart, "r", encoding="utf-8", errors="ignore") as f:
            changes_y_apart_at_leastYear = int(f.readline())


        changes_summary_day, average_change_d = sentence_detectAndCompare_s.run_model(past_d_changes,
                                                                                      changes_d_apart_at_leastDays,
                                                                                      "day", )
        changes_summary_week, average_change_w = sentence_detectAndCompare_s.run_model(past_w_changes,
                                                                                       changes_w_apart_at_leastWeek,
                                                                                       "week", )
        changes_summary_month, average_change_m = sentence_detectAndCompare_s.run_model(past_m_changes,
                                                                                        changes_m_apart_at_leastMonth,
                                                                                        "month", )
        changes_summary_year, average_change_y = sentence_detectAndCompare_s.run_model(past_y_changes,
                                                                                       changes_y_apart_at_leastYear,
                                                                                       "year", )



        # === UI Output Data ===
        ui_add_output_data._ui_add_output_data(call.ui_day_changes(), past_d_changes, changes_summary_day, average_change_d, changes_d_apart_at_leastDays, "Day", "days")
        ui_add_output_data._ui_add_output_data(call.ui_week_changes(), past_w_changes, changes_summary_week, average_change_w, changes_w_apart_at_leastWeek, "Week", "weeks")
        ui_add_output_data._ui_add_output_data(call.ui_month_changes(), past_m_changes, changes_summary_month, average_change_m, changes_m_apart_at_leastMonth, "Month", "months")
        ui_add_output_data._ui_add_output_data(call.ui_year_changes(), past_y_changes, changes_summary_year, average_change_y, changes_y_apart_at_leastYear, "Year", "years")

        # === AI Device and Input Verification ===
        OutputDevice = instance.summarise_device()
        main_devices = Mean_device_s.get_main_device()
        average_accuracy = Mean_device_s.get_main_accuracy()
        input_data, too_long, re_was_subbed = txt_data.verify_input("list")


        # === UI predicted location ===
        country, confidence,country_global, confidence_global = location_detect_s.predict_location(input_data,tag_trainer,add_to_training_data)
        country_trends_list =  location_trends_s.run_script()


        # === Output Writer Function ===

        max_lines = 12 #add settings for

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def write_userbase_changes(file, label, past_range, summary, avg, apart_unit, bypass_limit):

            def truncate_lines(text):
                if not text:
                    return "None_Found"
                lines = text.splitlines()
                if len(lines) > max_lines:
                    return '\n'.join(lines[:max_lines]) + f"\n...and {len(lines) - max_lines} more lines. Removed due to limit cap."
                return text

            file.write("|--------------------------------------------|\n")
            file.write(f"|              {label} Changes             |\n")
            file.write(f"|  Changes to your userbase over the past {past_range} {label.lower()}s:\n")
            if bypass_limit:
                file.write("  " + (summary if summary else "None_Found") + "\n")
            else:
                file.write("  " + truncate_lines(summary) + "\n")

            file.write(f"|  Average Changes over the past {past_range} {label.lower()}s:\n")

            if bypass_limit:
                file.write("  " + (avg if avg else "None_Found") + "\n")
            else:
                file.write("  " + truncate_lines(avg) + "\n")

            file.write(f"|   *Only includes changes at least {apart_unit} {label.lower()}s apart.\n")
            file.write("|--------------------------------------------|\n")
            file.write("\n")


        # === Write Output to File ===
        from returns.dataprofiles.scripts import run_profile_builder
        from extra_models.Sulfur.GeneralScripts.LocalUserEvents import events_hoster
        from extra_models.Sulfur.GeneralScripts.LocalUserEvents import general_model_debug
        recent_events = events_hoster.read_events()
        run_profile_builder.count_build_file_cache_permanent()
        count_user_sessions = run_profile_builder.read_count_cache()
        events_to_script_percentage = events_hoster.calculate_events_average()
        hashtags, emojis, casing, matched_keywords = general_model_debug.return_keys(str(input_data).strip("[]'"))
        flesch_score, grade_level, smog_index, gunning_fog, sentence_count, word_count = general_model_debug.return_reading_score(str(input_data).strip("[]'"))
        lemmas, pos_counts, bigrams_list, keyphrases = general_model_debug.analyze_with_nltk(str(input_data).strip("[]'"))
        formality, score, found_slang, tone = general_model_debug.analyze_register(str(input_data).strip("[]'"))
        sentiment, sentiment_score, toxicity_flag,toxicity_score = general_model_debug.return_semantic_flags(str(input_data).strip("[]'"))
        primary_intent, audience, polarity = general_model_debug.intent_classification(str(input_data).strip("[]'"))
        speech_act_type, tense, mood_2 = general_model_debug.speech_act_analysis_advanced(str(input_data).strip("[]'"))

        (
            final_score,
            mood_score,
            change_score,
            anomalies,
            anomaly_block
        ) = general_model_debug.render_check_anomaly_flags(input_file=file_path_output)

        (
            speech_act,
            tense,
            mood,
            sentence_type,
            clause_count,
            tokens
        )  = general_model_debug.speech_act_analysis(str(input_data).strip("[]'"))

        results = filter_text.preprocess_chat(str(input_data).strip("[]'"))
        tokens = results["tokens"]
        keyword_frequency = results["keyword_frequency"]
        num_words = results["num_words"]
        num_chars = results["num_chars"]
        avg_word_length = results["avg_word_length"]
        num_exclamations = results["num_exclamations"]
        num_questions = results["num_questions"]
        num_periods = results["num_periods"]
        percent_uppercase = results["percent_uppercase"]
        file_path_usermodel_debug = call.User_Model_Debug()
        file_path_advancedusermodel_debug = call.Advanced_User_Model_Debug()


        #==================================================================================
        def rank_sulfur_output(return_dict=None, weights=None):
            """
            Single-function scorer: ranks sulfur output variables 1-100 (100 = very data-rich / high priority).
            - If return_dict is None, attempts to call _return_statements_sulfur().
            - weights: optional dict of per-key weights (positive numbers). Unspecified keys get small default weight.
            Returns a dict: {
                "overall_score": int 1-100,
                "per_key_scores": {key: int 1-100, ...},
                "ranked": [(key, score), ...]  # sorted desc
            }
            """

            # --- helpers (kept inside one function) ---
            def _is_placeholder(s):
                if s is None:
                    return True
                try:
                    t = str(s).strip().lower()
                except Exception:
                    return True
                if t == "":
                    return True
                placeholders = ("not available", "no data", "failed", "none", "n/a", "unknown", "prediction failed")
                for p in placeholders:
                    if p in t:
                        return True
                return False

            def _normalize_numeric_to_100(v):
                # numeric normalization: probabilities (0..1) -> 0..100, otherwise clamp to 0..100
                try:
                    f = float(v)
                except Exception:
                    return 0.0
                if 0.0 <= f <= 1.0:
                    return max(0.0, min(100.0, f * 100.0))
                return max(0.0, min(100.0, f))

            def _score_string(s):
                if s is None:
                    return 1
                txt = str(s).strip()
                if _is_placeholder(txt):
                    return 5
                length = len(txt)
                words = len(txt.split())
                lines = txt.count("\n") + 1
                # heuristics: length gives most; words add; multiple lines add small bonus
                length_score = min(70.0, length / 4.0)  # 280+ chars -> 70
                word_bonus = min(20.0, words * 1.5)  # word contribution
                line_bonus = min(10.0, (lines - 1) * 3.0)  # multiline bonus
                comma_bonus = 5.0 if "," in txt else 0.0
                total = length_score + word_bonus + line_bonus + comma_bonus
                # make sure small but non-zero
                return int(max(1, min(100, round(total))))

            def _score_value(k, v):
                # Booleans: True -> 80, False -> 20 (presence matters)
                if isinstance(v, bool):
                    return 80 if v else 20
                # Numbers
                if isinstance(v, (int, float)):
                    n = _normalize_numeric_to_100(v)
                    return int(max(1, min(100, round(n))))
                # Dicts: score by averaging sub-values (strings/numbers)
                if isinstance(v, dict):
                    if not v:
                        return 5
                    subs = []
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float)):
                            subs.append(_normalize_numeric_to_100(sv))
                        else:
                            subs.append(_score_string(sv))
                    avg = sum(subs) / max(1, len(subs))
                    bonus = min(15.0, len(v))  # bonus for richer dict structure
                    return int(max(1, min(100, round(avg + bonus))))
                # Fallback: try to parse number-like strings for known numeric keys
                numeric_like_keys = {
                    "PREDICTED_USER_DEVICE_ACCURACY", "PREDICTED_USER_LOCATION_CONFIDENCE",
                    "GLOBAL_OVERALL_ACCURACY", "USER_MOOD_PREDICTED_ACCURACY",
                    "GLOBAL_MOOD_PREDICTED_ACCURACY", "MOOD_AVERAGE_ACCURACY_ALL"
                }
                if isinstance(v, str) and k in numeric_like_keys:
                    try:
                        return int(max(1, min(100, round(_normalize_numeric_to_100(float(v))))))
                    except Exception:
                        pass
                # otherwise treat as string
                return _score_string(v)

            # --- acquire return_dict ---
            if return_dict is None:
                try:
                    return_dict = _return_statements_sulfur()
                    if not isinstance(return_dict, dict):

                        from scripts.ai_renderer_sentences.error import SulfurError
                        raise SulfurError(message=f"_return_statements_sulfur() did not return a dict")
                except Exception as e:

                    from scripts.ai_renderer_sentences.error import SulfurError
                    raise SulfurError(message=f"Could not obtain sulfur output:  {str(e)}")

            # --- compute per-key scores ---
            per_key_scores = {}
            for k, v in return_dict.items():
                try:
                    per_key_scores[k] = int(max(1, min(100, _score_value(k, v))))
                except Exception:
                    per_key_scores[k] = 1

            # --- build weight map ---
            # default sensible weights for some known important keys (you can override via weights arg)
            default_weights = {
                "INPUT_TEXT": 0.15,
                "PREDICTED_USER_DEVICE_ACCURACY": 0.10,
                "PREDICTED_USER_DEVICE": 0.05,
                "USER_MOOD_PREDICTED": 0.08,
                "USER_MOOD_PREDICTED_ACCURACY": 0.06,
                "GLOBAL_MOOD_PREDICTED": 0.03,
                "GLOBAL_MOOD_PREDICTED_ACCURACY": 0.02,
                "USER_SENTENCE_TYPE": 0.04,
                "USER_SENTENCE_INTENT": 0.04,
                "PREDICTED_USER_LOCATION_COUNTRY": 0.03,
                "PREDICTED_USER_LOCATION_CONFIDENCE": 0.04,
                "GLOBAL_OVERALL_ACCURACY": 0.03,
                "RESPONSE_TOTAL_TIME": 0.01
            }
            # start with tiny default for all keys
            small_default = 0.5 / max(1, len(per_key_scores))
            weights_used = {}
            for k in per_key_scores.keys():
                w = default_weights.get(k, small_default)
                weights_used[k] = float(w)

            # apply overrides if provided
            if isinstance(weights, dict):
                for k, w in weights.items():
                    try:
                        if w is None:
                            continue
                        weights_used[k] = float(w)
                    except Exception:
                        pass

            # normalize weights to sum=1
            total_w = sum(weights_used.values())
            if total_w <= 0:
                # fallback to equal weights
                n = len(weights_used) if weights_used else 1
                for k in list(weights_used.keys()):
                    weights_used[k] = 1.0 / n
            else:
                for k in list(weights_used.keys()):
                    weights_used[k] = weights_used[k] / total_w

            # --- compute weighted overall score ---
            overall = 0.0
            for k, score in per_key_scores.items():
                w = weights_used.get(k, 0.0)
                overall += float(score) * float(w)

            overall_score = int(max(1, min(100, round(overall))))

            # --- produce ranked list ---
            ranked = sorted(per_key_scores.items(), key=lambda kv: kv[1], reverse=True)

            return overall_score,per_key_scores,ranked

        overall_score, per_key_scores, ranked = rank_sulfur_output()

        with open(file_path_usermodel_debug, "w", encoding="utf-8", errors="ignore") as file:
            file.write(f"Tokens:  {str(tokens)} \n")
            file.write(f"Frequency of keywords : {keyword_frequency}\n")
            file.write(f"Percent of Uppercase letters : {percent_uppercase}%\n")
            file.write(f"Average word length : {avg_word_length} characters.\n")
            file.write(f"Flesch Reading Ease : {flesch_score}\n")
            file.write(f"Flesch-Kincaid Grade : {grade_level}\n")
            file.write(f"SMOG Index : {smog_index}\n")
            file.write(f"Gunning Fog : {gunning_fog}\n")
            file.write(f"Sentences: {sentence_count} | Words: {word_count}\n")
            file.write(f"Lemmas : {lemmas}\n")
            file.write(f"Position counts: : {pos_counts}\n")
            file.write(f"List of bi-grams : {smog_index}\n")
            file.write(f"Gunning Fog : {bigrams_list}\n")
            file.write(f"Keyframes: {keyphrases}\n")
            file.write(f"General sentiment : {sentiment}\n")
            file.write(f"General sentiment score : {sentiment_score}\n")
            file.write(f"General toxicity flags : {toxicity_flag}\n")
            file.write(f"General toxicity score : {toxicity_score}\n")
            file.write(f"(Count) words : {str(num_words)} \n")
            file.write(f"(Count) characters : {str(num_chars)} \n")
            file.write(f"(Count) exclamations : {str(num_exclamations)} \n")
            file.write(f"(Count) questions : {str(num_questions)} \n")
            file.write(f"(Count) periods : {str(num_periods)} \n")
            file.write(f"(Count) current sessions : {str(count_user_sessions)} \n")
            file.write(f"(String Count) hashtags : {str(hashtags)} \n")
            file.write(f"(String Count) emojis : {str(emojis)} \n")
            file.write(f"(String Count) casing : {str(casing)} \n")
            file.write(f"(String Count) matched_keywords : {str(matched_keywords)} \n")
            file.write(f"Recent user actions : {recent_events}\n")
            file.write(f"Percentage of user actions to sulfur runs: {events_to_script_percentage}%\n")
        with open(file_path_advancedusermodel_debug, "w", encoding="utf-8", errors="ignore") as file:
            file.write(f"Advanced speech actions:  {str(speech_act)} \n")
            file.write(f"Advanced speech action type:  {str(speech_act_type)} \n")
            file.write(f"Advanced speech tense:  {str(tense)} \n")
            file.write(f"Advanced mood:  {str(mood)} \n")
            file.write(f"Advanced mood II (further processed with a different model):  {str(mood_2)} \n")
            file.write(f"Advanced tense:  {str(tense)} \n")
            file.write(f"Advanced sentence types:  {str(sentence_type)} \n")
            file.write(f"Advanced clause counts:  {str(clause_count)} \n")
            file.write(f"Advanced token count:  {str(tokens)} \n")
            file.write(f"Advanced formality:  {str(formality)} \n")
            file.write(f"Advanced formality [positive] : informality [negative] score:  {str(score)} \n")
            file.write(f"Advanced slang that was found:  {str(found_slang)} \n")
            file.write(f"User's tone:  {str(tone)} \n")
            file.write(f"User's primary intent:  {str(primary_intent)} \n")
            file.write(f"User's audience:  {str(audience)} \n")
            file.write(f"User's polarity:  {str(polarity)} \n")
            file.write(f"Final anomaly score:  {str(final_score)} \n")
            file.write(f"Final mood score:  {str(mood_score)} \n")
            file.write(f"Final change score:  {str(change_score)} \n")
            file.write(f"Final detected anomalies:  {str(anomalies)} \n")
            file.write(f"Final detected anomalies (block) :  {str(anomaly_block)} \n")
        file_path_outputscore = call.Output_score()
        with open(file_path_outputscore, "w", encoding="utf-8", errors="ignore") as file:
            file.write(f"Overall_score: {overall_score}\n")
            file.write(f"Per_key_scores: {per_key_scores}\n")
            file.write(f"Ranked: {ranked}\n")
        if not endpoint_custom:
            try:
                with open(file_path_output, "w", encoding="utf-8", errors="ignore") as file:
                    file.write("------------------------------------------------\n")
                    file.write("|                 Sulfur Output*               |\n")
                    file.write("------------------------------------------------\n")
                    file.write("\n")
                    file.write("|---------------INPUT---------------\n")
                    file.write("|                                   \n")
                    file.write(f"|    Input (text) : {input_data}   \n")
                    file.write("|                                   \n")
                    if too_long:
                        file.write("|   Input Error : Input is too long. Stripped to below cap.     \n")
                    if re_was_subbed:
                        file.write("|   Certain unaccepted parts of the input may be removed.   \n")
                        file.write("|   This could affect output.                               \n")
                    file.write("|----------------------------------\n")
                    file.write("\n")
                    file.write("------------------------------------------------\n")
                    file.write("\n|                 DEVICES                     |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Predicts the users device using an extremely large scale intelligence model.\n")
                    file.write("|                                   \n")
                    file.write("|  *Predicted using Machine Learning.*  \n")
                    file.write("|                                   \n")
                    file.write(f"|  Predicted Device : {OutputDevice}   \n")
                    file.write("|                                   \n")
                    file.write(f"|  Predicted Device Accuracy : {Device_Accuracy}%  \n")
                    file.write("|                                   \n")
                    file.write(f"|  Main/Mean Devices : {main_devices}  \n")
                    file.write("|                                   \n")
                    file.write(f"|  Average/Mean Accuracy: {average_accuracy}%  \n")
                    file.write("|                                   \n")
                    file.write("|----------------------------------\n")
                    file.write("\n")

                    file.write("------------------------------------------------\n")
                    file.write("\n|                 PREFERENCES                |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Predicts the users preferences.\n")
                    file.write("|                                   \n")
                    file.write("|   Basic version limit: 3 preferred words!     \n")
                    file.write("*|  Predicted using Hard Values.*               \n")
                    file.write("|                                   \n")
                    file.write(f"|  User(s) preferred words : {','.join(preferences_user)} [Not Summarised] \n")
                    file.write("|                                   \n")
                    file.write(f"|  Average preferred words : {preferences_text_global} [Not Summarised]    \n")
                    file.write("|                                   \n")
                    file.write("|----------------------------------\n")

                    def section(title, user, global_val, extra=""):
                        file.write("|----------------------------------\n")
                        file.write("|                                   \n")
                        file.write(f"|          {title}          \n")
                        file.write("|                                   \n")
                        file.write(f"|  User(s): {user} {extra}  \n")
                        file.write(f"|  Average: {global_val} {extra}   \n")
                        file.write("|                                   \n")
                        file.write("|----------------------------------\n")

                    section("Nouns", wanted_noun_most_important_user, wanted_noun_most_important_global, "[To an extent of noun]")
                    section("Verbs", wanted_verb_most_important_user, wanted_verb_most_important_global, "[To an extent of verb]")
                    section("Adjectives", adjective_describe_user, adjective_describe_global, "[To an extent of adjective]")
                    section("Mood", mood_user, mood_global, "[To an extent of emotion]")

                    file.write("|----------------------------------------------------------------------------------------------------\n")
                    file.write("|                                                                                                   \n")
                    file.write(f"|      Predicted Mood Accuracy : {mood_accuracy_user}% (user), {mood_accuracy_global}% (global)    \n")
                    file.write(f"|      Average <User : Mean> Accuracy : {average_mood_accuracy}%                                   \n")
                    file.write("|       Predicted using a Neural Network.*                                                          \n")
                    file.write(
                        "|                                                                                                   \n")
                    file.write(f"|      Sentence Type : {stype_user} ({acc_sent_user * 100}%)                                       \n")
                    file.write(f"|      Sentence Intent : {sintent_user} ({acc_intent_user * 100}%)                                 \n")
                    file.write(
                        "|                                                                                                   \n")
                    file.write(f"|      Global Type : {avg_sent_types} ({acc_sent_global * 100}%)                                   \n")
                    file.write(f"|      Global Intent : {avg_intent_types} ({acc_intent_global * 100}%)                             \n")
                    file.write(f"|      Overall Accuracy : {avg_accuracy_global * 100}%                                             \n")
                    file.write("|                                                                                                   \n")
                    file.write("|----------------------------------------------------------------------------------------------------\n")
                    file.write("\n")

                    file.write("------------------------------------------------\n")
                    file.write("|                   USER INSIGHT                |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Predicts the userbase intent over a period of time.\n")
                    file.write(
                        "|                                                                                                   \n")
                    write_userbase_changes(file, "Day", past_d_changes, changes_summary_day, average_change_d, changes_d_apart_at_leastDays,False)
                    file.write(
                        "|                                                                                                   \n")
                    write_userbase_changes(file, "Week", past_w_changes, changes_summary_week, average_change_w, changes_w_apart_at_leastWeek,False)
                    file.write(
                        "|                                                                                                   \n")
                    write_userbase_changes(file, "Month", past_m_changes, changes_summary_month, average_change_m, changes_m_apart_at_leastMonth,False)
                    file.write(
                        "|                                                                                                   \n")
                    write_userbase_changes(file, "Year", past_y_changes, changes_summary_year, average_change_y, changes_y_apart_at_leastYear,False)
                    file.write(
                        "|                                                                                                   \n")

                    file.write("\n")
                    file.write("------------------------------------------------\n")
                    file.write("|              USER INSIGHT/LOCATION           |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Predicts the user's location with a medium intelligence model.\n")
                    file.write("|\n")
                    file.write(f"|  User(s) predicted location : {str(country)} \n")
                    file.write(f"|  User(s) predicted location accuracy : {confidence * 100}%\n")
                    file.write("|\n")
                    file.write(f"|  Average predicted location : {str(country_global)} \n")
                    file.write(f"|  Average predicted location accuracy : {confidence_global}%\n")
                    file.write("|\n")
                    file.write("|----------------------------------------------------|\n")
                    file.write("\n")
                    file.write(country_trends_list)
                    file.write("\n\n")

                    file.write("\n")
                    file.write("------------------------------------------------\n")
                    file.write("|               USER OPPORTUNITIES             |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Predicts the users opportunities (intent / business class).\n")
                    file.write("|\n")
                    file.write(
                        f"|  User(s) Main opportunity: {primary_opp}, Subsidiary opportunity: {subsidiary_opp}\n")
                    file.write(f"|  User(s) opportunities accuracy : {str(acc_opp_user * 100)}% \n")
                    file.write("|\n")
                    file.write("|----------------------------------------------------|\n")

                    file.write("\n")
                    file.write("------------------------------------------------\n")
                    file.write("|               USER : MODEL DEBUG             |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Uses quick and efficient inference models to make advanced predictions of already existing output.\n")
                    file.write(
                        "|   [!] It's recommended to use this debug with the already existing output on a 80:20 ratio for better accuracy.\n")
                    file.write("|\n")
                    file.write(f"|  Tokens:  {str(tokens)} \n")
                    file.write(f"|  Frequency of keywords : {keyword_frequency}\n")
                    file.write(f"|  Percent of Uppercase letters : {percent_uppercase}%\n")
                    file.write(f"|  Average word length : {avg_word_length} characters.\n")
                    file.write("|\n")
                    file.write(f"|  Flesch Reading Ease : {flesch_score}\n")
                    file.write(f"|  Flesch-Kincaid Grade : {grade_level}\n")
                    file.write(f"|  SMOG Index : {smog_index}\n")
                    file.write(f"|  Gunning Fog : {gunning_fog}\n")
                    file.write(f"|  Sentences: {sentence_count} | Words: {word_count}\n")
                    file.write("|\n")
                    file.write(f"|  Lemmas : {lemmas}\n")
                    file.write(f"|  Position counts: : {pos_counts}\n")
                    file.write(f"|  List of bi-grams : {smog_index}\n")
                    file.write(f"|  Gunning Fog : {bigrams_list}\n")
                    file.write(f"|  Keyframes: {keyphrases}\n")
                    file.write("|\n")
                    file.write(f"|  General sentiment : {sentiment}\n")
                    file.write(f"|  General sentiment score : {sentiment_score}\n")
                    file.write(f"|  General toxicity flags : {toxicity_flag}\n")
                    file.write(f"|  General toxicity score : {toxicity_score}\n")
                    file.write("|\n")
                    file.write(f"|  (Count) words : {str(num_words)} \n")
                    file.write(f"|  (Count) characters : {str(num_chars)} \n")
                    file.write(f"|  (Count) exclamations : {str(num_exclamations)} \n")
                    file.write(f"|  (Count) questions : {str(num_questions)} \n")
                    file.write(f"|  (Count) periods : {str(num_periods)} \n")
                    file.write(f"|  (String Count) hashtags : {str(hashtags)} \n")
                    file.write(f"|  (String Count) emojis : {str(emojis)} \n")
                    file.write(f"|  (String Count) casing : {str(casing)} \n")
                    file.write(f"|  (String Count) matched_keywords : {str(matched_keywords)} \n")
                    file.write("|\n")
                    file.write(f"|  (Count) current sessions : {str(count_user_sessions)} \n")
                    file.write("|\n")
                    file.write(f"|  Recent user actions : {recent_events}\n")
                    file.write("|\n")
                    file.write(f"| Percentage of user actions to sulfur runs: {events_to_script_percentage}%\n")
                    file.write("|\n")
                    file.write("|----------------------------------------------------|\n")

                    file.write("\n")
                    file.write("------------------------------------------------\n")
                    file.write("|          ADVANCED USER : MODEL DEBUG         |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write("|   [!] Description: Uses AI or large inference models to make advanced predictions of already existing output.\n")
                    file.write("|   [!] It's recommended to use this debug with the already existing output on a 80:20 ratio for better accuracy.\n")
                    file.write("|\n")
                    file.write(f"|  Advanced speech actions:  {str(speech_act)} \n")
                    file.write(f"|  Advanced speech action type:  {str(speech_act_type)} \n")
                    file.write(f"|  Advanced speech tense:  {str(tense)} \n")
                    file.write(f"|  Advanced mood:  {str(mood)} \n")
                    file.write(f"|  Advanced mood II (further processed with a different model):  {str(mood_2)} \n")
                    file.write(f"|  Advanced tense:  {str(tense)} \n")
                    file.write(f"|  Advanced sentence types:  {str(sentence_type)} \n")
                    file.write(f"|  Advanced clause counts:  {str(clause_count)} \n")
                    file.write(f"|  Advanced token count:  {str(tokens)} \n")
                    file.write(f"|  Advanced formality:  {str(formality)} \n")
                    file.write(f"|  Advanced formality [positive] : informality [negative] score:  {str(score)} \n")
                    file.write(f"|  * If the number is positive, it is a formal score. Opposites apply. \n")
                    file.write(f"|  Advanced slang that was found:  {str(found_slang)} \n")
                    file.write(f"|  User's tone:  {str(tone)} \n")
                    file.write(f"|  User's primary intent:  {str(primary_intent)} \n")
                    file.write(f"|  User's audience:  {str(audience)} \n")
                    file.write(f"|  User's polarity:  {str(polarity)} \n")
                    file.write("|\n")
                    file.write(f"|  Final anomaly score:  {str(final_score)} \n")
                    file.write(f"|  Final mood score:  {str(mood_score)} \n")
                    file.write(f"|  Final change score:  {str(change_score)} \n")
                    file.write("|\n")
                    file.write(f"|  Final detected anomalies:  {str(anomalies)} \n")
                    file.write("|\n")
                    file.write(f"|  Final detected anomalies (block) :  {str(anomaly_block)} \n")
                    file.write("|\n")
                    file.write("|----------------------------------------------------|\n")
                    file.write("\n")
                    file.write("------------------------------------------------\n")
                    file.write("|                   OUTPUT SCORE               |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(
                        "|   [!] Description: Provides a general output score the data profile renderer.\n")
                    file.write("|\n")
                    file.write(f"|  Overall_score: {overall_score}\n")
                    file.write(f"|  Per_key_scores: {per_key_scores}\n")
                    file.write(f"|  Ranked: {ranked}\n")
                    file.write("|\n")
                    file.write("|----------------------------------------------------|\n")
                    file.write("\n\n")

                    # === Optional Extra Output ===
                    file_path_ui_extra_output_settings = call.settings_ui_write_to_seperate_output()
                    with open(file_path_ui_extra_output_settings, "r", encoding="utf-8", errors="ignore") as file_extra:
                        ui_extra_output = file_extra.readline().strip()


                    file_path_extra_output = call.Output_UserInsight()
                    if ui_extra_output == "yes":
                        with open(file_path_extra_output, "w", encoding="utf-8", errors="ignore") as file:
                            file.write("--------------USER INSIGHT---------------\n")
                            write_userbase_changes(file, "Day", past_d_changes, changes_summary_day, average_change_d, changes_d_apart_at_leastDays,True)
                            write_userbase_changes(file, "Week", past_w_changes, changes_summary_week, average_change_w, changes_w_apart_at_leastWeek,True)
                            write_userbase_changes(file, "Month", past_m_changes, changes_summary_month, average_change_m, changes_m_apart_at_leastMonth,True)
                            write_userbase_changes(file, "Year", past_y_changes, changes_summary_year, average_change_y, changes_y_apart_at_leastYear,True)
                            file.write(country_trends_list)
                            file.write("\n\n")
                _dataprofiles_write_output(file_path_output)
                run_profile_builder.write_output_to_logs(file_path_output)
                #=================================DATAPROFILE!
                file_path_dp_settings = call.settings_auto_render_dp()
                cache_LocalDataProfileSulfurCount = call.cache_LocalDataProfileSulfurCount()
                with open(file_path_dp_settings, "r", encoding="utf-8", errors="ignore") as file: render_dp_auto = file.readline().strip()
                if render_dp_auto == "yes":
                    def update_counter(file_path="counter.txt"):
                        import os

                        """
                        Reads a file containing a number.
                        - If file is empty or does not exist -> write 1.
                        - If it has a number -> write next highest number.
                        Then check if the new number is 1 or a multiple of 20.
                        Returns (new_number, is_special) where is_special is True if the file was empty/missing
                        or the new number is 1 or a multiple of 20.
                        """

                        was_empty = False

                        # Ensure the file exists
                        if not os.path.exists(file_path):
                            with open(file_path, "w") as f:
                                f.write("1")
                            new_number = 1
                            was_empty = True
                        else:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read().strip()

                            if not content:  # empty file
                                new_number = 1
                                was_empty = True
                            else:
                                try:
                                    current = int(content)
                                    new_number = current + 1
                                except ValueError:
                                    # If the file has invalid content, reset to 1
                                    new_number = 1
                                    # treat invalid content as not-empty unless you want to mark it as empty:
                                    # was_empty = True

                            with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                                f.write(str(new_number))

                        # Check conditions
                        is_special = was_empty or (new_number == 1) or (new_number % 20 == 0)
                        return new_number, is_special

                    number, special = update_counter(file_path=cache_LocalDataProfileSulfurCount)
                    if special:
                        profile_instance = run_profile_builder.DataProfiles()
                        profile = profile_instance.generate_data_profile(is_main=True)
                        file_path_profile = run_profile_builder.file_path_dataprofileJSON()
                        with open(file_path_profile, "w", encoding="utf-8",errors="ignore") as file: file.write(profile)
                hours, minutes, seconds, total_time_ms = finish_render._finish_script(
                    start_time,
                    main=is_main
                )

                with open(file_path_output, "a", encoding="utf-8", errors="ignore") as file:
                    now = datetime.now()
                    file.write("------------------------------------------------\n")
                    file.write("|                   RESPONSE                   |\n")
                    file.write("------------------------------------------------\n")
                    file.write("|\n")
                    file.write(f"|  Response Time : {hours}h {minutes}m {seconds}s, {total_time_ms}ms\n")
                    file.write(f"|  Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}  \n")
                    file.write("|   General debugging only.\n")
                    file.write("|   *Settings can be changed. SulfurAI may make mistakes.*\n")
                    file.write("|---------------------------------------|\n")





                            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                #####################------------------------------------------------API RETURNS-----------------------------------------------

                import json
                data_dict = _return_statements_sulfur()
                json_string = json.dumps(data_dict)  # ✅ safely convert dict to JSON string
                return json_string

            except IOError as e:
                print(f"Error writing output: {e}")

        if endpoint_custom:
            try:
                with open (file_path_output, "w", encoding="utf-8", errors="ignore") as file:
                    file.write(str(_return_statements_sulfur()))
                return _return_statements_sulfur()
            except Exception as e:
                import time
                print(f"Error during endpoint custom return: {e}")
                time.sleep(100)


    except Exception as e:
        print(f"Unhandled exception during script run: {e}")



# python
# python
def _return_statements_sulfur():
    g = globals()
    return {
        "INPUT_TEXT": g.get("input_data", "Input not available"),
        "INPUT_TOO_LONG": g.get("too_long", False),
        "INPUT_HAD_UNACCEPTED_PARTS": g.get("re_was_subbed", False),
        "PREDICTED_USER_DEVICE": g.get("OutputDevice", "Device prediction failed"),
        "PREDICTED_USER_DEVICE_ACCURACY": g.get("Device_Accuracy", 0),
        "DATABASE_AVERAGE_DEVICE": g.get("main_devices", "No data available"),
        "DATABASE_AVERAGE_DEVICE_ACCURACY": g.get("average_accuracy", 0),
        "USER_MOOD_PREDICTED": g.get("mood_user", "Mood prediction failed"),
        "GLOBAL_MOOD_PREDICTED": g.get("mood_global", "Global mood prediction failed"),
        "USER_MOOD_PREDICTED_ACCURACY": g.get("mood_accuracy_user", 0),
        "GLOBAL_MOOD_PREDICTED_ACCURACY": g.get("mood_accuracy_global", 0),
        "MOOD_AVERAGE_ACCURACY_ALL": g.get("average_mood_accuracy", 0),
        "USER_SENTENCE_TYPE": g.get("stype_user", "Sentence type prediction failed"),
        "USER_SENTENCE_INTENT": g.get("sintent_user", "Sentence intent prediction failed"),
        "USER_SENTENCE_TYPE_ACCURACY": g.get("acc_sent_user", 0),
        "USER_SENTENCE_INTENT_ACCURACY": g.get("acc_intent_user", 0),
        "GLOBAL_SENTENCE_TYPE": g.get("avg_sent_types", "Global sentence type prediction failed"),
        "GLOBAL_SENTENCE_INTENT": g.get("avg_intent_types", "Global sentence intent prediction failed"),
        "GLOBAL_SENTENCE_TYPE_ACCURACY": g.get("acc_sent_global", 0),
        "GLOBAL_SENTENCE_INTENT_ACCURACY": g.get("acc_intent_global", 0),
        "GLOBAL_OVERALL_ACCURACY": g.get("avg_accuracy_global", 0),
        "PREDICTED_USER_LOCATION_COUNTRY": g.get("country", "Location prediction failed"),
        "PREDICTED_USER_LOCATION_CONFIDENCE": g.get("confidence", 0),
        "PREDICTED_USER_LOCATION_COUNTRY_GLOBAL": g.get("country_global", "Global location prediction failed"),
        "PREDICTED_USER_LOCATION_CONFIDENCE_GLOBAL": g.get("confidence_global", 0),
        "USER_MAIN_OPPORTUNITY": g.get("primary_opp", "User opportunity prediction failed"),
        "USER_SUBSIDIARY_OPPORTUNITY": g.get("subsidiary_opp", "User opportunity prediction failed"),
        "USER_OPPORTUNITY_ACCURACY": g.get("acc_opp_user", 0),
        "RESPONSE_TOTAL_TIME": {
            "HOURS": g.get("hours", 0),
            "MINUTES": g.get("minutes", 0),
            "SECONDS": g.get("seconds", 0),
            "TOTAL_TIME_MS": g.get("total_time_ms", 0),
        },
        "MODEL": g.get("model", "Model not available"),
        "ADVANCED_MODEL_DEBUG": {
            "SPEECH_ACT": g.get("speech_act"),
            "SPEECH_ACT_TYPE": g.get("speech_act_type"),
            "TENSE": g.get("tense"),
            "MOOD": g.get("mood"),
            "MOOD_2": g.get("mood_2"),
            "SENTENCE_TYPE": g.get("sentence_type"),
            "CLAUSE_COUNT": g.get("clause_count"),
            "TOKENS": g.get("tokens"),
            "FORMALITY": g.get("formality"),
            "SCORE": g.get("score"),
            "FOUND_SLANG": g.get("found_slang"),
            "TONE": g.get("tone"),
            "PRIMARY_INTENT": g.get("primary_intent"),
            "AUDIENCE": g.get("audience"),
            "POLARITY": g.get("polarity"),
            "FINAL_SCORE": g.get("final_score"),
            "MOOD_SCORE": g.get("mood_score"),
            "CHANGE_SCORE": g.get("change_score"),
            "ANOMALIES": g.get("anomalies"),
            "ANOMALY_BLOCK": g.get("anomaly_block"),
        },
        "MODEL_DEBUG": {
            "KEYWORD_FREQUENCY": g.get("keyword_frequency"),
            "PERCENT_UPPERCASE": g.get("percent_uppercase"),
            "AVG_WORD_LENGTH": g.get("avg_word_length"),
            "NUM_EXCLAMATIONS": g.get("num_exclamations"),
            "NUM_WORDS": g.get("num_words"),
            "NUM_CHARS": g.get("num_chars"),
            "NUM_QUESTIONS": g.get("num_questions"),
            "NUM_PERIODS": g.get("num_periods"),
            "HASHTAGS": g.get("hashtags"),
            "EMOJIS": g.get("emojis"),
            "CASING": g.get("casing"),
            "MATCHED_KEYWORDS": g.get("matched_keywords"),
            "COUNT_USER_SESSIONS": g.get("count_user_sessions"),
            "FLESCH_SCORE": g.get("flesch_score"),
            "GRADE_LEVEL": g.get("grade_level"),
            "SMOG_INDEX": g.get("smog_index"),
            "GUNNING_FOG": g.get("gunning_fog"),
            "LEMMAS": g.get("lemmas"),
            "POS_COUNTS": g.get("pos_counts"),
            "BIGRAMS_LIST": g.get("bigrams_list"),
            "KEYPHRASES": g.get("keyphrases"),
            "SENTIMENT": g.get("sentiment"),
            "SENTIMENT_SCORE": g.get("sentiment_score"),
            "TOXICITY_FLAG": g.get("toxicity_flag"),
            "TOXICITY_SCORE": g.get("toxicity_score"),
            "SENTENCE_COUNT": g.get("sentence_count"),
            "WORD_COUNT": g.get("word_count"),
        }
    }

def api_render_dp(api_key,USE_API,API_MODEL):
    """
    Uses large scale LLM models to generate a data profile of the input data - the end SulfurAI goal.

    When running, SulfurAI will switch to a "deep think" mode which consumes more resources and time.


    Returns:

        offline_dict: a dictionary containing AI assumed variables

        summary_dict: a dictionary containing the AI summary and annotations

    """
    from returns.dataprofiles.scripts import run_profile_builder
    import json
    import re

    profile_instance = run_profile_builder.DataProfiles()
    profile_str = profile_instance.generate_data_profile(is_main=False,API_KEY=api_key,use_api=USE_API,api_model=API_MODEL)

    # --- Extract the two sections ---
    if not USE_API:
        offline_match = re.search(r"==== OFFLINE PROFILE JSON ====(.*?)=====================================", profile_str, re.S)
        summary_match = re.search(r"==== AI SUMMARY & ANNOTATIONS ====(.*)", profile_str, re.S)

        offline_str = offline_match.group(1).strip() if offline_match else ""
        summary_str = summary_match.group(1).strip() if summary_match else ""

        # --- Parse JSON offline profile ---
        try:
            offline_dict = json.loads(offline_str)
        except Exception:
            offline_dict = {"raw_text": offline_str}

        # --- Keep AI summary as raw text (not JSON-formatted) ---
        summary_dict = {"ai_summary": summary_str}

    # --- Save combined JSON ---
    file_path_profile = run_profile_builder.file_path_dataprofileJSON()
    with open(file_path_profile, "w", encoding="utf-8", errors="ignore") as file:
        file.write(str(profile_str))

    # --- Return split dicts ---
    if not USE_API: return offline_dict, summary_dict
    if USE_API: return profile_str





