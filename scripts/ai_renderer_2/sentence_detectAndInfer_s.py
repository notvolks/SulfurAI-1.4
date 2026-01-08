import os
from datetime import datetime
from collections import Counter

from scripts.ai_renderer_sentences import error
error = error.error
from setup.verification.input_text import txt_data


def get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()

call = get_call_file_path()

ARTIFACT_DIR = call.artifacts_dir()


import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress most TF logging (errors only)
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import pandas as pd
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_usage = True
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else: gpu_usage = False


class SentenceIntentModel:
    from sklearn.preprocessing import LabelEncoder
    def __init__(self):
        self.tokenizer = None
        self.sent_type_encoder = None
        self.intent_encoder = None
        self.opp_encoder = None
        self.model = None
        self.maxlen = None

    def build_model(self, sentences, sentence_types, intents, opportunities):
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.tokenizer.fit_on_texts(sentences)
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, padding='post')
        self.maxlen = padded.shape[1]

        self.sent_type_encoder = self.LabelEncoder()
        self.intent_encoder = self.LabelEncoder()
        self.opp_encoder = self.LabelEncoder()

        y_sent = self.sent_type_encoder.fit_transform(sentence_types)
        y_intent = self.intent_encoder.fit_transform(intents)
        y_opp = self.opp_encoder.fit_transform(opportunities)

        # Model architecture
        input_layer = Input(shape=(self.maxlen,))
        x = Embedding(input_dim=20000, output_dim=64)(input_layer)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)

        sent_type_out = Dense(len(self.sent_type_encoder.classes_), activation='softmax', name="sentence_type")(x)
        intent_out = Dense(len(self.intent_encoder.classes_), activation='softmax', name="intent")(x)
        opp_out = Dense(len(self.opp_encoder.classes_), activation='softmax', name="opportunity")(x)

        self.model = Model(inputs=input_layer, outputs=[sent_type_out, intent_out, opp_out])

        self.model.compile(
            loss={
                "sentence_type": "sparse_categorical_crossentropy",
                "intent": "sparse_categorical_crossentropy",
                "opportunity": "sparse_categorical_crossentropy"
            },
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=["accuracy", "accuracy", "accuracy"]
        )

        batch_size = 256 if gpu_usage else 32

        self.model.fit(
            padded,
            {"sentence_type": y_sent, "intent": y_intent, "opportunity": y_opp},
            epochs=10,
            batch_size=batch_size,
            verbose=0 if not gpu_usage else 0
        )

        # Save artifacts
        self.model.save(os.path.join(ARTIFACT_DIR, 'sentence_intent_model.keras'))
        pickle.dump(self.tokenizer, open(os.path.join(ARTIFACT_DIR, 'tokenizer.pkl'), 'wb'))
        pickle.dump(self.sent_type_encoder, open(os.path.join(ARTIFACT_DIR, 'sent_type_encoder.pkl'), 'wb'))
        pickle.dump(self.intent_encoder, open(os.path.join(ARTIFACT_DIR, 'intent_encoder.pkl'), 'wb'))
        pickle.dump(self.opp_encoder, open(os.path.join(ARTIFACT_DIR, 'opp_encoder.pkl'), 'wb'))

    def generate_dummy_data(self, num_samples):
        dummy_sentences = [
            "This is a random sentence.",
            "What do you mean?",
            "I don't understand.",
            "Can you help me?",
            "Wow, that's amazing!",
            "It's raining outside."
        ]
        dummy_sentence_types = ["question", "statement", "command", "exclamation"]
        dummy_intents = ["asking", "informing", "commanding", "expressing", "comparing"]
        dummy_opportunities = [
            "Business_eCommerce",
            "Business_Consulting",
            "Business_Education",
            "NonBusiness_Social",
            "NonBusiness_Entertainment"
        ]

        sentences, types, intents, opps = [], [], [], []
        for _ in range(num_samples):
            sentences.append(random.choice(dummy_sentences))
            types.append(random.choice(dummy_sentence_types))
            intents.append(random.choice(dummy_intents))
            opps.append(random.choice(dummy_opportunities))

        return sentences, types, intents, opps

    @staticmethod
    def load_training_data(file_path, min_samples=50, min_unique_opps=3, min_per_class=5):
        """
        Loads training CSV and enforces minimal dataset size and opportunity-label diversity.
        Returns (sentences, sentence_types, intents, opportunities) or None to trigger fallback.
        """
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return None

        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            required = {"sentence", "sentence_type", "intent"}

            # basic presence checks
            if df.empty or not required.issubset(set(df.columns)):
                return None

            # require minimum number of rows
            if len(df) < min_samples:
                return None

            # Opportunity column MUST exist for real training (to avoid using incomplete CSVs)
            if "opportunity" not in df.columns:
                return None

            # Clean opportunity column: coerce obvious bad values to NaN, trim whitespace
            df["opportunity"] = df["opportunity"].where(df["opportunity"].notnull(), other=np.nan)
            df["opportunity"] = df["opportunity"].astype(str).str.strip()
            df.loc[df["opportunity"].str.lower().isin(["nan", "none", "na", "n/a", ""]), "opportunity"] = np.nan

            # drop rows missing core fields
            df = df.dropna(subset=["sentence", "sentence_type", "intent", "opportunity"])
            if df.empty or len(df) < min_samples:
                return None

            # Check opportunity label diversity
            opp_counts = df["opportunity"].value_counts()
            num_unique_opps = len(opp_counts)
            if num_unique_opps < min_unique_opps:
                return None

            # Ensure each class has at least `min_per_class` samples
            if (opp_counts < min_per_class).any():
                small_classes = opp_counts[opp_counts < min_per_class].to_dict()
                return None

            # Passed all checks — return cleaned lists
            sentences = df["sentence"].astype(str).tolist()
            sentence_types = df["sentence_type"].astype(str).tolist()
            intents = df["intent"].astype(str).tolist()
            opportunities = df["opportunity"].astype(str).tolist()

            return sentences, sentence_types, intents, opportunities

        except Exception as e:
            return None

    def predict_sentence(self, sentence, add_to_training_data):
        from extra_models.Sulfur.Models.base_sulfur_drl_build.filter import filter_text

        # Try to load model & encoders (prefer ARTIFACT_DIR)
        try:
            if self.model is None:
                model_path = os.path.join(ARTIFACT_DIR, 'sentence_intent_model.keras')
                if os.path.exists(model_path):
                    self.model = load_model(model_path)
                elif os.path.exists('sentence_intent_model.h5'):
                    self.model = load_model('sentence_intent_model.h5')
        except Exception:
            # ignore loading errors here; downstream code may still use encoders if present
            pass

        try:
            if self.tokenizer is None:
                tok_path = os.path.join(ARTIFACT_DIR, 'tokenizer.pkl')
                if os.path.exists(tok_path):
                    self.tokenizer = pickle.load(open(tok_path, 'rb'))
                else:
                    self.tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
        except Exception:
            pass

        try:
            if self.sent_type_encoder is None:
                e_path = os.path.join(ARTIFACT_DIR, 'sent_type_encoder.pkl')
                if os.path.exists(e_path):
                    self.sent_type_encoder = pickle.load(open(e_path, 'rb'))
                else:
                    self.sent_type_encoder = pickle.load(open('sent_type_encoder.pkl', 'rb'))
        except Exception:
            pass

        try:
            if self.intent_encoder is None:
                i_path = os.path.join(ARTIFACT_DIR, 'intent_encoder.pkl')
                if os.path.exists(i_path):
                    self.intent_encoder = pickle.load(open(i_path, 'rb'))
                else:
                    self.intent_encoder = pickle.load(open('intent_encoder.pkl', 'rb'))
        except Exception:
            pass

        try:
            if self.opp_encoder is None:
                o_path = os.path.join(ARTIFACT_DIR, 'opp_encoder.pkl')
                if os.path.exists(o_path):
                    self.opp_encoder = pickle.load(open(o_path, 'rb'))
                else:
                    self.opp_encoder = pickle.load(open('opp_encoder.pkl', 'rb'))
        except Exception:
            pass

        # Tokenize / pad
        seq = self.tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(seq, padding='post', maxlen=self.maxlen)

        # Run prediction (safely)
        sent_pred, intent_pred, opp_pred = self.model.predict(padded, verbose=0)

        # Decode predictions (guard with try/except)
        try:
            sent_type = self.sent_type_encoder.inverse_transform([np.argmax(sent_pred)])[0]
        except Exception:
            sent_type = ""

        try:
            intent = self.intent_encoder.inverse_transform([np.argmax(intent_pred)])[0]
        except Exception:
            intent = ""

        try:
            opp = self.opp_encoder.inverse_transform([np.argmax(opp_pred)])[0]
        except Exception:
            opp = ""

        opp_str = "" if opp is None else str(opp).strip()

        # default split
        if "_" in opp_str:
            primary_opp, subsidiary_opp = opp_str.split("_", 1)
        else:
            # fallback heuristics: detect prefix keywords, otherwise put all into subsidiary
            low = opp_str.lower()
            if low.startswith("business"):
                primary_opp = "Business"
                subsidiary_opp = opp_str[len("Business"):].lstrip("_ ").strip() or "General"
            elif low.startswith("nonbusiness") or low.startswith("non_business") or low.startswith("non-business"):
                primary_opp = "NonBusiness"
                subsidiary_opp = opp_str.split("_", 1)[-1] if "_" in opp_str else "General"
            else:
                # nothing obvious — treat whole as subsidiary with NonBusiness primary
                primary_opp = "NonBusiness"
                subsidiary_opp = opp_str or "General"

        # Default training_data_df to empty string so return shape is consistent
        training_data_df = ""

        # If we have a training CSV path configured, try to read it and optionally append
        if file_path_path_versionDATA_name_sentences is not None:
            try:
                training_data_df = pd.read_csv(file_path_path_versionDATA_name_sentences)
                training_data_df.columns = training_data_df.columns.str.strip()

                # Clean input before saving
                if isinstance(sentence, list):
                    text = ''.join(item.replace(',', '') for item in sentence)
                else:
                    text = str(sentence)

                cleaned_input_text = text.strip()
                if cleaned_input_text:
                    result = filter_text.preprocess_chat(cleaned_input_text)
                    clean_text = result['clean_text'].strip()

                    # Skip saving if too short
                    file_path_savedata = call.settings_save_training_data()
                    with open(file_path_savedata, "r", encoding="utf-8", errors="ignore") as f:
                        save_data = f.read().strip()

                    if (not result["annotations"].get("is_too_short", False)) and save_data == "yes":
                        if add_to_training_data:
                            # Use the model's predicted opportunity (opp) when saving
                            new_row = {
                                'sentence': clean_text,
                                'sentence_type': sent_type,
                                'intent': intent,
                                'opportunity': opp,
                                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }

                            new_row_df = pd.DataFrame([new_row])
                            training_data_df = pd.concat([training_data_df, new_row_df], ignore_index=True)
                            training_data_df.to_csv(file_path_path_versionDATA_name_sentences, index=False)
            except Exception as e:
                # Keep training_data_df as empty string if reading/saving fails
                training_data_df = ""
                print(f"Warning: error reading/saving training CSV: {e}")

        # ALWAYS return the same 4-tuple in the same order:
        # (sentence_type, intent, opportunity, training_data_df_or_empty)
        return sent_type, intent, primary_opp, subsidiary_opp, training_data_df

    def evaluate_model(self, sentences, sentence_types, intents, opportunities):
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, padding='post', maxlen=self.maxlen)

        y_sent = self.sent_type_encoder.transform(sentence_types)
        y_intent = self.intent_encoder.transform(intents)
        y_opp = self.opp_encoder.transform(opportunities)

        loss, sent_loss, intent_loss, opp_loss, sent_acc, intent_acc, opp_acc = self.model.evaluate(
            padded,
            {"sentence_type": y_sent, "intent": y_intent, "opportunity": y_opp},
            verbose=0
        )
        return sent_acc, intent_acc, opp_acc


def sentence_intent_and_infer(add_to_training_data):
    from sklearn.metrics import accuracy_score
    acc_sent_global = 0
    acc_intent_global = 0
    avg_accuracy_global = 0
    avg_sent_types = ""
    avg_intent_types = ""
    global file_path_path_versionDATA_name_sentences
    file_path_path_versionDATA_name_sentences = call.versionDATA_trainingdata_sentences()
    training = SentenceIntentModel.load_training_data(file_path_path_versionDATA_name_sentences)
    model_obj = SentenceIntentModel()

    if training:
        sentences, sentence_types, intents, opportunities = training

    else:
        sentences, sentence_types, intents, opportunities = model_obj.generate_dummy_data(50)


    model_obj.build_model(sentences, sentence_types, intents, opportunities)

    # ----------------------USER
    prompt, too_long, re_was_subbed = txt_data.verify_input("string")
    user_input = prompt.strip()
    stype_user, sintent_user, primary_opp, subsidiary_opp, _ = model_obj.predict_sentence(user_input, add_to_training_data)
    file_path_OutputData_Sentence_Intent_User = call.Sentence_Intent_User()
    file_path_OutputData_Sentence_Type_User = call.Sentence_Type_User()
    with open(file_path_OutputData_Sentence_Intent_User, "w", encoding="utf-8", errors="ignore") as file: file.write(sintent_user)
    with open(file_path_OutputData_Sentence_Type_User, "w", encoding="utf-8", errors="ignore") as file: file.write(stype_user)
    acc_sent_user, acc_intent_user, acc_opp_user = model_obj.evaluate_model(
        sentences, sentence_types, intents, opportunities
    )
    #print(f"\n📊 Final Sentence Type Accuracy: {acc_sent:.2%}")
    #print(f"📊 Final Intent Accuracy: {acc_intent:.2%}")
    # ----------------------GLOBAL
    if os.access(file_path_path_versionDATA_name_sentences, os.R_OK):
        try:


            with open(file_path_path_versionDATA_name_sentences, 'r', encoding='utf-8') as file:
                file_contents = file.read()


            df = pd.read_csv(file_path_path_versionDATA_name_sentences)
            df.columns = df.columns.str.strip()


            sentences = df['sentence'].tolist()
            true_sent_types = df['sentence_type'].tolist()
            true_intents = df['intent'].tolist()

            # Python
            def get_predictions(model, tokenizer, sentences, maxlen):
                sequences = tokenizer.texts_to_sequences(sentences)
                padded = pad_sequences(sequences, padding='post', maxlen=maxlen)
                sent_preds, intent_preds, opp_preds = model.predict(padded, verbose=0)
                sent_type_preds = np.argmax(sent_preds, axis=1)
                intent_preds = np.argmax(intent_preds, axis=1)
                opp_preds = np.argmax(opp_preds, axis=1)
                return sent_type_preds, intent_preds, opp_preds


            true_sent_types_encoded = model_obj.sent_type_encoder.transform(true_sent_types)
            true_intents_encoded = model_obj.intent_encoder.transform(true_intents)

            # Get the predictions from the model
            # Python
            pred_sent_types, pred_intents, pred_opps = get_predictions(model_obj.model, model_obj.tokenizer, sentences,
                                                                       model_obj.maxlen)


            sent_type_accuracy = accuracy_score(true_sent_types_encoded, pred_sent_types)
            intent_accuracy = accuracy_score(true_intents_encoded, pred_intents)
            # Prepare true_opps and make sure all labels exist in the fitted encoder
            true_opps = df['opportunity'].astype(str).str.strip().tolist()

            # Encoder classes (allowed labels)
            allowed = set(model_obj.opp_encoder.classes_.tolist())

            # If any true_opps value is not in encoder classes, map it to a fallback that IS in classes.
            # Choose fallback as the most common class from encoder.classes_ (index 0 if none)
            fallback = model_obj.opp_encoder.classes_[0] if len(model_obj.opp_encoder.classes_) > 0 else random.choice([
                "Business_eCommerce", "Business_Consulting", "Business_Education",
                "NonBusiness_Social", "NonBusiness_Entertainment"
            ])

            sanitized_true_opps = [val if val in allowed else fallback for val in true_opps]
            true_opps_encoded = model_obj.opp_encoder.transform(sanitized_true_opps)

            opp_accuracy = accuracy_score(true_opps_encoded, pred_opps)

            avg_accuracy = (sent_type_accuracy + intent_accuracy + opp_accuracy) / 3

            acc_sent_global = sent_type_accuracy #sentence accuracy
            acc_intent_global = intent_accuracy #intent accuracy
            avg_accuracy_global = avg_accuracy #average accuracy
            avg_sent_types =  model_obj.sent_type_encoder.inverse_transform(pred_sent_types)
            avg_intent_types = model_obj.intent_encoder.inverse_transform(pred_intents)
            avg_sent_types = Counter(avg_sent_types).most_common(1)[0][0] #sentence type
            avg_intent_types = Counter(avg_intent_types).most_common(1)[0][0] #intent type

            try:

                file_path_OutputData_Sentence_Intent_Global = call.Sentence_Intent_Global()
                file_path_OutputData_Sentence_Type_Global = call.Sentence_Type_Global()
                file_path_OutputData_Sentence_Accuracy_Average_Global = call.Sentence_Accuracy_Average_Global()
                file_path_OutputData_Sentence_Intent_Accuracy_Global = call.Sentence_Accuracy_Intent_Global()
                file_path_OutputData_Sentence_Type_Accuracy_Global = call.Sentence_Accuracy_Type_Global()
                file_path_OutputData_Opportunity_Accuracy_Global = call.Sentence_Oppurtunity_Accuracy()
                file_path_OutputData_Opportunity = call.Sentence_Oppurtunity()
                with open(file_path_OutputData_Sentence_Intent_Global, "w", encoding="utf-8", errors="ignore") as file: file.write(avg_intent_types)
                with open(file_path_OutputData_Sentence_Type_Global, "w", encoding="utf-8", errors="ignore") as file: file.write(avg_sent_types)
                with open(file_path_OutputData_Sentence_Accuracy_Average_Global, "w", encoding="utf-8", errors="ignore") as file: file.write(str(avg_accuracy_global * 100))
                with open(file_path_OutputData_Sentence_Intent_Accuracy_Global, "w", encoding="utf-8", errors="ignore") as file:  file.write(str(acc_intent_global * 100))
                with open(file_path_OutputData_Sentence_Type_Accuracy_Global, "w", encoding="utf-8",errors="ignore") as file: file.write(str(acc_sent_global * 100))
                with open(file_path_OutputData_Opportunity_Accuracy_Global, "w", encoding="utf-8",errors="ignore") as file:  file.write(str(acc_opp_user * 100))
                with open(file_path_OutputData_Opportunity, "w", encoding="utf-8",errors="ignore") as file:file.write(f"Main opportunity: {primary_opp}, Subsidiary opportunity: {subsidiary_opp}")


            except (AttributeError,TypeError,FileNotFoundError,OSError,IOError,UnicodeEncodeError,NameError) as e:
                print(f"Output_WRITE error: {e}")
                error("er1", "Output_WRITE", "Error writing output for sentence_detectAndInfer_s.py", "13")
        except PermissionError as e:
            print(f"Permission error: {e}")
            error("er1", "Output_PERMISSION", "Error getting permission for sentence_detectAndInfer_s.py", "14")
        except Exception as e:
            print(f"An error occurred: {e}")
            error("er1", "Output_EXCEPTION", "Error general for sentence_detectAndInfer_s.py", "15")
    else:
        print(f"Cannot access the file. Please check the permissions for: {file_path_path_versionDATA_name_sentences}")
        error("er1", "Output_PERMISSION_ACCESS", "Error getting permission to access VERSIONDATA/SENTENCE (GENERAL) for sentence_detectAndInfer_s.py", "16")

    return (
        stype_user, sintent_user, primary_opp,subsidiary_opp,
        acc_sent_user, acc_intent_user, acc_opp_user,
        avg_sent_types, avg_intent_types,
        acc_sent_global, acc_intent_global,
        avg_accuracy_global
    )



