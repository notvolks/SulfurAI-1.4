
################ Welcome to SulfurAI!
### Functions here should be modified with proper intent.
### This python script was written in the Holly format. To find out how it works go into setup/HollyFormat/ReadMe.txt
### This python script is designed to host all SulfurAI API functions for python and run via the __main__ tag.
#-
### LAYOUT:
# ---------------GOING DOWN!
##### -Importing base level items, including TOS notice.
##### -PIP install, installing all dependancies.
##### -Importing external level items, including file paths,
##### -Runs the architecture, ensuring it runs cleanly and does not return an error.
##### -Hosts the sulfur scripts, runs secondary modules and processes with machine learning + neural networking
##### -Writes to the output and API files.
##### -Hosts the module (API) files.
##### -Hosts the MODEL (AI) files.

# ---------------SECTIONS:
#1) SET-UP
#2) RENDER SCRIPTS
#3) MAIN FUNCTIONS
#4) API MODULES
#5) SERVER API MODULES
#6) CUSTOM AI MODELS


#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#|                                                                                                                                                                        |
#|                                                                       S E T - U P                                                                                      |
#|                                                                                                                                                                        |
#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from scripts.ai_renderer_sentences import error
from extra_models.Sulfur.GeneralScripts.LocalUserEvents import events_hoster






print_verti_list = error.print_verti_list
error_print = error.error

if __name__ == "__main__":
    TOS = [
        "--------------------------------------------------------------------------------------------------",
        "|                                          SULFUR AI                                              |",
        "--------------------------------------------------------------------------------------------------",
        "| By using this application you agree to the Terms of Service listed in the project files.",
        "| If you do not consent, stop using our services.",
        "| If you cannot find it, install a new version OR look in the root folder for 'Terms of Service.txt' .",
        "--------------------------------------------------------------------------------------------------",
        "|",
    ]
    print_verti_list(TOS)
    from extra_models.Sulfur.Models.manager import find_models
    find_models.add_active_model("SulfurAI")

# DELETING THE TOS NOTICE SCRIPT RESULTS IN INSTANT TERMINATION OF SULFUR WARRANTY AND CANCELS YOUR CONTRACT. IT IS *IN VIOLATION* OF THE TOS.
# YOU MAY BE INDEFINITELY BANNED FROM SULFUR SERVICES IF YOU REMOVE THIS TOS NOTICE SCRIPT WITHOUT PRIOR WRITTEN CONSENT BY VOLKSHUB GROUP.

def _get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()

# Call file paths
call = _get_call_file_path()



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
####################ENV VARIABLES
try:  os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
except Exception as e: print(f"Warning: Could not set PYGAME_HIDE_SUPPORT_PROMPT: {e}")

try: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except Exception as e:  print(f"Warning: Could not set TF_CPP_MIN_LOG_LEVEL: {e}")

try:  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
except Exception as e: print(f"Warning: Could not set TF_ENABLE_ONEDNN_OPTS: {e}")

try:  os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
except Exception as e:   print(f"Warning: Could not set CUDA_PATH: {e}")
  

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
####INSTALLER + VENV SET UP
from setup.depenintsall.python311 import pipdependancyinstaller
pipdependancyinstaller.init()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### WRAPPERS
import warnings
from functools import wraps
def not_available(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is not available",
            category=DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#####################------------------------------------------------MODULES------------------------------------------------

import sys
from datetime import datetime
import time,math
from scripts.ai_renderer_sentences import module_restore_trainingData

module_restore_trainingData.restore_data()

folder_path_verify = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/training_data/dependancy_data/QVOL')
file_name_verify = "vrs.txt"
file_path_verify = os.path.join(folder_path_verify, file_name_verify)

# Global variables
global machine_model_checkDevice_result, machine_model_checkDevice_result_accuracy, accuracy
machine_model_checkDevice_result = 0
machine_model_checkDevice_result_accuracy = 0
accuracy = 0


# Global variables for device output
global Sulfur_Output_Device_Desktop_Percent, Sulfur_Output_DeviceMobileORother_Percent, OutputDevice, Device_Result
Device_Result = "NOT_SET"



file_path_OutputData_name_Device = call.device()  # Assume this returns a single path
(folder_path_OuputData, folder_path_OuputData_name_Device_accuracy, file_path_OutputData_name_Device_accuracy) = call.device_accuracy()
(folder_path_OuputData, folder_path_OuputData_name_Response_Time_MS, file_path_OutputData_name_Response_Time_MS) = call.response_time()
(folder_path_trainingData_grammar, folder_path_trainingData_grammar_name, file_path_trainingData_grammar) = call.grammar()
(folder_path_output, folder_path_output_name, file_path_output) = call.output()
(folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk) = call.training_data_sk()



#####################------------------------------------------------ARCHITECTURE VERIFICATION------------------------------------------------
####################Architechture:
ARCH_IS_VALID = 0
file_path_arch = call.arch_runner()
folder_path_arch = call.arch_runner_folder()
if os.path.exists(file_path_arch): ARCH_IS_VALID += 1
if os.path.exists(folder_path_arch): ARCH_IS_VALID += 1
##########define ARCHITECTURE
def start_arch():
    arch = arch_runner.Arch()
    arch.check_lines()
    arch.verify_input()

##########run ARCHITECTURE

if ARCH_IS_VALID == 2:
    try:
        from extra_models.Sulfur.ArchitectureBuild import arch_runner
        start_arch()
    except (ImportError,AttributeError,TypeError,FileNotFoundError,ValueError) as e:
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message="Architechture failed to start.")

else:
    try:
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message="ARCHITECTURE COULD NOT BE VERIFIED")
    except (ImportError,TypeError,AttributeError,KeyboardInterrupt) as e:
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message="ARCHITECTURE COULD NOT BE VERIFIED' RAN INTO AN ERROR)!")



#####################------------------------------------------------SULFUR------------------------------------------------

def _variable_call():
    """
     Initializes and returns core variables used in the Sulfur AI script.

     Returns:
         tuple: Contains version string, file links, and device percentages (all default to 0 or empty).
     """
    version = "[DRL]"
    file_link = ""
    file_link_a = ""
    file_link_o = ""
    Sulfur_Output_Device_Desktop_Percent = 0
    Sulfur_Output_DeviceMobileORother_Percent = 0
    return version, file_link, file_link_a, file_link_o, Sulfur_Output_Device_Desktop_Percent, Sulfur_Output_DeviceMobileORother_Percent

version, file_link, file_link_a, file_link_o, Sulfur_Output_Device_Desktop_Percent, Sulfur_Output_DeviceMobileORother_Percent = _variable_call()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
brick_out = error.brick_out
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#|                                                                                                                                                                        |
#|                                                      R E N D E R     S C R I P T S                                                                                     |
#|                                                                                                                                                                        |
#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Certain render scripts have been moved to 'extra_models/Sulfur/RenderScript/Build'.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _rest_of_the_script(tag_trainer,return_statements,start_time,output_file_path,add_to_training_data=True,endpoint_custom=False):
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

    from extra_models.Sulfur.RenderScript.Build import render_scripts_general
    if __name__ == "__main__": is_main = True
    else: is_main = False
    return_id = render_scripts_general._rest_of_the_script(
        tag_trainer=tag_trainer,
        return_statements=return_statements,
        start_time=start_time,
        output_file_path=output_file_path,
        add_to_training_data=add_to_training_data,
        is_main=is_main,
        endpoint_custom=endpoint_custom,
    )


    return return_id


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#####################------------------------------------------------API RETURNS-----------------------------------------------
def _return_statements_sulfur():
    from extra_models.Sulfur.RenderScript.Build import render_scripts_general
    return render_scripts_general._return_statements_sulfur()







# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#|                                                                                                                                                                        |
#|                                                      M A I N     F U N C T I O N S                                                                                     |
#|                                                                                                                                                                        |
#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


# Menu & Variables
start_time = datetime.now()
start_time_printed = start_time.strftime(
    "%Y-%m-%d %H:%M:%S")  # Calculating what it starts at for rest_of_the_script(attributes)
start_time_ms = f".{start_time.microsecond // 1000:03d}"

if __name__ == "__main__":

    list_menu = [
        f"---------------------------Sulfur AI {version}---------------------------",
        "",

    ]  # Menu
    print_verti_list(list_menu)


    try:
        from returns.dataprofiles.scripts import run_profile_builder
        events_hoster.write_event("event_RanSulfurViaMain")
        run_profile_builder.setup_dataprofile_if()
        _rest_of_the_script("None", True, start_time,"[]", add_to_training_data=True)
        if __name__ == "__main__":
            time.sleep(100)  # Developer debug only
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError

        raise SulfurError(message=f"{e}")


def _call_file_input():
    """
    Initializes the `file_link` variable to the path of the input file used by SulfurAI.
    """
    global file_link
    file_link = call.file_path_input()

def _call_file_attributes():
    """
      Initializes the `file_link_a` variable to the path of the attributes file.
      """
    global file_link_a
    file_link_a = call.file_path_attributes()

def _call_file_output():
    """
    Initializes the `file_link_o` variable to the path of the output file.
    """
    global file_link_o
    file_link_o = file_path_output

def _broadcast_verify(file):
    """
    Checks if a given file exists.

    Args:
        file (str): File path to check.

    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.exists(file)

# Debugger
if not os.path.exists(call.folder_path_input()):  # Checks if it can be verified
    error_print("er1", "DEPENDANCY_FILE", "VERIFY",1)
else:
    with open(file_path_verify, 'r') as file:
        lines = file.readlines()
        if lines:
            first_line = lines[0].strip()
            if first_line != "os1_tvt":
                error_print("er2", "Verification", "INSTALL NEW VER",1)
                brick_out(1000)


#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#|                                                                                                                                                                        |
#|                                                            A P I   M O D U L E                                                                                         |
#|                                                                                                                                                                        |
#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

#------------------------------------------------MODULE IMPORT SCRIPTS (API)------------------------------------------------
#####################       Some API scripts are held in a different module. Check API/Python/Sulfur for more information.
#---------------------------------------------------------------------------------------------------------------------------
from setup.exceptions import except_host

from api.Python.Sulfur import api_setup_local
from api.Python.Sulfur import api_get_output_data
from api.Python.Sulfur import api_get_output_data_ui

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import SulfurConsole
sulfur_console = SulfurConsole.console()
print_api_debug = sulfur_console.check_api_debug_print(print_status=False)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def setup_local(directory=None):
    """
    Sets up SulfurAI locally by adding the given directory to the PYTHONPATH.
    If no directory is provided, uses the current working directory.
    """
    from returns.dataprofiles.scripts import run_profile_builder
    events_hoster.write_event("event_SetupAPILocal")
    run_profile_builder.setup_dataprofile_if()
    api_setup_local.setup_local(directory=None)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_via_trainer(tag_trainer):
    """
        Executes SulfurAI in training mode using a given tag.

        Args:
            tag_trainer (str): Identifier tag to label training session.
        """
    try:
        from datetime import datetime
        start_time = datetime.now()
        _rest_of_the_script(tag_trainer, False, start_time, "[]")
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"{e}")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_locally(input_string, add_to_training_data=True):
    """
    Runs SulfurAI locally by overwriting the input file. All returns are in a dictionary with detailed variable descriptions.

    The API output contains:

      • INPUT_TEXT: The input string that was processed.
      • INPUT_TOO_LONG: A flag indicating whether the input exceeded the character cap.
      • INPUT_HAD_UNACCEPTED_PARTS: A flag indicating whether unsupported characters were stripped.
      • PREDICTED_USER_DEVICE: The device predicted for the user.
      • PREDICTED_USER_DEVICE_ACCURACY: Accuracy of the user device prediction.
      • DATABASE_AVERAGE_DEVICE: The predicted device from the database.
      • DATABASE_AVERAGE_DEVICE_ACCURACY: Accuracy of the database device prediction.
      • USER_MOOD_PREDICTED: The predicted mood of the user.
      • GLOBAL_MOOD_PREDICTED: The predicted mood of the overall userbase.
      • USER_MOOD_PREDICTED_ACCURACY: Accuracy of the user mood prediction.
      • GLOBAL_MOOD_PREDICTED_ACCURACY: Accuracy of the global mood prediction.
      • MOOD_AVERAGE_ACCURACY_ALL: Average mood accuracy combining user and global.
      • USER_SENTENCE_TYPE: Predicted sentence type for the input.
      • USER_SENTENCE_INTENT: Predicted sentence intent for the input.
      • USER_SENTENCE_TYPE_ACCURACY: Accuracy of the sentence type prediction.
      • USER_SENTENCE_INTENT_ACCURACY: Accuracy of the sentence intent prediction.
      • GLOBAL_SENTENCE_TYPE: Predicted sentence type for the overall dataset.
      • GLOBAL_SENTENCE_INTENT: Predicted sentence intent for the overall dataset.
      • GLOBAL_SENTENCE_TYPE_ACCURACY: Accuracy of the global sentence type prediction.
      • GLOBAL_SENTENCE_INTENT_ACCURACY: Accuracy of the global sentence intent prediction.
      • GLOBAL_OVERALL_ACCURACY: Overall accuracy from the database predictions.
      • PREDICTED_USER_LOCATION_COUNTRY: Predicted country based on the input.
      • PREDICTED_USER_LOCATION_CONFIDENCE: Confidence level of the user location prediction.
      • PREDICTED_USER_LOCATION_COUNTRY_GLOBAL: Global country prediction.
      • PREDICTED_USER_LOCATION_CONFIDENCE_GLOBAL: Global location prediction accuracy.
      • USER_MAIN_OPPORTUNITY: The main opportunity prediction for the user.
      • USER_SUBSIDIARY_OPPORTUNITY: The subsidiary opportunity from the prediction.
      • USER_OPPORTUNITY_ACCURACY: Accuracy for the opportunity prediction.
      • MODEL: Details of the current AI model used.
      • ADVANCED_MODEL_DEBUG: A nested dictionary of advanced model debug variables such as:
          - SPEECH_ACT, SPEECH_ACT_TYPE, TENSE, MOOD, MOOD_2, SENTENCE_TYPE, CLAUSE_COUNT,
          - TOKENS, FORMALITY, SCORE, FOUND_SLANG, TONE, PRIMARY_INTENT, AUDIENCE,
          - POLARITY, FINAL_SCORE, MOOD_SCORE, CHANGE_SCORE, ANOMALIES, ANOMALY_BLOCK.
      • MODEL_DEBUG: A nested dictionary of model debug details including:
          - KEYWORD_FREQUENCY, PERCENT_UPPERCASE, AVG_WORD_LENGTH, NUM_EXCLAMATIONS,
          - NUM_WORDS, NUM_CHARS, NUM_QUESTIONS, NUM_PERIODS, HASHTAGS, EMOJIS,
          - CASING, MATCHED_KEYWORDS, COUNT_USER_SESSIONS, FLESCH_SCORE, GRADE_LEVEL,
          - SMOG_INDEX, GUNNING_FOG, LEMMAS, POS_COUNTS, BIGRAMS_LIST, KEYPHRASES,
          - SENTIMENT, SENTIMENT_SCORE, TOXICITY_FLAG, TOXICITY_SCORE, SENTENCE_COUNT, WORD_COUNT.
      • RESPONSE_TOTAL_TIME: A nested dictionary with the response times:
          - HOURS, MINUTES, SECONDS, TOTAL_TIME_MS.

    Args:
        input_string (str): The input to be processed by SulfurAI.
        add_to_training_data (bool, optional): Whether to add the input to training data [default: True].

    Returns:
        dict: A dictionary containing the above variables and their descriptions.
    """
    from extra_models.Sulfur.Models.manager import find_models
    find_models.add_active_model("SulfurAI")
    current_dir_i_mod = os.path.abspath(os.path.join(os.path.dirname(__file__), ))
    folder_path_input_mod = os.path.join(current_dir_i_mod, 'data', 'training_data',)
    file_name_input_mod = 'Input.txt'  # Debugger variables
    file_path_input_mod = os.path.join(folder_path_input_mod, file_name_input_mod)
    events_hoster.write_event("event_RanSulfurViaAPI")

    try:
        with open(file_path_input_mod, "w", encoding="utf-8", errors="ignore") as file_mod:
            file_mod.write(input_string)
    except TypeError:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"Input_string must be a *string*!")


    old_cwd = os.getcwd()
    try:
        os.chdir(current_dir_i_mod)  # Switch working directory to SulfurAI root

        start_time = datetime.now()
        start_time_printed = start_time.strftime("%Y-%m-%d %H:%M:%S")
        start_time_ms = f".{start_time.microsecond // 1000:03d}"

        import json
        returned = _rest_of_the_script("None", True, start_time,"[]", add_to_training_data)

        try:
            returned_final = json.loads(returned)
        except TypeError as e:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"JSON decode failed — input was None")

        return returned_final

    except (NameError, TypeError, FileNotFoundError, IOError, ValueError, AttributeError) as e:
        except_host.handle_sulfur_exception(e, call)

    finally:
        os.chdir(old_cwd)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_output_data(strip_newline_marker=False):
    """
    Returns the output.txt content as a list.
    Useful to save code and efficiency.

    """
    return api_get_output_data.get_output_data(strip_newline_marker=False)


def get_output_data_ui(strip_newline_marker=False):
    """
    Returns the output_userinsight.txt content as a list.
    Useful to save code and efficiency.

    Extra arguments:

     strip_newline_marker = False/True [DEFAULT: False]

     -Adds whether to include the '\n' tag to each item in the return statement.

    """
    return api_get_output_data_ui.get_output_data_ui(strip_newline_marker=False)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#make it override the current process when a new input is added [] - custom variable decider!


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#|                                                                                                                                                                        |
#|                                                      S E R V E R     A P I   M O D U L E                                                                               |
#|                                                                                                                                                                        |
#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

_server_singleton = None
class server():
    """
    Hybrid HTTP + legacy-file-queue SulfurAI server.
    - Preserves legacy file layout: <project>/api/Python/SulfurServerSystem/cache/id_*/output.txt
    - Preserves legacy queue file: call.api_server_python_cache_input()
    - Provides HTTP endpoints: POST /input, GET /output, GET /wait_output, DELETE /cache

    > [!IMPORTANT!]
    > The SulfurServerSystem API does *not* support deep-thinking (aka profile rendering) mode.
    """
    def __init__(self, host="127.0.0.1", port=8000):
        # imports & attachments (so class remains self-contained)
        from flask import Flask, request, jsonify, has_request_context
        from queue import LifoQueue
        from threading import Thread, Lock
        import threading, time, os, json, ast, uuid

        self.Flask = Flask
        self.request = request
        self.jsonify = jsonify
        self.has_request_context = has_request_context
        self.LifoQueue = LifoQueue
        self.Thread = Thread
        self.Lock = Lock
        self.time = time
        self.os = os
        self.json = json
        self.ast = ast
        self.uuid = uuid

        # try to import SulfurError if present
        try:
            from scripts.ai_renderer_sentences.error import SulfurError
            self.SulfurError = SulfurError
        except Exception:
            class SulfurError(Exception):
                pass
            self.SulfurError = SulfurError

        # module-level run_locally fallback
        self.run_locally_fn = globals().get("run_locally", None)

        # call object (original code uses call.api_server_python_cache_input())
        try:
            # call() factory exists earlier in file (from your original script)
            call_obj = globals().get("call", None)
            self.call = call_obj
            self.legacy_queue_file = call_obj.api_server_python_cache_input() if (call_obj and callable(getattr(call_obj, "api_server_python_cache_input", None))) else None
        except Exception:
            self.legacy_queue_file = None

        # base paths (preserve original structure)
        current_dir = self.os.path.abspath(self.os.path.join(self.os.path.dirname(__file__), ))
        self.current_dir = current_dir
        self.cache_base = self.os.path.join(current_dir, "api", "Python", "SulfurServerSystem", "cache")
        self.input_txt_global = self.os.path.join(current_dir, "data", "training_data", "Input.txt")

        # ensure cache dir exists
        self.os.makedirs(self.cache_base, exist_ok=True)

        # HTTP app
        self.app = self.Flask(__name__)
        self.host = host
        self.port = port

        # in-memory LIFO queue + bookkeeping
        self.queue = self.LifoQueue()
        self.id_lock = self.Lock()
        self._local_id_counter = 0

        # Thread control
        self._stop_event = threading.Event()

        # start worker thread
        w = self.Thread(target=self._worker_loop, daemon=True)
        w.start()
        self._worker_thread = w
        self.last_processed_id = None

        # start legacy queue watcher thread (if legacy queue path known)
        if self.legacy_queue_file:
            fw = self.Thread(target=self._file_watcher_loop, daemon=True)
            fw.start()
            self._file_watcher_thread = fw

        # bind endpoints
        self.app.add_url_rule("/input", "add_input_endpoint", self.add_input_endpoint, methods=["POST"])
        self.app.add_url_rule("/output", "get_output_endpoint", self.get_output_endpoint, methods=["GET"])
        self.app.add_url_rule("/wait_output", "wait_for_output_endpoint", self.wait_for_output_endpoint, methods=["GET"])
        self.app.add_url_rule("/cache", "clear_local_endpoint_cache", self.clear_local_endpoint_cache, methods=["DELETE"])

    # ---------------- helpers ----------------
    @staticmethod
    def get():
        global _server_singleton
        if _server_singleton is None:
            _server_singleton = server()
        return _server_singleton

    def _make_new_id(self):
        with self.id_lock:
            self._local_id_counter += 1
            counter = self._local_id_counter
        ts_ms = int(self.time.time() * 1000)
        short = self.uuid.uuid4().hex[:6]
        return f"id_{ts_ms}_{counter}_{short}"

    def _enqueue(self, task_id, input_string, legacy_append=False):
        """Put into in-memory LIFO queue and optionally append to legacy file."""
        # push tuple for worker
        self.queue.put((task_id, input_string))
        # optionally append to legacy queue file for external tools compatibility
        if legacy_append and self.legacy_queue_file:
            try:
                with open(self.legacy_queue_file, "a", encoding="utf-8", errors="ignore") as fh:
                    fh.write(input_string + "\n")
            except Exception:
                pass

    def _extract_input(self, input_string):
        # If running under HTTP, use request.json
        from flask import request
        if self.has_request_context():
            if request.json and "input_string" in request.json:
                return request.json["input_string"]
        # If called directly in Python, accept the argument
        return input_string

    # ---------------- legacy file watcher ----------------
    def _file_watcher_loop(self):
        """
        Watches legacy queue file and processes NEW lines only.
        After reading lines, the file is TRUNCATED so they are not re-read.
        Prevents infinite id folder creation.
        """
        path = self.legacy_queue_file

        while not self._stop_event.is_set():
            try:
                # If file doesn't exist yet, skip.
                if not self.os.path.exists(path):
                    self.time.sleep(0.5)
                    continue

                # Read ALL current lines ONCE.
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [line.strip() for line in fh.readlines() if line.strip()]

                # If empty, nothing to do.
                if not lines:
                    self.time.sleep(0.5)
                    continue

                # --- CRITICAL FIX ---
                # Immediately clear the file so lines aren't re-read.
                with open(path, "w", encoding="utf-8", errors="ignore") as clear:
                    clear.write("")
                # ---------------------

                # Process each line EXACTLY ONE TIME.
                for line in lines:
                    task_id = self._make_new_id()
                    folder = self.os.path.join(self.cache_base, task_id)

                    try:
                        # Create folder + status
                        self.os.makedirs(folder, exist_ok=True)
                        with open(self.os.path.join(folder, "input.txt"), "w", encoding="utf-8") as ff:
                            ff.write(line)
                        with open(self.os.path.join(folder, "status.json"), "w", encoding="utf-8") as sf:
                            self.json.dump({"status": "queued", "id": task_id}, sf)
                    except Exception:
                        pass

                    # Push to in-memory LIFO queue (important!)
                    self.queue.put((task_id, line))

            except Exception:
                # never crash; just delay and continue
                self.time.sleep(0.5)

            self.time.sleep(0.5)

    def _worker_loop(self):
        """
        Single self-contained worker loop. Includes an internal helper to append logs into
        status.log and keep a structured 'log' array inside status.json (bounded).
        Replace your existing _worker_loop with this function body.
        """
        import os, time, json, tempfile, traceback

        # Config
        MAX_LOG_ENTRIES = 200
        debug_log = os.path.join(tempfile.gettempdir(), "sulfur_worker_debug.log")

        def _debug_write(s):
            try:
                with open(debug_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.time():.3f} {s}\n")
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except Exception:
                        pass
            except Exception:
                pass

        # Internal helper: append plain-line log and also update status.json['log'] atomically
        def append_status_log_and_line(job_dir, task_id, status_line):
            """
            Append status_line (string) to job_dir/status.log and also append a structured
            entry to job_dir/status.json['log'] (timestamped). Keeps last MAX_LOG_ENTRIES entries.
            """
            try:
                os.makedirs(job_dir, exist_ok=True)
            except Exception:
                # if we cannot create job dir, still attempt to record debug
                _debug_write(f"[append_status] mkdir failed for {job_dir}")

            # 1) append plain-line to status.log (fast)
            try:
                log_path = os.path.join(job_dir, "status.log")
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(status_line + "\n")
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except Exception:
                        pass
            except Exception as e:
                _debug_write(f"[append_status] append log failed: {e}")

            # 2) update status.json with a bounded 'log' array atomically
            status_path = os.path.join(job_dir, "status.json")
            tmp_path = status_path + ".tmp"
            try:
                status_obj = {}
                # if existing status.json present, try to load and preserve fields
                if os.path.exists(status_path):
                    try:
                        with open(status_path, "r", encoding="utf-8") as fh:
                            status_obj = json.load(fh) or {}
                    except Exception:
                        status_obj = {"status": status_obj.get("status", "processing"), "id": task_id}

                # ensure minimal fields
                status_obj.setdefault("id", task_id)
                status_obj.setdefault("status", status_obj.get("status", "processing"))
                log_arr = status_obj.get("log", [])
                if not isinstance(log_arr, list):
                    log_arr = list(log_arr) if log_arr is not None else []

                entry = {"ts": int(time.time()), "msg": status_line}
                log_arr.append(entry)
                # keep bounded
                if len(log_arr) > MAX_LOG_ENTRIES:
                    log_arr = log_arr[-MAX_LOG_ENTRIES:]
                status_obj["log"] = log_arr

                # atomic write
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(status_obj, fh, indent=2, ensure_ascii=False)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except Exception:
                        pass
                try:
                    os.replace(tmp_path, status_path)
                except Exception:
                    # best-effort replacement
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

            except Exception as e:
                _debug_write(f"[append_status] status.json update failed: {e}")
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        # ---------- worker main loop ----------
        while True:
            try:
                task = self.queue.get(block=True, timeout=None)
            except Exception:
                # resilient - tiny sleep and continue
                time.sleep(0.05)
                continue

            try:
                # normalize task shape
                try:
                    task_id, input_string, legacy_append = task
                except Exception:
                    if isinstance(task, (list, tuple)) and len(task) >= 2:
                        task_id = task[0]
                        input_string = task[1]
                        legacy_append = False
                    else:
                        task_id = getattr(task, "id", f"id_{int(time.time() * 1000)}")
                        input_string = str(task)
                        legacy_append = False

                # compute job paths
                job_dir = os.path.join(self.cache_base, task_id) if hasattr(self, "cache_base") else None
                if job_dir:
                    try:
                        os.makedirs(job_dir, exist_ok=True)
                    except Exception as e:
                        _debug_write(f"[WORKER] mkdir failed for {job_dir}: {e}")

                # log processing start
                try:
                    if job_dir:
                        append_status_log_and_line(job_dir, task_id, f"> {task_id} processing started")
                    _debug_write(f"[WORKER] start id={task_id} pid={os.getpid()} cwd={os.getcwd()}")
                except Exception:
                    pass

                # call run_locally safely
                result = None
                try:
                    if callable(self.run_locally_fn):
                        try:
                            if job_dir:
                                append_status_log_and_line(job_dir, task_id, f"> {task_id} calling run_locally")
                            start_t = time.time()
                            result = self.run_locally_fn(input_string, True)
                            elapsed = time.time() - start_t
                            if job_dir:
                                append_status_log_and_line(job_dir, task_id,
                                                           f"> {task_id} run_locally done elapsed={elapsed:.2f}s")
                            _debug_write(f"[WORKER] run_locally returned id={task_id} elapsed={elapsed:.2f}s")
                        except Exception as exc:
                            tb = traceback.format_exc()
                            _debug_write(f"[WORKER] run_locally EXC id={task_id}: {repr(exc)}")
                            _debug_write(tb)
                            if job_dir:
                                append_status_log_and_line(job_dir, task_id,
                                                           f"> {task_id} run_locally EXC: {repr(exc)}")
                                append_status_log_and_line(job_dir, task_id,
                                                           f"> {task_id} traceback written to debug log")
                            result = {"ERROR": "run_locally raised exception", "exception": str(exc), "traceback": tb}
                    else:
                        result = {"ERROR": "no run_locally function available", "input": input_string}
                except Exception as outer_exc:
                    tb = traceback.format_exc()
                    _debug_write(f"[WORKER] unexpected EXC id={task_id}: {outer_exc}")
                    _debug_write(tb)
                    if job_dir:
                        append_status_log_and_line(job_dir, task_id, f"> {task_id} unexpected EXC: {outer_exc}")
                    result = {"ERROR": "unexpected worker exception", "exception": str(outer_exc), "traceback": tb}

                # normalize result
                normalized = None
                try:
                    if isinstance(result, dict):
                        normalized = result
                    elif isinstance(result, str):
                        try:
                            normalized = json.loads(result)
                        except Exception:
                            normalized = {"RAW_RETURN": result}
                    elif result is None:
                        normalized = {"ERROR": "run_locally returned None"}
                    else:
                        try:
                            normalized = json.loads(str(result))
                        except Exception:
                            normalized = {"RAW_RETURN": str(result)}
                except Exception as e:
                    tb = traceback.format_exc()
                    _debug_write(f"[WORKER] normalization EXC id={task_id}: {e}")
                    _debug_write(tb)
                    normalized = {"ERROR": "normalization failed", "exception": str(e), "traceback": tb}

                # write outputs
                try:
                    out_json_path = os.path.join(job_dir, "output.json") if job_dir else None
                    out_txt_path = os.path.join(job_dir, "output.txt") if job_dir else None
                    status_path = os.path.join(job_dir, "status.json") if job_dir else None

                    if job_dir and not os.path.exists(job_dir):
                        os.makedirs(job_dir, exist_ok=True)

                    # atomic write output.json
                    if out_json_path:
                        try:
                            tmp_json = out_json_path + ".tmp"
                            with open(tmp_json, "w", encoding="utf-8") as fh:
                                json.dump(normalized, fh, indent=2, ensure_ascii=False)
                                fh.flush()
                                try:
                                    os.fsync(fh.fileno())
                                except Exception:
                                    pass
                            try:
                                os.replace(tmp_json, out_json_path)
                            except Exception:
                                if os.path.exists(tmp_json):
                                    os.remove(tmp_json)
                        except Exception as e:
                            _debug_write(f"[WORKER] writing output.json failed id={task_id}: {e}")
                            if job_dir:
                                append_status_log_and_line(job_dir, task_id,
                                                           f"> {task_id} writing output.json failed: {e}")

                    # write human-readable output.txt
                    if out_txt_path:
                        try:
                            with open(out_txt_path, "w", encoding="utf-8") as ftxt:
                                if isinstance(normalized, dict) and "RAW_RETURN" in normalized:
                                    ftxt.write(str(normalized["RAW_RETURN"]))
                                elif isinstance(normalized, dict) and (
                                        "OUTPUT_TEXT" in normalized or "output" in normalized):
                                    out_text = normalized.get("OUTPUT_TEXT") or normalized.get("output") or str(
                                        normalized)
                                    ftxt.write(str(out_text))
                                else:
                                    ftxt.write(json.dumps(normalized, ensure_ascii=False, indent=2))
                        except Exception as e:
                            _debug_write(f"[WORKER] writing output.txt failed id={task_id}: {e}")
                            if job_dir:
                                append_status_log_and_line(job_dir, task_id,
                                                           f"> {task_id} writing output.txt failed: {e}")

                    # update final status.json (merge and keep previous log if any)
                    try:
                        status_obj = {"status": "finished", "id": task_id, "finished_at": int(time.time())}
                        if isinstance(normalized, dict) and normalized.get("ERROR"):
                            status_obj["status"] = "error"
                            status_obj["error_summary"] = normalized.get("ERROR") or normalized.get("exception")

                        # preserve previous log entries
                        if status_path and os.path.exists(status_path):
                            try:
                                with open(status_path, "r", encoding="utf-8") as fh:
                                    prev = json.load(fh) or {}
                            except Exception:
                                prev = {}
                        else:
                            prev = {}

                        status_obj["log"] = prev.get("log", []) if isinstance(prev.get("log", []), list) else []
                        # keep bounded
                        if len(status_obj["log"]) > MAX_LOG_ENTRIES:
                            status_obj["log"] = status_obj["log"][-MAX_LOG_ENTRIES:]

                        # write atomically
                        if status_path:
                            tmp_status = status_path + ".tmp"
                            try:
                                with open(tmp_status, "w", encoding="utf-8") as fh:
                                    json.dump(status_obj, fh, indent=2, ensure_ascii=False)
                                    fh.flush()
                                    try:
                                        os.fsync(fh.fileno())
                                    except Exception:
                                        pass
                                try:
                                    os.replace(tmp_status, status_path)
                                except Exception:
                                    if os.path.exists(tmp_status):
                                        os.remove(tmp_status)
                            except Exception as e:
                                _debug_write(f"[WORKER] writing final status.json failed id={task_id}: {e}")
                                if job_dir:
                                    append_status_log_and_line(job_dir, task_id,
                                                               f"> {task_id} writing final status.json failed: {e}")
                    except Exception as e:
                        _debug_write(f"[WORKER] status object creation failed id={task_id}: {e}")
                        if job_dir:
                            append_status_log_and_line(job_dir, task_id, f"> {task_id} status object failed: {e}")

                    # notify finished and record
                    try:
                        done_msg = f"[WORKER FINISHED] id={task_id} folder={job_dir}"
                        try:
                            print(done_msg, flush=True)
                        except Exception:
                            pass
                        if job_dir:
                            append_status_log_and_line(job_dir, task_id, f"> {task_id} finished")
                        _debug_write(done_msg)
                        try:
                            self.last_processed_id = task_id
                        except Exception:
                            pass
                    except Exception as e:
                        _debug_write(f"[WORKER] post-finish notify failed id={task_id}: {e}")

                except Exception as e:
                    tb = traceback.format_exc()
                    _debug_write(f"[WORKER] writing outputs EXC id={task_id}: {e}")
                    _debug_write(tb)
                    if job_dir:
                        append_status_log_and_line(job_dir, task_id, f"> {task_id} writing outputs EXC: {e}")

            finally:
                try:
                    try:
                        self.queue.task_done()
                    except Exception:
                        pass
                except Exception:
                    pass

            # yield
            time.sleep(0.01)

    # ---------------- endpoints ----------------
    def add_input_endpoint(self, input_string: str = None):
        """
        POST /input
        Accepts JSON { "input_string": "..." } or plain text body.
        Returns {"status":"queued","id": "<id>"}.
        """
        # accept JSON or form or raw body:
        try:
            data = None
            if self.has_request_context():
                data = self.request.get_json(silent=True)
            if data and isinstance(data, dict) and "input_string" in data:
                input_string = data["input_string"]
            elif self.has_request_context() and not input_string:
                # maybe raw body or form
                input_string = (self.request.form.get("input_string") or (self.request.data.decode("utf-8") if self.request.data else None))
        except Exception:
            pass

        if input_string is None:
            return self.jsonify({"error":"input_string is required"}), 400
        if not isinstance(input_string, str):
            return self.jsonify({"error":"input_string must be a string"}), 400

        # create id, folder, enqueue (LIFO). Also append to legacy queue file so external tools see it.
        task_id = self._make_new_id()
        folder = self.os.path.join(self.cache_base, task_id)
        try:
            self.os.makedirs(folder, exist_ok=True)
            with open(self.os.path.join(folder, "input.txt"), "w", encoding="utf-8", errors="ignore") as inf:
                inf.write(input_string)
            with open(self.os.path.join(folder, "status.json"), "w", encoding="utf-8", errors="ignore") as sf:
                self.json.dump({"status":"queued","id":task_id,"queued_at":int(self.time.time())}, sf)
        except Exception:
            pass

        # enqueue and append to legacy queue file for compatibility
        self._enqueue(task_id, input_string, legacy_append=True)

        if self.has_request_context():
            return self.jsonify({"status": "queued", "id": task_id}), 200
        else:
            return {"status": "queued", "id": task_id}, 200

    def get_output_endpoint(self, input_string: str = None, id: str = None):
        """
        GET /output?id=... (preferred)
        GET /output?input_string=... (fallback)
        """

        # When inside Flask request
        if self.has_request_context():
            id = self.request.args.get("id", None, type=str)
            input_string = self.request.args.get("input_string", None, type=str)

        # Lookup by id
        if id:
            folder = self.os.path.join(self.cache_base, id)

            if self.os.path.isdir(folder):
                out_json = self.os.path.join(folder, "output.json")
                out_txt = self.os.path.join(folder, "output.txt")

                # ---- output.json EXISTS ----
                if self.os.path.exists(out_json):
                    with open(out_json, "r", encoding="utf-8") as f:
                        data = self.json.load(f)

                    if self.has_request_context():
                        return self.jsonify(data), 200
                    else:
                        return data

                # ---- output.txt EXISTS ----
                if self.os.path.exists(out_txt):
                    with open(out_txt, "r", encoding="utf-8") as f:
                        txt = f.read()

                    # Try parse JSON
                    try:
                        data = self.json.loads(txt)
                    except Exception:
                        data = {"raw": txt}

                    if self.has_request_context():
                        return self.jsonify(data), 200
                    else:
                        return data

            # No output found
            if self.has_request_context():
                return self.jsonify({"error": "Output not found", "id": id}), 404
            else:
                return {"error": "Output not found", "id": id}, 404

    def wait_for_output_endpoint(self, input_string: str = None, id: str = None, timeout: float = None, max_timeout: float = None):
        """
        GET /wait_output?id=... OR /wait_output?input_string=...
        Polls until output.txt/output.json exists or times out.
        """
        if self.has_request_context():
            id = self.request.args.get("id", None, type=str)
            input_string = self.request.args.get("input_string", None, type=str)
            timeout = self.request.args.get("timeout", default=1.0, type=float)
            max_timeout = self.request.args.get("max_timeout", default=20.0, type=float)
        else:
            timeout = float(timeout) if timeout is not None else 1.0
            max_timeout = float(max_timeout) if max_timeout is not None else 20.0

        start = self.time.time()
        while True:
            # by id
            if id:
                folder = self.os.path.join(self.cache_base, id)
                if self.os.path.exists(self.os.path.join(folder, "output.json")) or self.os.path.exists(self.os.path.join(folder, "output.txt")):
                    return self.jsonify({"status":"found","id":id}), 200
            else:
                if input_string:
                    for folder_name in self.os.listdir(self.cache_base):
                        subdir = self.os.path.join(self.cache_base, folder_name)
                        if self.os.path.isdir(subdir):
                            out_json = self.os.path.join(subdir, "output.json")
                            if self.os.path.exists(out_json):
                                try:
                                    with open(out_json, "r", encoding="utf-8", errors="ignore") as f:
                                        data = self.json.load(f)
                                    if isinstance(data, dict) and data.get("INPUT_TEXT", [None])[0] == input_string:
                                        return self.jsonify({"status":"found","id":folder_name}), 200
                                except Exception:
                                    pass
                            # fallback input.txt match
                            inp = self.os.path.join(subdir, "input.txt")
                            if self.os.path.exists(inp):
                                try:
                                    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
                                        content = f.read()
                                    if content == input_string and (self.os.path.exists(out_json) or self.os.path.exists(os.path.join(subdir, "output.txt"))):
                                        return self.jsonify({"status":"found","id":folder_name}), 200
                                except Exception:
                                    pass
            if self.has_request_context():
                return self.jsonify({"status": "timeout"}), 202
            else:
                return {"status": "timeout"}, 202

    def clear_local_endpoint_cache(self):
        """
        DELETE /cache - removes id_* folders and clears legacy queue file.
        """
        import shutil
        for folder in self.os.listdir(self.cache_base):
            folder_path = self.os.path.join(self.cache_base, folder)
            if self.os.path.isdir(folder_path) and folder.startswith("id_"):
                try:
                    shutil.rmtree(folder_path)
                except Exception:
                    pass
        # clear legacy queue file if exists
        if self.legacy_queue_file and self.os.path.exists(self.legacy_queue_file):
            try:
                with open(self.legacy_queue_file, "w", encoding="utf-8", errors="ignore") as ff:
                    ff.writelines("")
            except Exception:
                pass
        return self.jsonify({"status":"cache cleared"}), 200

    def host_local_endpoint(self, host=None, port=None, reload=False):
        """
        Start Flask server (blocking).
        Use host/port args to override initialization ones.
        """
        h = host or self.host
        p = port or self.port
        # The app.run is blocking; call this in a thread if you want non-blocking.
        self.app.run(host=h, port=p, debug=reload)


#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#|                                                                                                                                                                        |
#|                                                                  A I  M O D E L S                                                                                      |
#|                                                                                                                                                                        |
#|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

def retrieve_current_models():
    """
    Returns the current AI models in the build.

    """
    events_hoster.write_event("event_APIRetrieveCurrentModels")
    from extra_models.Sulfur.Models.manager import find_models
    current_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
    ))
    target_directory = os.path.join(current_dir, "extra_models", "Sulfur", "Models")
    all_configs_names = find_models.list_all_models_in_directory(target_directory)
    return all_configs_names

def retrieve_active_model():
    """
    Returns the current AI model being used.

    [!NOTE]: Only shows the last used model, scripts using dynamic models will not be shown here.

    ----

    Returns:
        (2 vars)

        model: The active model name.


        timestamp: The timestamp of when the model was activated.

    """
    events_hoster.write_event("event_APIRetrieveActiveModel")
    from extra_models.Sulfur.Models.manager import find_models
    current_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
    ))
    result = find_models.get_active_model_from_file()
    return result or (None, None)

def retrieve_active_model_history():
    """
    Returns a list of dictionaries containing the history of active models used.

    Can be cleared via cache.

    """
    events_hoster.write_event("event_APIRetrieveActiveModelHistory")
    from extra_models.Sulfur.Models.manager import find_models
    current_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
    ))
    result = find_models.get_active_model_history()
    return result or None


@not_available
def add_model_config():
    """
    Adds a model configuration to the SulfurAI build.

      ------------------------------


        ``WIP. DO NOT USE.``
    -------------------------------

    Arguments:

        none


    -------------------------------
    """
    pass

def render_dataprofile(USE_API: str = False,
                       API_MODEL: str = "gemini-2.5-flash",
                       API_KEY: str = ""):

    #do not f*cking change the api code i swear to the lord

    """
    Uses large-scale LLM models to generate a data profile of the input data — the end SulfurAI goal.

    When running, SulfurAI will switch to a "deep think" mode which consumes more resources and time.

    [SUPPORTED MODELS]: 'gemini-2.5-flash'

    ----

    Arguments:
        USE_API: uses a selected model to render instead of local hardware; requires API_KEY.
        API_KEY: provides the API key for USE_API.


    ----

    Returns:

        (see below)

        ----

        IF NOT USING API:

            offline_dict: a dictionary containing AI-assumed variables.
            summary_dict: a dictionary containing the AI summary and annotations.

        ----

        IF USING API:

            A single dictionary structured as:
            {"id": str, "object": "api_response", "created": int, "usage": {"summary_characters": int, "timestamp": str}, "data": {"ADVANCED_SUMMARY": str, "NORMAL_SUMMARY": str}}

            Any logged errors (if present) may also be included or returned with this dictionary.
    """
    if not USE_API and not API_KEY == "":
        print("[TRACEBACK]: Script must not provide API_KEY when USE_API is set to 'False'.")
        time.sleep(10)
        exit()
    if USE_API and API_KEY == "":
        print("[TRACEBACK]: Script must provide API_KEY when USE_API is set to 'True'.")
        time.sleep(10)
        exit()
    try:
        from extra_models.Sulfur.RenderScript.Build import render_scripts_general
        if not USE_API:
            offline_dict, summary_dict = render_scripts_general.api_render_dp(API_KEY,USE_API,API_MODEL)
            return offline_dict, summary_dict
        if USE_API:
            return_ = render_scripts_general.api_render_dp(API_KEY, USE_API, API_MODEL)
            return return_
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"{e}")


def return_dataprofiles():
    """
    Returns two types of profiles: `API` and `MANUAL_RENDERS`.

    Automatically detects the type of profile and returns the appropriate structure.
    Returns another variable "API" or "MANUAL_RENDER" to indicate which type was returned.

    ----

    API Profiles:
        > Returns a dictionary containing:
            >> id

            >> object

            >> created

            >> usage

            >> data (containing): ADVANCED_SUMMARY, NORMAL_SUMMARY

    ----

    MANUAL_RENDER Profiles:
        > Returns a dictionary containing:

            >> advanced_data_profile

            >> raw_advanced_summary

            >> normal_profile_summary

            >> other_sections
    """
    try:
        from returns.dataprofiles.scripts import run_profile_builder
        path = call.profile_default()
        dict,type = run_profile_builder.load_split_profiles(file_path=path)
        return dict,type
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"{e}")




def return_offline_dataprofiles():
    """
    [!WILL NOT WORK ON API-RENDERED PROFILES!]



    Extract the block under a heading like '==== OFFLINE PROFILE JSON ====' and attempt to parse it as JSON.

    Returns a tuple: (parsed_json_or_None, raw_captured_text).
      - If parsing succeeds, parsed_json_or_None is the parsed object (dict/list) and raw text is the captured raw block.
      - If parsing fails or section missing, parsed_json_or_None is None and second return is "" or the captured raw block.
    """
    try:
        from returns.dataprofiles.scripts import run_profile_builder
        parsed, raw_block = run_profile_builder.load_offline_profiles()
        return parsed, raw_block
    except Exception as e:
        print(f"[DEBUG:] Failed to render script:: return_offline_dataprofiles(), did you try to run an API-rendered profile?")
        print("[DEBUG//TRACEBACK:]", e)


def return_annotations_dataprofiles():
    """
        [!WILL NOT WORK ON API-RENDERED PROFILES!]


        Extract the text under a line matching 'AI Annotations:' (case-insensitive).
        Returns the captured block trimmed, or "" if not present.
    """
    try:
        from returns.dataprofiles.scripts import run_profile_builder
        raw_block = run_profile_builder.load_ai_annotations()
        return raw_block
    except Exception as e:
        print(f"[DEBUG:] Failed to render script:: return_offline_dataprofiles(), did you try to run an API-rendered profile?")
        print("[DEBUG//TRACEBACK:]", e)



















