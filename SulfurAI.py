
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
#Certain render scripts have been moved to 'VersionFiles/Sulfur/RenderScript/Build'.
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


class server():
    """
    The SulfurAI Server System (SSS) is a server system that runs SulfurAI in parallel.
    To find out more, see the ``developer SDK``.
    """
    def __init__(self):
        self._mngr_wn()

    def _mngr_wn(self):
        server_manager_warning = ["|----------------------------------|",
                                  "|    SULFUR SERVER SYSTEM    ",
                                  "|        ",
                                  "|   ⚠️ Certain SSS functions may require an active internet connection.     ",
                                  "|----------------------------------|",
                                  "|        ",
                                  "|    📖 INITIATING SSS....    ",
                                  "|----------------------------------|",

                                  ]
        for item in server_manager_warning: print(item)

    @classmethod
    def host_local_endpoint(self,timeout_max=90,delete_cache=True,add_to_training_data=True,setup_local_auto=True,priority_processing=False):
        #############Hosts a local SulfurAI endpoint API server (SSS).

        """
        Hosts a local SulfurAI endpoint API server (SSS).
        Allows you to run SulfurAI in a loop and add as many inputs as wanted.

        ------------------------------

        [! WARNING !]

            Runs SulfurAI in a shadow instance, meaning it replaces your current inputs and processes the local SulfurScript.

            *DO NOT RUN IN AN ONLINE SERVER. THIS COULD LEAK DATA!*

        -------------------------------

        Main arguments:
            ``timeout_max`` = [int] [DEFAULT: 90]
         - after the timeout_max seconds, if the current render is not finished - the script will stop processing it and move on.
         - [!] Warning, this does not guarantee the script will stop processing. In BETA testing!

          ``delete_cache`` = True/False [DEFAULT: True]
         - decides whether to delete all cache after the script is closed.

          ``setup_local_auto`` = True/False [DEFAULT: True]
         - decides whether to automatically set up the local environment for SulfurAI.
         - (disabling this will mean you must manually set up the env with 'SulfurAI.setup_local()')

        -------------------------------

        Extra arguments:
            ``add_to_training_data`` = True/False [DEFAULT: True]
         - decides whether to add this input to the SulfurAI training data

            ``priority_processing`` = True/False [DEFAULT: False]
         - decides whether to use LIFO (Last in, First out) processing for inputs. Means latest inputs will be processed after the current input.


        -------------------------------



        """
        from func_timeout import func_timeout, FunctionTimedOut
        server()
        events_hoster.write_event("event_SetupServerAPI")
        if setup_local_auto: setup_local()
        current_dir_i_mod = os.path.abspath(os.path.join(os.path.dirname(__file__), ))
        folder_path_input_mod = os.path.join(current_dir_i_mod, 'data', 'training_data',)
        file_name_input_mod = 'Input.txt'  # Debugger variables
        file_path_input_mod = os.path.join(folder_path_input_mod, file_name_input_mod)

        def delete_first_line(file_path):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if len(lines) > 1:
                    with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                        f.writelines(lines[1:])
                else:
                    # If there's only one line or it's empty, clear the file
                    open(file_path, "w").close()
            except FileNotFoundError:
                pass  # Ignore if file doesn't exist

        def priority_process(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                #if len(lines) < 2:

                    #return None

                last_line = lines.pop()  # remove the last line
                lines.insert(0, last_line)  # insert it at the top

                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(lines)


                return last_line.strip('\n')

            except FileNotFoundError:

                wim = error.who_imported_me()
                error.error("er1", "SCRIPT_TRACEBACK", f"File not found: {file_path} issue at imports: {wim}", "nAn")
                return None
            except Exception as e:
                wim = error.who_imported_me()
                error.error("er1", "SCRIPT_TRACEBACK", f"Error: {e} issue at imports: {wim}", "nAn")
                return None

        try:
            print("💻 Setting up local server endpoint via SSS (SulfurAI Server System)...")
            while True:
                try:
                    if priority_processing: print(f"--⚠️ Priority processing enabled. Latest inputs will be processed first with LIFO.")

                    file_path_input_api_server = call.api_server_python_cache_input()
                    print("--💻 Waiting for action from endpoint...")
                    while os.path.getsize(file_path_input_api_server) == 0:  time.sleep(0.1)
                    print("--💻 Commencing endpoint....")

                    if priority_processing:
                        priority_item = priority_process(file_path_input_api_server)
                        print(f"--💻 Priority item: {priority_item} took priority processing next.")

                    # After priority_process has rearranged the file, read the first line (the priority line)
                    with open(file_path_input_api_server, "r", encoding="utf-8", errors="ignore") as file_mod:
                        input_string = file_mod.readline()

                    delete_first_line(file_path_input_api_server)  # Delete the first line after reading it

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
                    current_dir = os.path.abspath(os.path.join(
                        os.path.dirname(__file__),
                    ))

                    base_path = os.path.join(current_dir, 'api', 'Python', 'SulfurServerSystem', 'cache')
                    os.makedirs(base_path, exist_ok=True)

                    # Count existing folders in base_path
                    existing_folders = [f for f in os.listdir(base_path)
                                        if os.path.isdir(os.path.join(base_path, f)) and f.startswith("id_")]

                    next_id = len(existing_folders) + 1
                    new_folder_name = f"id_{next_id}"
                    new_folder_path = os.path.join(base_path, new_folder_name)
                    os.makedirs(new_folder_path, exist_ok=False)

                    # Create text file inside the new folder
                    file_path = os.path.join(new_folder_path, "output.txt")

                    ####################---------------------rendering

                    def render():
                        result = _rest_of_the_script("None", True, start_time=start_time, output_file_path=file_path, add_to_training_data=add_to_training_data,
                                                     endpoint_custom=True)
                        print("🚶 Ran successful loop of endpoint.")
                        return result

                    try:
                        result = func_timeout(timeout_max, render)
                    except FunctionTimedOut:
                        print("⏱️ Server timed out. Could not process input, moving on to the next input.")








                except (NameError, TypeError, FileNotFoundError, IOError, ValueError, AttributeError) as e:
                    except_host.handle_sulfur_exception(e, call)

                finally:
                    os.chdir(old_cwd)
        except KeyboardInterrupt:
            if delete_cache:
                print("⚠️ Deleting cache..")
                current_dir = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                ))
                base_path = os.path.join(current_dir, 'api', 'Python', 'SulfurServerSystem', 'cache')
                import shutil

                if not os.path.exists(base_path):
                    print("Base path doesn't exist.")
                    return

                for folder in os.listdir(base_path):
                    folder_path = os.path.join(base_path, folder)
                    if os.path.isdir(folder_path) and folder.startswith("id_"):
                        try:
                            shutil.rmtree(folder_path)
                            print(f"Deleted: {folder_path}")
                        except Exception as e:
                            print(f"Failed to delete {folder_path}. Reason: {e}")
                file_path_input_api_server = call.api_server_python_cache_input()
                with open(file_path_input_api_server, "w", encoding="utf-8", errors="ignore") as file_mod: file_mod.writelines("")
                print(f"Deleted input cache.")


    @classmethod
    def add_input_endpoint(self,input_string,extra_debug=True):
        #############Adds input to the local SulfurAI endpoint API server (SSS).

        """
        Adds input to the local SulfurAI endpoint API server (SSS).


        -------------------------------

        Main arguments:
          ``input_string`` [str]
        - Adds an input to the server cache to be processed.

        -------------------------------

        Extra arguments:
            ``extra_debug`` = True/False [DEFAULT: True]
         - decides whether to print debug statements for this function.


        -------------------------------
        """

        if extra_debug: print(f"💻 Adding {str(input_string)} to the local endpoint via SSS (SulfurAI Server System)...")
        if type(input_string) is not str:
            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"Input_string must be a *string*!")
        file_path_input_api_server = call.api_server_python_cache_input()
        with open(file_path_input_api_server, "a", encoding="utf-8",errors="ignore") as file_mod:
            file_mod.write(input_string + "\n")



    @classmethod
    def get_output_endpoint(self,input_string,extra_debug=True):
        #############Gets a specific output from the local SulfurAI endpoint API server (SSS).

        """
        Gets a specific output from the local SulfurAI endpoint API server (SSS).


        -------------------------------

        Main arguments:
          ``input_string`` [str]
        - Takes an input string and finds it inside the server, returning the API output

        -------------------------------

        Extra arguments:
            ``extra_debug`` = True/False [DEFAULT: True]
         - decides whether to print debug statements for this function.

        -------------------------------

        Returns:
            - Returns a dictionary with the Sulfur API output.
            To see what the API would return, see the documentation of `SulfurAI.run_locally()`.




        -------------------------------
        """

        import ast,re
        if extra_debug: print(f"💻 Attempting to find {str(input_string)} in the local endpoint via SSS (SulfurAI Server System)...")
        if type(input_string) is not str:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"Input_string must be a *string*!")
        current_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
        ))
        base_path = os.path.join(current_dir, 'api', 'Python', 'SulfurServerSystem', 'cache')

        for folder_name in os.listdir(base_path):
            subfolder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(subfolder_path):

                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(".txt"):
                        txt_file_path = os.path.join(subfolder_path, file_name)


                        with open(txt_file_path, "r", encoding="utf-8", errors="ignore") as file: cached_file = file.read()

                        def clean_wrappers(text):
                            """Remove wrapper calls so ast.literal_eval can parse."""
                            # np.str_('x')  -> 'x'
                            text = re.sub(r"np\.str_\(\s*(['\"].*?['\"])\s*\)", r"\1", text, flags=re.DOTALL)

                            # np.float64(1.23) -> 1.23
                            text = re.sub(r"np\.float64\(\s*([0-9.+-eE]+)\s*\)", r"\1", text)

                            # Counter({'A':1}) -> {'A':1}
                            text = re.sub(r"Counter\(\s*(\{.*?\})\s*\)", r"\1", text, flags=re.DOTALL)

                            return text

                        cached_file = clean_wrappers(cached_file)
                        data_dict = ast.literal_eval(cached_file)
                        if data_dict["INPUT_TEXT"][0] == input_string:
                            print(f"💻 Found {str(input_string)} in the local endpoint via SSS (SulfurAI Server System)...")
                            return data_dict
                        else: pass
        print(f"⚠️ Could not find {str(input_string)} in the local endpoint via SSS (SulfurAI Server System).")
        print(f"⚠️ Check if the input was added correctly, if the server is running or if the input has been cached (processed) yet.")

    @classmethod
    def wait_for_output_endpoint(self,input_string,timeout=3,max_timeout=20,extra_debug=True):
        #############Waits until a specific output from the local SulfurAI endpoint API server (SSS) has been processed.
        #############Basically a re-purposed function of `get_output_endpoint()` that allows that function to be used in the same script that added the input to the server.

        """
        Waits until a specific output from the local SulfurAI endpoint API server (SSS) has been processed.

          ------------------------------

        [! WARNING !]

            Does not return anything, it just waits until the output is processed.
            To get the output, use `SulfurAI.server.get_output_endpoint(input_string)`.



        -------------------------------

        -------------------------------

        Main arguments:
          input_string [str]
        - Takes an input string and finds it inside the server, returning the API output

        -------------------------------

        Extra arguments:
          ``timeout`` = [int] DEFAULT: 3
         - Waits a maximum of `timeout` seconds and checks if the output has been processed.

           ``max_timeout`` = [int] DEFAULT: 20
         - Waits a maximum of `max_timeout` seconds and checks if the output has been found, if not - it exits the script.

            ``extra_debug`` = True/False [DEFAULT: True]
         - decides whether to print debug statements for this function.


        -------------------------------
        """

        import ast,re
        if extra_debug: print(
            f"💻 Waiting to find {str(input_string)} in the local endpoint via SSS (SulfurAI Server System)...")
        print(f"📖 Max timeout is {max_timeout} seconds, timeout is {timeout} seconds.")

        if type(input_string) is not str:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"Input_string must be a *string*!")
        current_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
        ))
        base_path = os.path.join(current_dir, 'api', 'Python', 'SulfurServerSystem', 'cache')

        found = False
        start_time = time.time()  # record when we start

        while not found:
            for folder_name in os.listdir(base_path):
                subfolder_path = os.path.join(base_path, folder_name)
                if os.path.isdir(subfolder_path):
                    for file_name in os.listdir(subfolder_path):
                        if file_name.endswith(".txt"):
                            txt_file_path = os.path.join(subfolder_path, file_name)

                            with open(txt_file_path, "r", encoding="utf-8", errors="ignore") as file:
                                cached_file = file.read()

                            def clean_wrappers(text):
                                """Remove wrapper calls so ast.literal_eval can parse."""
                                # np.str_('x')  -> 'x'
                                text = re.sub(r"np\.str_\(\s*(['\"].*?['\"])\s*\)", r"\1", text, flags=re.DOTALL)

                                # np.float64(1.23) -> 1.23
                                text = re.sub(r"np\.float64\(\s*([0-9.+-eE]+)\s*\)", r"\1", text)

                                # Counter({'A':1}) -> {'A':1}
                                text = re.sub(r"Counter\(\s*(\{.*?\})\s*\)", r"\1", text, flags=re.DOTALL)

                                return text

                            cached_file = clean_wrappers(cached_file)
                            data_dict = ast.literal_eval(cached_file)
                            if data_dict["INPUT_TEXT"][0] == input_string:
                                print(
                                    f"💻 Found {str(input_string)} in the local endpoint via SSS (SulfurAI Server System)...")
                                print(
                                    f"📖 To return the API output, use `SulfurAI.server.get_output_endpoint(input_string)`.")
                                found = True
                                break
                    if found:
                        break

            if not found:
                elapsed = time.time() - start_time
                if elapsed >= max_timeout:
                    print(f"⚠️ Max timeout of {max_timeout} seconds reached without finding input.")
                    break
                time.sleep(timeout)

    @classmethod
    def clear_local_endpoint_cache(self,extra_debug=True):
        ############Clears the local SulfurAI endpoint API server (SSS) cache.

        """
        Clears the local SulfurAI endpoint API server (SSS) cache.

          ------------------------------

        [! WARNING !]

            SulfurAI.server.host_local_endpoint on default automatically clears the cache after the script is closed.
            Change this to false with `SulfurAI.server.host_local_endpoint(delete_cache=False)` to keep the cache.



        -------------------------------

        Extra arguments:

            ``extra_debug`` = True/False [DEFAULT: True]
         - decides whether to print debug statements for this function.


        -------------------------------
        """

        if extra_debug: print(
            f"💻 Deleting the local endpoint cache via SSS (SulfurAI Server System)...")
        print("⚠️ Deleting cache..")
        current_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
        ))
        base_path = os.path.join(current_dir, 'api', 'Python', 'SulfurServerSystem', 'cache')
        import shutil

        if not os.path.exists(base_path):
            print("Base path doesn't exist.")
            return

        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path) and folder.startswith("id_"):
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted: {folder_path}")
                except Exception as e:
                    print(f"Failed to delete {folder_path}. Reason: {e}")
        file_path_input_api_server = call.api_server_python_cache_input()
        with open(file_path_input_api_server, "w", encoding="utf-8", errors="ignore") as file_mod:
            file_mod.writelines("")
        print(f"Deleted input cache.")

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



















