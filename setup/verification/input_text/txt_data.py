
import re,os
too_long_id = False

def get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()

# Call file paths
call = get_call_file_path()


class Ensure():
    def __init__(self,max_length=50): #modify this via settings?
        file_path_input_limit = call.input_limit()
        try:
            with open(file_path_input_limit, 'r', encoding='utf-8', errors='ignore') as file:  max = int(file.readline().strip())
        except Exception as e:
            print(f"{e} occurred while running input verifier.")
            max = 50
        self.max_length = max


    def strip_input(self,input):

        input_text = input.strip()
        input_text = re.sub(r"[\'\";\\&<>%^\s,]+", "", input_text)
        if not input_text:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"Invalid username: empty after cleaning.")
        return input_text

    def strip_input_list(self, input_list):

        input_list_text = input_list.strip()
        input_list_text = re.sub(r"[\'\";\\&<>%^,]+", "", input_list_text)
        if not input_list_text:
            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"Invalid username: empty after cleaning.")
        return [input_list_text]


    def clear_length(self, input_list_text):
        joined_text = ""
        file_path_settings_name_input_process_limit = call.settings_input_process_limit()
        with open(file_path_settings_name_input_process_limit, 'r', encoding='utf-8', errors='ignore') as file:  limit_process_length = file.readline().strip()
        too_long = False
        if limit_process_length == "yes":
             if isinstance(input_list_text, list):
                joined_text = ",".join(input_list_text)
             else:
                joined_text = input_list_text
        elif limit_process_length == "no":
            joined_text = input_list_text



        if len(joined_text) > self.max_length:
            input_list_text = joined_text[:self.max_length]  # Truncate the joined string
            too_long = True


        return too_long, input_list_text



    def check_re_sub(self,input,string_or_list):
        re_was_subbed = False
        if string_or_list == "string":
            input_text = input.strip()
            input_text_modified = re.sub(r"[\'\";\\&<>%^\s,]+", "", input_text)
            re_was_subbed = True if input_text != input_text_modified else False
        elif string_or_list == "list":
            input_text = input.strip()
            input_text_modified = re.sub(r"[\'\";\\&<>%^,]+", "", input_text)
            re_was_subbed = True if input_text != input_text_modified else False

        return re_was_subbed




def call_ensure():
    return Ensure()

def verify_input(string_or_list):
    user_data_verified = ""
    too_long_id_list = False
    too_long_id = False
    too_long = False
    re_was_subbed = False
    user_data_input = ""
    current_dir_i = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))
    folder_path_input = os.path.join(current_dir_i, 'data', 'training_data')
    file_name_input = 'Input.txt'
    file_path_input = os.path.join(folder_path_input, file_name_input)
    with open(file_path_input, 'r', encoding='utf-8', errors='ignore') as file:  input_data = [line.strip() for line in file.readlines() if line.strip()]

    ensure = call_ensure()
    if string_or_list.lower() == "string":
        user_data_input = ensure.strip_input(",".join(input_data))
        too_long, user_data_verified = ensure.clear_length(user_data_input)
        re_was_subbed = ensure.check_re_sub(",".join(input_data),"string")
    elif string_or_list.lower() == "list":
        user_data_input = ensure.strip_input_list(",".join(input_data))
        too_long, user_data_verified = ensure.clear_length(user_data_input)
        re_was_subbed = ensure.check_re_sub(",".join(input_data), "list")


    return user_data_verified,too_long,re_was_subbed

