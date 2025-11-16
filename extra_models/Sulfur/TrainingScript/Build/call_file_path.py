import os

current_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir,   # up from Build to TrainingScript
    os.pardir,   # up to Sulfur
    os.pardir,    # up to VersionFiles
    os.pardir,
))

class Call():


    def device(self):
        # Define the expected folder path for Output_Data
        folder_path_OutputData = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OutputData_name_Device = "Device.txt"

        # Construct the full path to Device.txt
        file_path_OutputData_name_Device = os.path.join(folder_path_OutputData, folder_path_OutputData_name_Device)
        return  file_path_OutputData_name_Device

    def device_accuracy(self):

        folder_path_OuputData = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_name_Device_accuracy = "Device_Accuracy.txt"
        file_path_OutputData_name_Device_accuracy = os.path.join(folder_path_OuputData, folder_path_OuputData_name_Device_accuracy)
        return folder_path_OuputData, folder_path_OuputData_name_Device_accuracy, file_path_OutputData_name_Device_accuracy

    def mean_device(self):
        folder_path_OuputData = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_name_mean_device = "Main_Device.txt"
        file_path_OutputData_name_mean_device = os.path.join(folder_path_OuputData,
                                                                 folder_path_OuputData_name_mean_device)
        return  file_path_OutputData_name_mean_device

    def average_device_accuracy(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_name_mean_device_average_device_accuracy = "Average_accuracy.txt"
        file_path_OutputData_name_mean_device_average_device_accuracy = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_name_mean_device_average_device_accuracy)
        return file_path_OutputData_name_mean_device_average_device_accuracy

    def response_time(self):
        folder_path_OuputData = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_name_Response_Time_MS = "Response_Time_MS.txt"
        file_path_OutputData_name_Response_Time_MS = os.path.join(folder_path_OuputData, folder_path_OuputData_name_Response_Time_MS)
        return folder_path_OuputData, folder_path_OuputData_name_Response_Time_MS, file_path_OutputData_name_Response_Time_MS


    def grammar(self):
        folder_path_trainingData_grammar = os.path.join(current_dir, 'scripts', 'ai_renderer',"training_data","grammar_func=checkdevice")
        folder_path_trainingData_grammar_name = "RepositlessGrammar.txt"
        file_path_trainingData_grammar = os.path.join(folder_path_trainingData_grammar,folder_path_trainingData_grammar_name)
        return folder_path_trainingData_grammar,folder_path_trainingData_grammar_name,file_path_trainingData_grammar

    def output(self):
        folder_path_output = os.path.join(current_dir, 'data', 'training_data')
        folder_path_output_name = ("Output.txt")
        file_path_output = os.path.join(folder_path_output, folder_path_output_name)
        return folder_path_output,folder_path_output_name,file_path_output

    def arch(self):
        global folder_path_archFullTimeRendererSulfax,folder_path_archFullTimeRendererSulfax_name,file_path_archFullTimeRendererSulfax
        folder_path_archFullTimeRendererSulfax = os.path.join(current_dir,"scripts","ai_renderer_sentences","Sulfax-Architechture","build 001-01")
        folder_path_archFullTimeRendererSulfax_name = ("archFullTimeRendererSulfax.txt")
        file_path_archFullTimeRendererSulfax = os.path.join(folder_path_archFullTimeRendererSulfax,folder_path_archFullTimeRendererSulfax_name)
        return folder_path_archFullTimeRendererSulfax,folder_path_archFullTimeRendererSulfax_name,file_path_archFullTimeRendererSulfax

    def arch_colon(self):
        global folder_path_colon_symbol, folder_path_colon_symbol, file_path_colon_symbol
        folder_path_colon_symbol = os.path.join(current_dir,"scripts","ai_renderer_sentences","Sulfax-Architechture","build 001-01")
        folder_path_colon_symbol_name = ("colon-symbol.txt")
        file_path_colon_symbol = os.path.join(folder_path_colon_symbol, folder_path_colon_symbol_name)
        return folder_path_colon_symbol, folder_path_colon_symbol_name, file_path_colon_symbol

    def training_data(self):
        global folder_path_training_data,folder_path_training_data_name,file_path_training_data
        folder_path_training_data = os.path.join(current_dir,"scripts","ai_renderer", 'training_data', "data_train")
        #folder_path_training_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'DATA/training_data/data_train')
        folder_path_training_data_name = ("data.txt")
        file_path_training_data = os.path.join(folder_path_training_data, folder_path_training_data_name)
        return folder_path_training_data,folder_path_training_data_name,file_path_training_data


    def training_data_sk(self):
        global folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk
        folder_path_training_data_sk = os.path.join(current_dir, "scripts","ai_renderer", 'training_data', "data_train_sk")
        #folder_path_training_data_sk = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 #'DATA/training_data/data_train_sk')
        folder_path_training_data_name_sk = ("data.txt")
        file_path_training_data_sk = os.path.join(folder_path_training_data_sk, folder_path_training_data_name_sk)
        return folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk

    def backup_training_data_sk(self):
        global folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk
        folder_path_training_data_sk_backup = os.path.join(current_dir, "scripts","ai_renderer", 'training_data', "BACKUP-DO-NOT-DELETE", "backup_sk")
        # folder_path_training_data_sk = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        # 'DATA/training_data/data_train_sk')
        folder_path_training_data_name_sk_backup = ("data.txt")
        file_path_training_data_sk_backup = os.path.join(folder_path_training_data_sk_backup, folder_path_training_data_name_sk_backup)
        return file_path_training_data_sk_backup

    def backup_training_data(self):
        global folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk
        folder_path_training_data_backup = os.path.join(current_dir, "scripts","ai_renderer", 'training_data', "BACKUP-DO-NOT-DELETE", "backup_normal")
        # folder_path_training_data_sk = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        # 'DATA/training_data/data_train_sk')
        folder_path_training_data_name_backup = ("data.txt")
        file_path_training_data_backup = os.path.join(folder_path_training_data_backup, folder_path_training_data_name_backup)
        return file_path_training_data_backup

    def settings_extra_debug(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_name_extra_debug = "extra_debug.txt"
        file_path_settings_name_extra_device = os.path.join(folder_path_settings,
                                                             folder_path_settings_name_extra_debug)
        return file_path_settings_name_extra_device

    def settings_backup(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_name_backup = "backup_valid.txt"
        file_path_settings_name_backup = os.path.join(folder_path_settings,
                                                            folder_path_settings_name_backup)
        return file_path_settings_name_backup

    def preferences_user(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_name_preferences_user = "Preferences_user.txt"
        file_path_OutputData_name_preferences_user = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_name_preferences_user)
        return file_path_OutputData_name_preferences_user

    def preferences_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_name_preferences_global = "Preferences_global.txt"
        file_path_OutputData_name_preferences_global = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_name_preferences_global)
        return file_path_OutputData_name_preferences_global

    def Wanted_noun_most_important_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Wanted_noun_most_important_global = "Wanted_noun_most_important_global.txt"
        file_path_OutputData_Wanted_noun_most_important_global = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Wanted_noun_most_important_global)
        return file_path_OutputData_Wanted_noun_most_important_global

    def Wanted_noun_most_important_user(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Wanted_noun_most_important_user = "Wanted_noun_most_important_user.txt"
        file_path_OutputData_Wanted_noun_most_important_user = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Wanted_noun_most_important_user)
        return file_path_OutputData_Wanted_noun_most_important_user

    def Wanted_verb_most_important_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Wanted_verb_most_important_global = "Wanted_verb_most_important_global.txt"
        file_path_OutputData_Wanted_verb_most_important_global = os.path.join(folder_path_OuputData,
                                                                              folder_path_OuputData_Wanted_verb_most_important_global)
        return file_path_OutputData_Wanted_verb_most_important_global

    def Wanted_verb_most_important_user(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Wanted_verb_most_important_user = "Wanted_verb_most_important_user.txt"
        file_path_OutputData_Wanted_verb_most_important_user = os.path.join(folder_path_OuputData,
                                                                            folder_path_OuputData_Wanted_verb_most_important_user)
        return file_path_OutputData_Wanted_verb_most_important_user

    def Describe_adjective_most_important_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Describe_adjective_most_important_global = "Describe_adjective_global.txt"
        file_path_OutputData_Describe_adjective_most_important_global = os.path.join(folder_path_OuputData,
                                                                              folder_path_OuputData_Describe_adjective_most_important_global)
        return file_path_OutputData_Describe_adjective_most_important_global

    def Describe_adjective_most_important_user(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Describe_adjective_user = "Describe_adjective_user.txt"
        file_path_OutputData_Describe_adjective_user = os.path.join(folder_path_OuputData,
                                                                            folder_path_OuputData_Describe_adjective_user)
        return file_path_OutputData_Describe_adjective_user

    def mood_user(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_mood_user = "Preferences_Mood_User.txt"
        file_path_OutputData_mood_user = os.path.join(folder_path_OuputData,
                                                                            folder_path_OuputData_mood_user)
        return file_path_OutputData_mood_user

    def mood_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_mood_global = "Preferences_Mood_Global.txt"
        file_path_OutputData_mood_global = os.path.join(folder_path_OuputData,
                                                      folder_path_OuputData_mood_global)
        return file_path_OutputData_mood_global

    def mood_accuracy_user(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_mood_accuracy = "Preferences_Mood_Accuracy_User.txt"
        file_path_OutputData_mood_accuracy_user = os.path.join(folder_path_OuputData,
                                                      folder_path_OuputData_mood_accuracy)
        return file_path_OutputData_mood_accuracy_user

    def mood_accuracy_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_mood_accuracy = "Preferences_Mood_Accuracy_Global.txt"
        file_path_OutputData_mood_accuracy_global = os.path.join(folder_path_OuputData,
                                                               folder_path_OuputData_mood_accuracy)
        return file_path_OutputData_mood_accuracy_global

    def text_data_verify(self):
        folder_path_verify = os.path.join(current_dir, 'setup', 'verification',"input_text")
        folder_path_text_data_verify = "txt_data.py"
        file_path_text_data_verify = os.path.join(folder_path_verify,
                                                                 folder_path_text_data_verify)
        return file_path_text_data_verify

    def input_limit(self):
        global folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk
        folder_path_input_limit = os.path.join(current_dir, "scripts","ai_renderer", "input_limit")
        folder_path_name_input_limit = ("limit.txt")
        file_path_input_limit = os.path.join(folder_path_input_limit,
                                                      folder_path_name_input_limit)
        return file_path_input_limit

    def settings_input_process_limit(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_name_input_process_limit = "input_process_limit.txt"
        file_path_settings_name_input_process_limit = os.path.join(folder_path_settings,
                                                             folder_path_settings_name_input_process_limit)
        return file_path_settings_name_input_process_limit

    def versionDATA_trainingdata_sentences(self):
        folder_path_training_data_versionDATA = os.path.join(current_dir, 'scripts', 'ai_renderer_2',"training_data_sentences")
        folder_path_versionDATA_name_sentences = "data.csv"
        file_path_path_versionDATA_name_sentences = os.path.join(folder_path_training_data_versionDATA,
                                                             folder_path_versionDATA_name_sentences)
        return file_path_path_versionDATA_name_sentences

    def Sentence_Intent_User(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Intent_User = "Sentence_Intent_User.txt"
        file_path_OutputData_Sentence_Intent_User = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Sentence_Intent_User)
        return file_path_OutputData_Sentence_Intent_User

    def Sentence_Type_User(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Type_User = "Sentence_Type_User.txt"
        file_path_OutputData_Sentence_Type_User = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Sentence_Type_User)
        return file_path_OutputData_Sentence_Type_User

    def Sentence_Intent_Global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Intent_Global = "Sentence_Intent_Global.txt"
        file_path_OutputData_Sentence_Intent_Global = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Sentence_Intent_Global)
        return file_path_OutputData_Sentence_Intent_Global

    def Sentence_Type_Global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Type_Global = "Sentence_Type_Global.txt"
        file_path_OutputData_Sentence_Type_Global = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Sentence_Type_Global)
        return file_path_OutputData_Sentence_Type_Global

    def Sentence_Accuracy_Average_Global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Accuracy_Average_Global = "Sentence_Accuracy_Average_Global.txt"
        file_path_OutputData_Sentence_Accuracy_Average_Global = os.path.join(folder_path_OuputData,
                                                             folder_path_OuputData_Sentence_Accuracy_Average_Global)
        return file_path_OutputData_Sentence_Accuracy_Average_Global

    def Sentence_Accuracy_Intent_Global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Intent_Accuracy_Global = "Sentence_Intent_Accuracy_Global.txt"
        file_path_OutputData_Sentence_Intent_Accuracy_Global = os.path.join(folder_path_OuputData,
                                                                             folder_path_OuputData_Sentence_Intent_Accuracy_Global)
        return file_path_OutputData_Sentence_Intent_Accuracy_Global

    def Sentence_Accuracy_Type_Global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path_OuputData_Sentence_Type_Accuracy_Global = "Sentence_Type_Accuracy_Global.txt"
        file_path_OutputData_Sentence_Type_Accuracy_Global = os.path.join(folder_path_OuputData,
                                                                            folder_path_OuputData_Sentence_Type_Accuracy_Global)
        return file_path_OutputData_Sentence_Type_Accuracy_Global

    def settings_ui_days_ago(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_ago = "ui_days_ago.txt"
        file_path_settings_name_ui_days_ago = os.path.join(folder_path_settings,
                                                                   folder_path_settings_ui_days_ago)
        return file_path_settings_name_ui_days_ago

    def settings_ui_days_apart(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_apart = "ui_days_apart.txt"
        file_path_settings_name_ui_days_apart = os.path.join(folder_path_settings,
                                                           folder_path_settings_ui_days_apart)
        return file_path_settings_name_ui_days_apart

    def settings_ui_weeks_ago(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_ago = "ui_weeks_ago.txt"
        file_path = os.path.join(folder_path_settings,folder_path_settings_ui_days_ago)
        return file_path

    def settings_ui_weeks_apart(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_apart = "ui_weeks_apart.txt"
        file_path = os.path.join(folder_path_settings,folder_path_settings_ui_days_apart)
        return file_path

    def settings_ui_months_ago(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_ago = "ui_months_ago.txt"
        file_path = os.path.join(folder_path_settings,folder_path_settings_ui_days_ago)
        return file_path

    def settings_ui_months_apart(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_apart = "ui_months_apart.txt"
        file_path = os.path.join(folder_path_settings,folder_path_settings_ui_days_apart)
        return file_path

    def settings_ui_years_ago(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_ago = "ui_years_ago.txt"
        file_path = os.path.join(folder_path_settings, folder_path_settings_ui_days_ago)
        return file_path

    def settings_ui_years_apart(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path_settings_ui_days_apart = "ui_years_apart.txt"
        file_path = os.path.join(folder_path_settings, folder_path_settings_ui_days_apart)
        return file_path

    def ui_day_changes(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_day_changes.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def ui_week_changes(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_week_changes.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def ui_month_changes(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_month_changes.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def ui_year_changes(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_year_changes.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def arch_runner(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'extra_models', 'Sulfur',"ArchitectureBuild")
        folder_path = "arch_runner.py"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def arch_runner_folder(self):
        folder_path = os.path.join(current_dir, 'extra_models', 'Sulfur',"ArchitectureBuild")
        return folder_path

    def settings_auto_trainer_extra_debug(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        file_name = "autotrainer_extra_debug.txt"
        file_path = os.path.join(folder_path_settings, file_name)
        return file_path

    def settings_pip_fallback_amount(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        file_name = "pip_fallback_amount.txt"
        file_path = os.path.join(folder_path_settings, file_name)
        return file_path

    def settings_auto_trainer_delay(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        file_name = "autotrainer_extra_speed.txt"
        file_path = os.path.join(folder_path_settings, file_name)
        return file_path

    def Output_UserInsight(self):
        folder_path_output = os.path.join(current_dir, 'data', 'training_data',)
        folder_path_output_name = ("Output_UserInsight.txt")
        file_path = os.path.join(folder_path_output, folder_path_output_name)
        return file_path

    def Output_Data(self):
        folder_path_output = os.path.join(current_dir, 'data', 'training_data',)
        folder_path_output_name = ("Output.txt")
        file_path = os.path.join(folder_path_output, folder_path_output_name)
        return file_path

    def settings_ui_write_to_seperate_output(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        file_name = "ui_write_to_seperate_output.txt"
        file_path = os.path.join(folder_path_settings, file_name)
        return file_path

    def ui_predicted_location(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_Predicted_Location.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def ui_predicted_location_accuracy(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_Predicted_Location_Accuracy.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def ui_predicted_location_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_Predicted_Location_Global.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def ui_predicted_location_accuracy_global(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Insight_Predicted_Location_Accuracy_Global.txt"
        file_path = os.path.join(folder_path_OuputData, folder_path)
        return file_path

    def training_data_location(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir,"scripts","ai_renderer_sentences", "sentence_location_build","training_data_sentences")
        folder_path = "data.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def cache_LocalScriptHost(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LocalScriptHost.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def cache_LocalScriptDebugBool(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LocalScriptDebugBool.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def cache_LocalpipCacheDebug(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LocalpipCacheDebug.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path


    def EXTERNALAPP_dashboard_renderer(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir,'apps', 'SulfurDashboardAssets', 'renderer')
        folder_path = "snippet.html"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def EXTERNALAPP_dashboard_css(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir,'apps', 'SulfurDashboardAssets', 'styling')
        folder_path = "style.css"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def EXTERNALAPP_dashboard_js(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir,'apps', 'SulfurDashboardAssets', 'styling')
        folder_path = "script.js"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def EXTERNALAPP_dashboard_sulfurLogo64(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir,'apps', 'SulfurDashboardAssets', 'styling')
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir,'apps', 'SulfurDashboardAssets', 'styling')
        folder_path = "SulAiB64.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def EXTERNALAPP_dashboard_sulfurLogo(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'images',"logo")
        folder_path = "SulfurAILogo.jpeg"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def file_path_input(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data')
        folder_path = "Input.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def file_path_attributes(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data')
        folder_path = "Attributes.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def folder_path_input(self):
        folder_path = os.path.join(current_dir, 'data', 'training_data')
        return folder_path

    def api_server_python_cache_input(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'api', 'Python', 'SulfurServerSystem', 'cache_input')
        folder_path = "input.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def settings_save_training_data(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path = "trainingdata_add.txt"
        file_path = os.path.join(folder_path_settings,
                                                            folder_path)
        return file_path

    def User_Model_Debug(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "User_Model_Debug.txt"
        file_path = os.path.join(folder_path_OuputData,
                                                                  folder_path)
        return file_path

    def cache_LoadedDataProfileID(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LoadedDataProfileID.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def cache_LocalEventsHost(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LocalEventsHost.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def Sentence_Oppurtunity_Accuracy(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "Sentence_Oppurtunity_Accuracy.txt"
        file_path = os.path.join(folder_path_OuputData,
                                                                             folder_path)
        return file_path

    def Sentence_Oppurtunity(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "Sentence_Oppurtunity.txt"
        file_path = os.path.join(folder_path_OuputData,
                                 folder_path)
        return file_path

    def Advanced_User_Model_Debug(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "Advanced_User_Model_Debug.txt"
        file_path = os.path.join(folder_path_OuputData,
                                 folder_path)
        return file_path

    def Output_score(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'training_data', 'Output_Data')
        folder_path = "Output_Score.txt"
        file_path = os.path.join(folder_path_OuputData,
                                 folder_path)
        return file_path

    def settings_auto_render_dp(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path = "auto_render_dp.txt"
        file_path = os.path.join(folder_path_settings,
                                                            folder_path)
        return file_path

    def cache_LocalDataProfileSulfurCount(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LocalDataProfileSulfurCount.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def settings_debug_dp(self):
        folder_path_settings = os.path.join(current_dir, 'settings')
        folder_path = "debug_dp.txt"
        file_path = os.path.join(folder_path_settings,
                                                            folder_path)
        return file_path

    def return_cache(self):
        folder_path_settings = os.path.join(current_dir, 'returns', 'dataprofiles', 'scripts')
        folder_path = "return_cache.txt"
        file_path = os.path.join(folder_path_settings,
                                                            folder_path)
        return file_path

    def profile_default(self):
        """
        Could be deprecated in future versions.

        Last updated: dev_build v1.3.99-patch6
        """
        folder_path_settings = os.path.join(current_dir, 'returns', 'dataprofiles', 'profiles', 'default', 'profile')
        folder_path = "profile.json"
        file_path = os.path.join(folder_path_settings,
                                                            folder_path)
        return file_path

    def active_model(self):
        folder_path_settings = os.path.join(current_dir, 'extra_models', 'Sulfur', 'Models', 'active')
        folder_path = "active_model.txt"
        file_path = os.path.join(folder_path_settings,
                                                            folder_path)
        return file_path

    def cache_LocalActiveModelHistory(self):
        folder_path_OuputData_average_device_accuracy = os.path.join(current_dir, 'data', 'cache')
        folder_path = "LocalActiveModelHistory.txt"
        file_path = os.path.join(folder_path_OuputData_average_device_accuracy, folder_path)
        return file_path

    def artifacts_dir(self):
        folder_path = os.path.join(current_dir, 'scripts', 'ai_renderer_2', 'tensorflowDependancies')
        return folder_path

    def check_device_s(self):
        folder_path = os.path.join(current_dir, 'scripts', 'ai_renderer_sentences')
        folder_path_i = "Check_device_s.py"
        file_path = os.path.join(folder_path, folder_path_i)
        return file_path






call = Call()
file_path_OutputData_name_Device  = call.device()
folder_path_OuputData, folder_path_OuputData_name_Device_accuracy, file_path_OutputData_name_Device_accuracy = call.device_accuracy()
folder_path_OuputData, folder_path_OuputData_name_Response_Time_MS, file_path_OutputData_name_Response_Time_MS = call.response_time()
folder_path_trainingData_grammar,folder_path_trainingData_grammar_name,file_path_trainingData_grammar = call.grammar()
folder_path_output,folder_path_output_name,file_path_output = call.output()
folder_path_archFullTimeRendererSulfax,folder_path_archFullTimeRendererSulfax_name,file_path_archFullTimeRendererSulfax = call.arch()
folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk = call.training_data_sk()