from extra_models.Sulfur.TrainingScript.Build import call_file_path
from scripts.ai_renderer_sentences import error
call = call_file_path.Call()
error_print = error.error

file_path_training_data_sk_backup = call.backup_training_data_sk()
file_path_training_data_backup = call.backup_training_data()

_,_,file_path_training_data = call.training_data()
_,_, file_path_training_data_sk = call.training_data_sk()

file_path_settings_name_backup = call.settings_backup()
with open(file_path_settings_name_backup, "r", encoding="utf-8", errors="ignore") as file:
    allow_backup_td = file.readlines()

#training data
def restore_data():
    try:

        with open(file_path_training_data, 'r', encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            if len(lines) == 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!No training or unsuitable (too little) training data found. Will be replaced with backup")

                with open(file_path_training_data_backup, 'r', encoding="utf-8", errors="ignore") as backup_file:
                    training_data_backup = backup_file.readlines()


                with open(file_path_training_data, 'a', encoding="utf-8",
                          errors="ignore") as file:
                    file.write(''.join(training_data_backup))

        #training data_sk
        with open(file_path_training_data_sk, 'r', encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            if len(lines) == 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!No training or unsuitable (too little) training data found. Will be replaced with backup")
                with open(file_path_training_data_sk_backup, 'r', encoding="utf-8", errors="ignore") as backup_file:
                    training_data_backup = backup_file.readlines()

                with open(file_path_training_data_sk, 'a', encoding="utf-8",
                          errors="ignore") as file:
                    file.write(''.join(training_data_backup))
    except Exception as e:
        print(f"{e} backup training data found for SK model.")
        print("!!!!!!!!!!!!!!!!!!!!!!!Recommended action: install new version")
        error_print("er1", "DEPENDANCY_FILE", f"BACKUP TRAINING DATA : {e}","7")


################used to hardforce a reset

def restore_data_hardforce():
    if "yes" in allow_backup_td:
        try:

            with open(file_path_training_data, 'r', encoding="utf-8", errors="ignore") as file:

                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!No training or unsuitable (too little) training data found. Will be replaced with backup")

                with open(file_path_training_data_backup, 'r', encoding="utf-8", errors="ignore") as backup_file:
                    training_data_backup = backup_file.readlines()

                with open(file_path_training_data, 'a', encoding="utf-8",
                          errors="ignore") as file:
                    file.write(''.join(training_data_backup))

            # training data_sk
            with open(file_path_training_data_sk, 'r', encoding="utf-8", errors="ignore") as file:

                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!No training or unsuitable (too little) training data found. Will be replaced with backup")
                with open(file_path_training_data_sk_backup, 'r', encoding="utf-8", errors="ignore") as backup_file:
                    training_data_backup = backup_file.readlines()

                with open(file_path_training_data_sk, 'a', encoding="utf-8",
                          errors="ignore") as file:
                    file.write(''.join(training_data_backup))
        except Exception as e:
            print(f"{e} backup training data found for SK model.")
            print("!!!!!!!!!!!!!!!!!!!!!!!Recommended action: install new version")
            error_print("er1", "DEPENDANCY_FILE", f"BACKUP TRAINING DATA : {e}","7")
    else:
        print("Your settings prevented training data being written by backups.")
