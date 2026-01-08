from extra_models.Sulfur.TrainingScript.Build import call_file_path
from scripts.ai_renderer_sentences import error
error_print = error.error
import random,math
import re



call = call_file_path.Call()
folder_path_training_data,folder_path_training_data_name,file_path_training_data = call.training_data()
folder_path_training_data_sk, folder_path_training_data_name_sk, file_path_training_data_sk = call.training_data_sk()
file_path_OutputData_name_mean_device = call.mean_device()
file_path_OutputData_name_mean_device_average_device_accuracy = call.average_device_accuracy()

def get_main_device():

    desktop_count = 0
    mobile_count = 0
    with open(file_path_training_data, "r", encoding="utf-8", errors="ignore") as file:
        try:
            lines = file.readlines()
            for line in lines:
                if "[DEVICE : DESKTOP]" in line:  desktop_count += 1
                if "[DEVICE : MOBILE(OTHER)]" in line:  mobile_count += 1
        except ValueError:
            #print(f"Value_Error script Mean_device_s.PY, {desktop_count} {mobile_count}") business ver
            pass

    main_device = "DESKTOP" if desktop_count > mobile_count else "MOBILE(OTHER)"

    ##writing to the output file
    with open(file_path_OutputData_name_mean_device, "w", encoding="utf-8", errors="ignore") as file:
        file.write(main_device)

    return main_device

def get_main_accuracy():
    try:

        with open(file_path_training_data_sk, "r", encoding="utf-8", errors="ignore") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

            base_accuracy = 0.0

            for line in lines:
                if isinstance(line, list):
                    line_list = line
                else:
                    line_list = line.split(",")
                if len(line_list) < 2:
                    if ',' not in line:
                        pass
                    else:
                        print("Warning. Not enough items in index list (script: Mean_device_s.py) Resorting to dummy list. Risk: (1/10)")

                    if len(line_list) == 1:
                        try:
                            placeholder1 = int(float(line_list[0]))
                        except ValueError:

                            placeholder1 = random.randint(0, 100)
                    else:
                        placeholder1 = random.randint(0,
                                                      100)
                else:
                    try:
                        placeholder1 = int(float(line_list[-2]))
                    except ValueError:
                        print(f"Error parsing line: {line}")
                        continue

                    if placeholder1 < 99:
                        base_accuracy += placeholder1
                    else:
                        place_range = math.floor((placeholder1 / 2) + 1)
                        if place_range >= 99:
                            base_accuracy += random.randint(int(99 + place_range / 1.1), place_range)
                        else:
                            base_accuracy += place_range





    except Exception as e:
        if e == ValueError:
            print(f"Error converting '{line_list[-2]}' to float or other type.")
        else:
            print(f"Critical Training Data Error. A minimum of 3 parameters is needed. File path DATA/ai_renderer/training_data_sk/data.txt")
            error_print("er2","Critical Training Data Error. A minimum of 3 parameters is needed. File path DATA/ai_renderer/training_data_sk/data.txt","INSTALL NEW VER", "8")




    try:
      accuracy_average = base_accuracy / len(lines)
    except ZeroDivisionError:
         accuracy_average = 0.0
         print("ZeroDivisionError for average_accuracy. Check Training Data or Re-instate it.")
         error_print("er2", "TRAININGDATA  = MEAN_DEVICE_S", "INSTALL NEW VER","9")
    except ValueError:
         accuracy_average = 0.0
         print("ValueError for average_accuracy. Check Training Data or Re-instate it.")
         error_print("er2", "TRAININGDATA  = MEAN_DEVICE_S", "INSTALL NEW VER","10")


    if accuracy_average > 100: accuracy_average = 99.9
    with open(file_path_OutputData_name_mean_device_average_device_accuracy, "w", encoding="utf-8", errors="ignore") as file:
         file.write(str(accuracy_average))

    return accuracy_average







