import math
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
(folder_path_OuputData, folder_path_OuputData_name_Response_Time_MS, file_path_OutputData_name_Response_Time_MS) = call.response_time()

# python
from datetime import datetime


def _finish_script(start_time, main: bool):
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime object")

    finish_time = datetime.now()
    finish_time_printed = finish_time.strftime("%Y-%m-%d %H:%M:%S") + f".{finish_time.microsecond // 1000:03d}"

    if main:
        print(f"|                All tasks completed at {finish_time_printed}. Check output file.                |")

    total_time = finish_time - start_time
    total_seconds = int(total_time.total_seconds())
    total_time_ms = int((total_time.total_seconds() - total_seconds) * 1000)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if main:
        print(
            f"|                Time taken: {hours} hours, {minutes} minutes, {seconds} seconds, {total_time_ms} milliseconds.                |")
        menu_end = [
            "--------------------------------------------------------------------------------------------------",
            "|                                  SULFUR AI HAS FINISHED RENDERING.                               |",
            "--------------------------------------------------------------------------------------------------",
        ]
        from scripts.ai_renderer_sentences import error
        print_verti_list = error.print_verti_list
        print_verti_list(menu_end)

    with open(file_path_OutputData_name_Response_Time_MS, "w") as file:
        file.write(f"{hours} hours, {minutes} minutes, {seconds} seconds, {total_time_ms} milliseconds.")

    return hours, minutes, seconds, total_time_ms