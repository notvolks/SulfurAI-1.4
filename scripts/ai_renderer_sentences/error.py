import os, datetime, time
def who_imported_me():
    import inspect
    return inspect.stack()[2].filename
def print_verti_list(items):  # Print items vertically
    for item in items:
        print(item)

def write_error(error, type):
    current_dir_e_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    folder_path_error_log = os.path.join(current_dir_e_log,'data', 'ErrorLogs')
    file_name_error_log = 'EasyLog.txt'
    file_path_error_log = os.path.join(folder_path_error_log, file_name_error_log)

    current_dir_db_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    folder_path_error_debug_log = os.path.join(current_dir_db_log,'data', 'ErrorLogs', 'logs')
    file_name_error_debug_log = 'error_debug.txt'
    file_path_error_debug_log = os.path.join(folder_path_error_debug_log, file_name_error_debug_log)

    current_dir_d_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    folder_path_error_list_d_log = os.path.join(current_dir_d_log,'data', 'ErrorLogs', 'logs')
    file_name_error_list_d_log = 'list_debug.txt'
    file_path_error_list_d_log = os.path.join(folder_path_error_list_d_log, file_name_error_list_d_log)

    if type == "debug_error":
        with open(file_path_error_list_d_log, "r") as file:
            lines_rd = file.readlines()
            if lines_rd:
                first_line_rd = lines_rd[0].strip()
                error_list_debug = first_line_rd.split(",")
            else:
                with open(file_path_error_list_d_log, "w") as file:
                    file.write(" ")

        with open(file_path_error_debug_log, "a") as file:
            error_message = f"{','.join(error_list_debug)}, {error}"
            file.write(f"{error_message}")

    if len(type) > 0:
        with open(file_path_error_log, "a") as file:
            error_msg_split = error_message.split(",")
            time_now = datetime.datetime.now()
            time_printed = time_now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"#########ERROR {time_printed}: {error_msg_split}\n")

def error(layout_type, error_type, error_message, error_code):  # Print errors
    if layout_type == "er1":
        text = [
            "###ERROR####",
            f"##No ({error_type}) ~{error_message}~##",
            "##   Found     ##",
            "Install a new version."
        ]
        print_verti_list(text)
        print(f"Error code: {error_code}")
        print(f"Host scripts ID: {who_imported_me()}")
        write_error(f"Error Code ({error_code}) // No {error_type} ~{error_message} Found. **Solution - install new version** @host_file:{who_imported_me()}", "debug_error")
    elif layout_type == "er2":
        text = [
            f"{error_type} Failed...",
            f"#{error_message}#"
        ]
        print_verti_list(text)
        print(f"Error code: {error_code}")
        print(f"Host scripts ID: {who_imported_me()}")
        write_error(f"Error Code ({error_code}) // No {error_type} Failed. ~{error_message}. **Solution - install new version** @host_file:{who_imported_me()}", "debug_error")




def instant_shutdown(reason):
    try:
        exit()
    except (ImportError, SystemExit) as y:
        print(f"InstantShutDown-Initiated: {reason} Testing exit: Value:Exception// exit(sys):{y}")
        time.sleep(5)
        quit()


def brick_out(time2):
    time.sleep(time2)
    instant_shutdown("Timed out.")