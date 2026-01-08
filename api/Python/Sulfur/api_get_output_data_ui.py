

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

def get_output_data_ui(strip_newline_marker=False):

    """
    Returns the output_userinsight.txt content as a list.
    Useful to save code and efficiency.

    Extra arguments:

     strip_newline_marker = False/True [DEFAULT: False]

     -Adds whether to include the '\n' tag to each item in the return statement.

    """
    import os
    from setup.exceptions import except_host
    old_cwd = os.getcwd()
    try:
        file_path_ui = call.Output_UserInsight()
        if not strip_newline_marker:
            with open(file_path_ui, "r", encoding="utf-8", errors="ignore") as file:
                lines = [line.strip() for line in file.readlines()]
        else:
            with open(file_path_ui, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()
        return lines
    except (NameError, TypeError, FileNotFoundError, IOError, ValueError, AttributeError) as e:
        except_host.handle_sulfur_exception(e, call)

    finally:
        os.chdir(old_cwd)







    try:
        file_path_output = call.Output_Data()
        if not strip_newline_marker:
            with open(file_path_output, "r", encoding="utf-8", errors="ignore") as file: lines = [line.strip() for line in file.readlines()]
        else:
            with open(file_path_output, "r", encoding="utf-8", errors="ignore") as file: lines = file.readlines()

        return lines
    except (NameError, TypeError, FileNotFoundError, IOError, ValueError, AttributeError) as e:
        except_host.handle_sulfur_exception(e, call)

    finally:
        os.chdir(old_cwd)