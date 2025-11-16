
################ Welcome to the SulfurAI Console!
### Functions here should be modified with proper intent.
### This python script was written in the Holly format. To find out how it works go into VersionDATA/HollyFormat/ReadMe.txt
### This python script is designed to host all SulfurAI Console API functions for python and run via the __main__ tag.

### LAYOUT:
# ---------------GOING DOWN!
##### -Call path locator
##### -Console class:
        #-Remove print statements to all API files

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()
call = _get_call_file_path()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from setup.exceptions import except_host

class console():
    def __init__(self):
        pass

    def set_api_debug_print(self,value,print_status=True):
        """
           Sets the SulfurAI debug. If set to "True", Sulfur API modules will print debug statements.

            -------------------------------

            Main arguments:
              - value = string of 'True' or 'False'

            -------------------------------

            Extra arguments:
               - print_status = True/False [default = True]
                 - Decides whether this script prints debug statements.

        """

        if value == "True" or value == "False": pass
        else: raise ValueError("SULFUR EXCEPTION (Console.set_api_debug_print): value must be a string of 'True' or 'False'!")
        file_path_localhost_debug_print = call.cache_LocalScriptDebugBool()

        try:

         with open(file_path_localhost_debug_print, "w", encoding="utf-8", errors="ignore") as file: file.write(value)
         if print_status: print(f"✅ Console successfully wrote to the file.")

        except (NameError, TypeError, FileNotFoundError, IOError, ValueError, AttributeError) as e:
            except_host.handle_sulfur_exception(e, call)

    def check_api_debug_print(self, print_status=True):
        """
           Checks the status of the API debug print cache. For more info use _____.console.set_api_debug_print()

            -------------------------------

            Main arguments:
              - print_status = True/False [default = True]
                 - Decides whether this script prints debug statements.

            -------------------------------


            Returns:
                - Boolean value of True/False
        """
        file_path_localhost_debug_print = call.cache_LocalScriptDebugBool()
        try:

            with open(file_path_localhost_debug_print, "r", encoding="utf-8", errors="ignore") as file:
                first_char = file.read(1)

                if not first_char:
                    with open(file_path_localhost_debug_print, "w", encoding="utf-8", errors="ignore") as file:  file.write("True")
                    if print_status: print(f"⚠️ Console added 'False' to cache due to cache being empty.")

            with open(file_path_localhost_debug_print, "r", encoding="utf-8", errors="ignore") as file:
                txt = file.readlines()
                if print_status: print(f"✅ Console successfully read value as a return.")
                return any(line.strip() == "True" for line in txt)

        except (NameError, TypeError, FileNotFoundError, IOError, ValueError, AttributeError) as e:
            except_host.handle_sulfur_exception(e, call)



