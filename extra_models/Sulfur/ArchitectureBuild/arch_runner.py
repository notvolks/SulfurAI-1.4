from scripts.ai_renderer_sentences import error
error_print = error.error
import os

def get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()
call = get_call_file_path()

class Arch:
    instant_shutdown = error.instant_shutdown
    def __init__(self):
        (self.folder_path_archFullTimeRendererSulfax, self.folder_path_archFullTimeRendererSulfax_name,self.file_path_archFullTimeRendererSulfax) = call.arch()

    def arch_items_append(self, item):
        global arch_items
        arch_items.append(item)

    @staticmethod
    def arch_items_extend(item):
        global arch_items
        arch_items.extend(item)

    def check_run(self):
        global lines_list, arch_items
        if "run = True" in lines_list and "run = True" in arch_items:
            pass
        if "run = False" in lines_list and "run = False" in arch_items:
            self.instant_shutdown("ARCH ERROR.NOT RAN. INSTALL NEW VER.")
        if "start_arch = True" in lines_list and "start_arch = True" in arch_items:
            pass
        if "start_arch = False" in lines_list and "start_arch = False" in arch_items:
            self.instant_shutdown("ARCH ERROR. NOT STARTED. INSTALL NEW VER.")

    def check_lines(self):
        call = get_call_file_path()
        global lines_list, arch_items
        arch_items = ["run = True", "run = False", "start_arch = True", "start_arch = False"]
        folder_path_colon_symbol, folder_path_colon_symbol_name, file_path_colon_symbol = call.arch_colon()
        with open(file_path_colon_symbol, "r") as file:
            lines = file.readlines()
            colon_data = ', '.join(lines)
            if "{" in colon_data:
                self.arch_items_append("{")
            if "}" in colon_data:
                self.arch_items_append("}")

        with open(self.file_path_archFullTimeRendererSulfax, "r") as file:
            lines_list = [line.strip() for line in file.readlines()]
            self.check_run()

    def verify_input(self):
        file_path_text_data_verify = call.text_data_verify()
        if os.path.exists(file_path_text_data_verify):
            pass
        else:
            error_print("er2", "Verification_input", "INSTALL NEW VER", 12)
            self.instant_shutdown("VERIFY ERROR. NOT STARTED. INSTALL NEW VER. [input-data_vrfy]")

