import SulfurAI
SulfurAI.setup_local()

import SulfurConsole
sulfur_console = SulfurConsole.console()
print(sulfur_console.check_api_debug_print(print_status=False))
print(sulfur_console.set_api_debug_print("True",print_status=True))


INPUT_TEXT = SulfurAI.run_locally("test_endpoint",True)
print("Completed task.")
print(INPUT_TEXT["INPUT_TEXT"][0])
print(INPUT_TEXT["PREDICTED_USER_DEVICE"])
print(INPUT_TEXT["PREDICTED_USER_DEVICE_ACCURACY"])
print(INPUT_TEXT["USER_SENTENCE_INTENT"])
print(INPUT_TEXT["PREDICTED_USER_LOCATION_COUNTRY"])

print(SulfurAI.get_output_data())
print(SulfurAI.get_output_data_ui())

