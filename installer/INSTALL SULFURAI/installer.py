################ Welcome to SulfurAI installer!
### Functions here should be modified with proper intent.
### This python script was written in the Holly format. To find out how it works go into VersionDATA/HollyFormat/ReadMe.txt
### This python script is designed to install the SulfurAI dashboard and its dependencies.

### LAYOUT:
# ---------------GOING DOWN!
######- Calling files
#######-Installing packages

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





import time,subprocess,sys

try:
    import importlib
except ModuleNotFoundError: print("Module importlib not found. Due to the conditions, it cannot automatically be installed. Please install it using pip.")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from extra_models.Sulfur.InstallManager.data import SulfurAI_Installer_Manager
SulfurAI_Installer_Manager.init(main=True,running_from_installer=False)

print("Script has finished.")
time.sleep(100)