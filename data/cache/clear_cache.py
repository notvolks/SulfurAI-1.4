
################ Welcome to clear_cache!
### Functions here should be modified with proper intent.
### This python script was written in the Holly format. To find out how it works go into VersionDATA/HollyFormat/ReadMe.txt
### This python script is designed to clear all cache.

### LAYOUT:
# ---------------GOING DOWN!
##### -Call path locator
##### -Clear cache files & logs


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import importlib.util
import os
from extra_models.Sulfur.GeneralScripts.LocalUserEvents import events_hoster
events_hoster.write_event("event_ClearedCache")

current_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,

))


module_path = os.path.join(current_dir, 'extra_models', 'Sulfur', 'TrainingScript', 'Build', 'call_file_path.py')
module_name = "call_file_path"

spec = importlib.util.spec_from_file_location(module_name, module_path)
call_file_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(call_file_path)


call = call_file_path.Call()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cache_logs = {

    "LocalScriptDebugBool" : call.cache_LocalScriptDebugBool(),
    "LocalScriptHost" : call.cache_LocalScriptHost(),
    "LocalpipCacheDebug" : call.cache_LocalpipCacheDebug(),
    "LoadedDataProfileID" : call.cache_LoadedDataProfileID(),
    "LocalEventsHost": call.cache_LocalEventsHost(),
    "LocalDataProfileSulfurCount": call.cache_LocalDataProfileSulfurCount(),
    "LocalActiveModelHistory": call.cache_LocalActiveModelHistory(),
    "AppSecretsFound_appid_developerconsole": call.cache_AppSecretsFound_appid_developerconsole(),

}
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    confirmed = input("Confirm to clear ALL cache? (Y/n):").lower()
    if confirmed == "y": break
    elif confirmed == "n": exit()
    else: print("You must select either 'Y' or 'n'!")

#Assume cache was confirmed to clear

for name, file_path in cache_logs.items():
    try:
        with open(file_path, "w", encoding="utf-8", errors="ignore") as file: pass
    except Exception as e:
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"Failed to write to {name} at {file_path}: {e}")

print("All cache was successfully cleared.")
