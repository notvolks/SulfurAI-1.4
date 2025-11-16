

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

def setup_local(directory=None):
    """
    Sets up SulfurAI locally by adding the given directory to the PYTHONPATH.
    If no directory is provided, uses the current working directory.
    """
    import os, sys

    import SulfurConsole
    sulfur_console = SulfurConsole.console()
    print_api_debug = sulfur_console.check_api_debug_print(print_status=False)


    directory = os.path.abspath(directory or os.getcwd())

    # -- Cache writing --
    try:
        file_path_cache_LocalHost = call.cache_LocalScriptHost()
        os.makedirs(os.path.dirname(file_path_cache_LocalHost), exist_ok=True)
        with open(file_path_cache_LocalHost, "a", encoding="utf-8", errors="ignore") as file:
            file.write(f"LL{directory}\n")
        if print_api_debug: print(f"✅ Wrote cache: {file_path_cache_LocalHost}")
    except Exception as e:
        print(f"❌ Failed to write cache: {e}")
        return

    # -- Set up persistent PYTHONPATH (for new terminals) --
    if os.name == 'nt':
        profile_file = os.path.expanduser('~\\Documents\\WindowsPowerShell\\profile.ps1')
        line_to_add = f'$env:PYTHONPATH = "{directory};$env:PYTHONPATH"\n'
    else:
        profile_file = os.path.expanduser('~/.bashrc')
        line_to_add = f'export PYTHONPATH="$PYTHONPATH:{directory}"\n'

    try:
        os.makedirs(os.path.dirname(profile_file), exist_ok=True)
        if not os.path.exists(profile_file):
            with open(profile_file, 'w', encoding='utf-8') as f:
                f.write("# Created by SulfurAI setup_local\n")

        with open(profile_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not any(line.strip() == line_to_add.strip() for line in lines):
            with open(profile_file, 'a', encoding='utf-8') as f:
                f.write(line_to_add)
            if print_api_debug: print(f"✅ Added to {profile_file}")
        else:
            if print_api_debug: print("ℹ️ Already added to profile.")
    except Exception as e:
        print(f"❌ Failed to modify profile file: {e}")

    # -- Make it effective immediately in current Python process --
    if directory not in sys.path:
        sys.path.insert(0, directory)
        os.environ["PYTHONPATH"] = f"{directory};" + os.environ.get("PYTHONPATH", "")
        print(f"🔄 Added {directory} to sys.path for this session.")

    if print_api_debug: print("⚠️SulfurAI has now been setup. You can safely delete this line of code: '______.setup_local()'")