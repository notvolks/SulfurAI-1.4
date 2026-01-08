import time
import sys


def exit_manager(running_from_installer):
    print("SulfurAI was Installed.")
    init_path()
    if running_from_installer: print("Restart this program (SulfurAI) to apply changes.")
    time.sleep(10)
    exit()


def run_manager(running_from_installer):
    if not running_from_installer:
        Manager_UI = ["|--------------------------------------------------------|",
                      "|                    Sulfur AI Installer                 |",
                      "|--------------------------------------------------------|",
                      "|  This Script wants to install 'SULFURAI' |",
                      "|  Enter (Y/n) to confirm:                               |",
                      "|-------------------------------------------------------- | ",

                      ]
    else:
        Manager_UI = ["|--------------------------------------------------------|",
                      "|                    Sulfur AI Installer                 |",
                      "|--------------------------------------------------------|",
                      "|  This Script wants to Automatically install 'SULFURAI' |",
                      "|  Enter (Y/n) to confirm:                               |",
                      "|-------------------------------------------------------- | ",

                      ]
    for line in Manager_UI: print(line)
    while True:
        input_ = input(":").lower().strip()
        if input_ == "y":
            from setup.depenintsall.python311 import pipdependancyinstaller
            checks = pipdependancyinstaller.init(print_debug_op=True,RUN_INSTALLER_OP=True)
            if checks == "ALL_INSTALLED_AFTER_CHECK": exit_manager(running_from_installer)
            else:
                init_path()
                print("SulfurAI failed to install. Please re-try.")
                time.sleep(10)
                exit()
        if input_ == "n":
            if running_from_installer: print("You can install SulfurAI manually by going to INSTALLER/INSTALL SULFURAI")
            time.sleep(10)
            exit()
        else: print("Invalid input. Enter (Y/n) to confirm.")



def init(main=False,running_from_installer=False):
    if sys.version_info.major == 3 and sys.version_info.minor == 11: pass
    else:
        print("SulfurAI requires Python 3.11 to run. Please install Python 3.11 and re-run this script.")
        time.sleep(10)
        exit()
    from setup.depenintsall.python311 import pipdependancyinstaller
    checks = pipdependancyinstaller.init(print_debug_op=False,RUN_INSTALLER_OP=True)
    if checks == "NONE_INSTALLED_AFTER_CHECK": run_manager(running_from_installer)
    return checks


def _add_global_path():
    # Figure out where this script lives
    from pathlib import Path
    import sysconfig
    import site
    script_path = Path(__file__).resolve()

    # Project root is 4 levels up from this script
    project_root = script_path.parents[4]

    # Path to SulfurAI.py
    sulfurai_path = project_root / "SulfurAI.py"

    if not sulfurai_path.exists():
        print(f"[DEBUG:] Could not find SulfurAI.py at {sulfurai_path}")
        return

    print(f"[DEBUG]: Found SulfurAI.py at: {sulfurai_path}")

    # The directory that contains SulfurAI.py is what we want in sys.path
    path = sulfurai_path.parent

    targets = []

    # System Python site-packages
    system_site = sysconfig.get_paths()["purelib"]
    targets.append(Path(system_site) / "sulfurai.pth")

    # Virtualenv site-packages (only if different from system)
    venv_sites = site.getsitepackages()
    for venv_site in venv_sites:
        if venv_site != system_site:
            targets.append(Path(venv_site) / "sulfurai.pth")

    # Write .pth files
    for pth_file in targets:
        try:
            pth_file.parent.mkdir(parents=True, exist_ok=True)
            with open(pth_file, "w") as f:
                f.write(str(path) + "\n")
            print(f"[DEBUG]: Added {path} to {pth_file}")
        except PermissionError:
            print(f"[DEBUG]: Could not write to {pth_file} (need admin/sudo?)")

    print("\n[DEBUG]: You can now `import SulfurAI` inside this venv AND outside with system Python!")

def init_path():
    while True:
        print("Would you like to add SulfurAI to global PYTHONPATH?")
        print("This will allow you to 'import SulfurAI' in any Python script.")
        input_ca = input("(Y/n):").lower().strip()
        if input_ca != "y" and input_ca != "n": print("Invalid input. Enter (Y/n) to confirm.")
        else: break
    try:
     if input_ca == "y": _add_global_path()
    except Exception as e:
        print(f"Failed to add SulfurAI to global PYTHONPATH: {e}")
        print(f"Carrying on with the installation...")



