#!/usr/bin/env python3
# APP_refactored_safe.py — single CacheManager with name-based special-path detection


from __future__ import annotations
import os
import shutil
import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Any, Tuple
import SulfurAI
import deepsecrets
import json
import subprocess,sys
def _get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()
call = _get_call_file_path()

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Old index-based constants are replaced by explicit method-name detection:
DEFAULT_SPECIAL_METHOD_NAMES = {
    "api_server_python_cache",
    "profile_default_log_cache",
}

SPECIAL_INDICES = (7, 10)  # kept for reference only; not used by new logic

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("SulfurConsole")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def truncate_file(path: Path) -> None:
    try:
        path.write_text("")
    except Exception as exc:
        logger.exception("Failed to truncate %s: %s", path, exc)


def remove_path(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except Exception:
        logger.exception("Failed removing: %s", path)


def banner():
    print(YELLOW + "=" * 40)
    print("        SulfurAI Developer Console")
    print("=" * 40 + RESET)


def menu_main():
    print("\nChoose an option:")
    print(f"{YELLOW} 1{RESET} → Clear Cache")
    print(f"{YELLOW} 2{RESET} → Check for Secrets")
    print(f"{YELLOW} 3{RESET} → Reset Settings")
    print(f"{YELLOW} 4{RESET} → Tests")
    print(f"{YELLOW} 5{RESET} → Exit")


def menu_cache():
    print("\nChoose an option:")
    print(f"{YELLOW} 1{RESET} → All Cache")
    print(f"{YELLOW} 2{RESET} → Client Cache")
    print(f"{YELLOW} 3{RESET} → API Cache")
    print(f"{YELLOW} 4{RESET} → Output Cache")
    print(f"{YELLOW} 5{RESET} → Back to Main Menu")


def show_paths(title: str, paths: List[str]):
    print(f"\n{title}:")
    for p in paths:
        print(YELLOW + " - " + str(p) + RESET)


def ask(prompt="> "):
    try:
        return input("\n" + prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return "5"

import subprocess
import sys
import time
import threading
from pathlib import Path

def scan_for_secrets():
    root = normalize_path(call.sulfur_root())
    if root is None:
        print("Could not determine root path from call.sulfur_root()")
        return

    raw_out = call.cache_AppSecretsFound_appid_developerconsole()
    out_file = Path(raw_out)
    if out_file.suffix.lower() != ".json":
        out_file = out_file.with_suffix(".json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Correct exclusions, including .venv
    base_exclusions = [".venv", "__pycache__", ".git", "node_modules", "dist", "build"]

    exclusions = []
    for name in base_exclusions:
        candidate = Path(root) / name
        if candidate.exists():
            exclusions.append(str(candidate))

    excl_args = []
    if exclusions:
        ignore_file = call.EXTERNALAPP_deepsecretsignore_appid_developerconsole()
        excl_args = ["--excluded-paths", str(ignore_file)]

    spinner_on = True
    def spinner():
        symbols = "|/-\\"
        i = 0
        while spinner_on:
            print(f"\rScanning for secrets... {symbols[i%len(symbols)]}", end="", flush=True)
            i += 1
            time.sleep(0.12)

    t = threading.Thread(target=spinner, daemon=True)
    t.start()

    start = time.time()

    cmd = [
        sys.executable, "-m", "deepsecrets",
        "--target-dir", str(root),
        "--outfile", str(out_file),
        "--outformat", "json",
    ] + excl_args

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env
    )

    spinner_on = False
    t.join(timeout=0.1)

    elapsed = time.time() - start
    print(f"\n\nScan complete in {elapsed:.2f} seconds.")
    print("Report saved to:", out_file)

    if result.stdout.strip():
        print("\nSTDOUT:")
        print(result.stdout)



def normalize_path(raw):
    """
    Normalizes many possible return shapes.
    Priority rules:
    1. If it's a tuple/list: return the element containing 'file_path'.
    2. Then: return any element that looks like a full path (has slashes + extension).
    3. Then: any string/path-like element.
    4. Fallback to raw[0] if nothing else works.
    """

    # None → no path
    if raw is None:
        return None

    # Already path-like (string, Path, etc.)
    if isinstance(raw, (str, bytes, os.PathLike)):
        return Path(raw)

    # --- tuple or list ---
    if isinstance(raw, (tuple, list)):
        # 1) Search for variable names containing 'file_path'
        for elem in raw:
            if isinstance(elem, str) and "file_path" in elem.lower():
                return Path(elem)

        # 2) Search for elements that LOOK like a file path (heuristic)
        for elem in raw:
            if isinstance(elem, str):
                if ("/" in elem or "\\" in elem) and "." in elem:
                    return Path(elem)

        # 3) Any string/path-like
        for elem in raw:
            if isinstance(elem, (str, bytes, os.PathLike)):
                try:
                    return Path(elem)
                except:
                    pass

        # 4) Fallback: use first item
        try:
            return Path(str(raw[0]))
        except:
            return None

    # --- objects with `.path` attribute ---
    if hasattr(raw, "path"):
        try:
            return Path(raw.path)
        except:
            return None

    return None

class TestManager:


    def __init__(self, prompt_func=None):
        self.prompt = prompt_func or (lambda prompt: input("\n" + prompt).strip())

    def run(self):
        while True:
            clear_screen()
            banner()
            print("\nChoose a test option:")
            print(f"{YELLOW} 1{RESET} → Render Test")
            print(f"{YELLOW} 2{RESET} → API Test")
            print(f"{YELLOW} 3{RESET} → Back to Main Menu")

            choice = self.prompt("> ")

            if choice == "1":
                clear_screen()
                banner()
                self.render_test()
                self.prompt("Press Enter...")

            elif choice == "2":
                self._api_menu()

            elif choice == "3":
                return

    def _api_menu(self):
        while True:
            clear_screen()
            banner()
            print("\nChoose an API test:")
            print(f"{YELLOW} 1{RESET} → Full API Test")
            print(f"{YELLOW} 2{RESET} → Back")

            choice = self.prompt("> ")

            if choice == "1":
                clear_screen()
                banner()
                self.api_full_test()
                self.prompt("Press Enter...")

            elif choice == "2":
                return

    # ----- placeholders -----
    def render_test(self):
        print(GREEN + "Testing API call (SulfurAI.render_locally()):" + RESET)

        try:
            SulfurAI.run_locally("A test prompt for rendering.", add_to_training_data=False)
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)

    def api_full_test(self):
        print(YELLOW + "#Note: Does not test renderers requiring API keys." + RESET)

        print(GREEN + "Testing API call (SulfurAI.get_output_data()):" + RESET)

        # -------------------------------------------------------


        try:
            SulfurAI.get_output_data()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)


        #-------------------------------------------------------

        print(GREEN + "Testing API call (SulfurAI.get_output_data_ui()):" + RESET)

        try:
            SulfurAI.get_output_data_ui()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)

        # -------------------------------------------------------

        print(GREEN + "Testing API call (SulfurAI.server.get()):" + RESET)

        try:
            srv = SulfurAI.server.get()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)

        # -------------------------------------------------------

        print(GREEN + "Testing API call (SulfurAI.server.clear_local_endpoint_cache()):" + RESET)

        try:
            srv = SulfurAI.server.get()
            srv.clear_local_endpoint_cache()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)

        # -------------------------------------------------------

        print(GREEN + "Testing API call (SulfurAI.retrieve_current_models()):" + RESET)

        try:
            SulfurAI.retrieve_current_models()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)

        # -------------------------------------------------------

        print(GREEN + "Testing API call (SulfurAI.retrieve_active_model()):" + RESET)

        try:
            SulfurAI.retrieve_active_model()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)

        # -------------------------------------------------------

        print(GREEN + "Testing API call (SulfurAI.return_dataprofiles()):" + RESET)

        try:
            SulfurAI.return_dataprofiles()
        except Exception as e:
            print(RED + f"Render test failed: {e}" + RESET)
        else:
            print(GREEN + "Render test completed succesfully." + RESET)



class SettingsManager:
    """
    Manage resetting application settings to defaults.
    Expects call.LOCALAPP_paths_appid_settings() to return a mapping:
      key -> (path_like, default_value)
    The first element is treated as a file path, and the second is written into that file.
    """

    def __init__(self, call_provider: Optional[Callable[[], Any]] = None):
        # call_provider is optional; defaults to using the global 'call'
        self.call_provider = call_provider or (lambda: call)

    def _raw_paths(self):
        cp = self.call_provider()
        try:
            p = cp.LOCALAPP_paths_appid_settings()
        except Exception:
            p = {}

        if not isinstance(p, dict):
            return {}

        return p

    def preview(self) -> List[str]:
        """
        Returns a list of preview strings:
        "setting_name -> file_path | current: <value> -> default: <value>"
        """

        out = []

        for name, raw in self._raw_paths().items():
            path_candidate = None
            default = "<no-default>"

            if isinstance(raw, (list, tuple)) and len(raw) >= 1:
                path_candidate = raw[0]

            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                default = raw[1]

            p = normalize_path(path_candidate)

            current = "<missing>"
            try:
                if p and p.exists() and p.is_file():
                    current = p.read_text(encoding="utf-8").strip()
                elif p and p.exists() and p.is_dir():
                    current = "<directory>"
            except Exception as e:
                current = f"<error: {e}>"

            out.append(
                f"{name} -> {p or '<empty>'} | current: {repr(current)} -> default: {repr(str(default))}"
            )
        return out

    def reset(self, make_backups: bool = False) -> None:
        """
        Writes the default value into each settings file.
        Optionally makes backups by appending .bak, .bak1, .bak2, etc.
        """

        for name, raw in self._raw_paths().items():
            path_candidate = None
            default = None

            if isinstance(raw, (list, tuple)) and len(raw) >= 1:
                path_candidate = raw[0]

            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                default = raw[1]

            p = normalize_path(path_candidate)

            if p is None:
                print(f"[skip] {name}: no path resolved")
                continue

            # Skip directories
            try:
                if p.exists() and p.is_dir():
                    print(f"[skip] {name}: path is a directory ({p}), skipping write")
                    continue
            except Exception as e:
                print(f"[skip] {name}: error inspecting path {p}: {e}")
                continue

            try:
                # Ensure parent directories exist
                p.parent.mkdir(parents=True, exist_ok=True)

                # Backups
                if make_backups and p.exists() and p.is_file():
                    bak = p.with_suffix(p.suffix + ".bak")
                    idx = 0
                    candidate = bak

                    # if .bak exists, increment
                    while candidate.exists():
                        idx += 1
                        candidate = p.with_suffix(p.suffix + f".bak{idx}")

                    bak = candidate
                    try:
                        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
                        print(f" - backup created: {bak}")
                    except Exception:
                        pass

                # Write default (convert to string)
                default_text = "" if default is None else str(default)

                p.write_text(default_text, encoding="utf-8")
                print(f"Wrote default for '{name}' -> {p} = {default_text!r}")

            except Exception as e:
                print(f"Failed to write '{name}' -> {p}: {e}")


class CacheManager:
    """
    Unified cache manager that replaces the previous multiple classes.

    scope: one of 'all', 'client', 'api', 'output'
    """

    def __init__(
        self,
        scope: str = "all",
        call_provider: Optional[Callable[[], Any]] = None,
        special_method_names: Optional[Sequence[str]] = None,
    ):
        self.scope = scope.lower()
        self.call_provider = call_provider or self._default_call_provider
        # Use name-based detection for "wipe entire directory" behaviour
        self.special_method_names = set(special_method_names or DEFAULT_SPECIAL_METHOD_NAMES)

    def _default_call_provider(self):
        # deferred import to reflect original behaviour
        from extra_models.Sulfur.TrainingScript.Build import call_file_path

        return call_file_path.Call()

    # Each _paths_* returns a list of tuples: (method_name, raw_path_value)
    def _paths_all(self) -> List[Tuple[str, Optional[str]]]:
        call = self.call_provider()
        return [
            ("cache_LocalScriptDebugBool", call.cache_LocalScriptDebugBool()),
            ("cache_LocalScriptHost", call.cache_LocalScriptHost()),
            ("cache_LocalpipCacheDebug", call.cache_LocalpipCacheDebug()),
            ("cache_LoadedDataProfileID", call.cache_LoadedDataProfileID()),
            ("cache_LocalEventsHost", call.cache_LocalEventsHost()),
            ("cache_LocalDataProfileSulfurCount", call.cache_LocalDataProfileSulfurCount()),
            ("cache_LocalActiveModelHistory", call.cache_LocalActiveModelHistory()),
            ("api_server_python_cache", call.api_server_python_cache()),  # special
            ("return_cache", call.return_cache()),
            ("profile_default_cache", call.profile_default_cache()),
            ("profile_default_log_cache", call.profile_default_log_cache()),  # special
            ("file_path_input", call.file_path_input()),
            ("output", call.output()),

            ("Advanced_User_Model_Debug", call.Advanced_User_Model_Debug()),
            ("average_device_accuracy", call.average_device_accuracy()),
            ("Describe_adjective_most_important_global", call.Describe_adjective_most_important_global()),
            ("Describe_adjective_most_important_user", call.Describe_adjective_most_important_user()),
            ("device", call.device()),
            ("device_accuracy", call.device_accuracy()),
            ("mean_device", call.mean_device()),
            ("Output_score", call.Output_score()),
            ("preferences_global", call.preferences_global()),
            ("mood_accuracy_global", call.mood_accuracy_global()),
            ("mood_accuracy_user", call.mood_accuracy_user()),
            ("mood_global", call.mood_global()),
            ("mood_user", call.mood_user()),
            ("preferences_user", call.preferences_user()),
            ("response_time", call.response_time()),
            ("Sentence_Accuracy_Average_Global", call.Sentence_Accuracy_Average_Global()),
            ("Sentence_Accuracy_Intent_Global", call.Sentence_Accuracy_Intent_Global()),
            ("Sentence_Intent_Global", call.Sentence_Intent_Global()),
            ("Sentence_Intent_User", call.Sentence_Intent_User()),
            ("Sentence_Oppurtunity", call.Sentence_Oppurtunity()),
            ("Sentence_Oppurtunity_Accuracy", call.Sentence_Oppurtunity_Accuracy()),
            ("Sentence_Accuracy_Type_Global", call.Sentence_Accuracy_Type_Global()),
            ("Sentence_Type_Global", call.Sentence_Type_Global()),
            ("Sentence_Type_User", call.Sentence_Type_User()),
            ("ui_day_changes", call.ui_day_changes()),
            ("ui_month_changes", call.ui_month_changes()),
            ("ui_predicted_location", call.ui_predicted_location()),
            ("ui_predicted_location_accuracy_global", call.ui_predicted_location_accuracy_global()),
            ("ui_predicted_location_accuracy", call.ui_predicted_location_accuracy()),
            ("ui_predicted_location_global", call.ui_predicted_location_global()),
            ("ui_week_changes", call.ui_week_changes()),
            ("ui_year_changes", call.ui_year_changes()),
            ("User_Model_Debug", call.User_Model_Debug()),
            ("Wanted_noun_most_important_global", call.Wanted_noun_most_important_global()),
            ("Wanted_noun_most_important_user", call.Wanted_noun_most_important_user()),
            ("Wanted_verb_most_important_global", call.Wanted_verb_most_important_global()),
            ("Wanted_verb_most_important_user", call.Wanted_verb_most_important_user()),
            ("AppSecretsFound_appid_developerconsole", call.cache_AppSecretsFound_appid_developerconsole()),
        ]

    def _paths_api(self) -> List[Tuple[str, Optional[str]]]:
        call = self.call_provider()
        return [
            ("api_server_python_cache", call.api_server_python_cache()),  # special
        ]

    def _paths_output(self) -> List[Tuple[str, Optional[str]]]:
        call = self.call_provider()
        return [
            ("return_cache", call.return_cache()),
            ("profile_default_cache", call.profile_default_cache()),
            ("file_path_input", call.file_path_input()),
            ("output", call.output()),

            ("Advanced_User_Model_Debug", call.Advanced_User_Model_Debug()),
            ("average_device_accuracy", call.average_device_accuracy()),
            ("Describe_adjective_most_important_global", call.Describe_adjective_most_important_global()),
            ("Describe_adjective_most_important_user", call.Describe_adjective_most_important_user()),
            ("device", call.device()),
            ("device_accuracy", call.device_accuracy()),
            ("mean_device", call.mean_device()),
            ("Output_score", call.Output_score()),
            ("preferences_global", call.preferences_global()),
            ("mood_accuracy_global", call.mood_accuracy_global()),
            ("mood_accuracy_user", call.mood_accuracy_user()),
            ("mood_global", call.mood_global()),
            ("mood_user", call.mood_user()),
            ("preferences_user", call.preferences_user()),
            ("response_time", call.response_time()),
            ("Sentence_Accuracy_Average_Global", call.Sentence_Accuracy_Average_Global()),
            ("Sentence_Accuracy_Intent_Global", call.Sentence_Accuracy_Intent_Global()),
            ("Sentence_Intent_Global", call.Sentence_Intent_Global()),
            ("Sentence_Intent_User", call.Sentence_Intent_User()),
            ("Sentence_Oppurtunity", call.Sentence_Oppurtunity()),
            ("Sentence_Oppurtunity_Accuracy", call.Sentence_Oppurtunity_Accuracy()),
            ("Sentence_Accuracy_Type_Global", call.Sentence_Accuracy_Type_Global()),
            ("Sentence_Type_Global", call.Sentence_Type_Global()),
            ("Sentence_Type_User", call.Sentence_Type_User()),
            ("ui_day_changes", call.ui_day_changes()),
            ("ui_month_changes", call.ui_month_changes()),
            ("ui_predicted_location", call.ui_predicted_location()),
            ("ui_predicted_location_accuracy_global", call.ui_predicted_location_accuracy_global()),
            ("ui_predicted_location_accuracy", call.ui_predicted_location_accuracy()),
            ("ui_predicted_location_global", call.ui_predicted_location_global()),
            ("ui_week_changes", call.ui_week_changes()),
            ("ui_year_changes", call.ui_year_changes()),
            ("User_Model_Debug", call.User_Model_Debug()),
            ("Wanted_noun_most_important_global", call.Wanted_noun_most_important_global()),
            ("Wanted_noun_most_important_user", call.Wanted_noun_most_important_user()),
            ("Wanted_verb_most_important_global", call.Wanted_verb_most_important_global()),
            ("Wanted_verb_most_important_user", call.Wanted_verb_most_important_user()),

            ("Output_UserInsight", call.Output_UserInsight()),
            ("profile_default_log_cache", call.profile_default_log_cache()),  # kept (was special in original)
        ]

    def _paths_client(self) -> List[Tuple[str, Optional[str]]]:
        call = self.call_provider()
        return [
            ("cache_LocalScriptDebugBool", call.cache_LocalScriptDebugBool()),
            ("cache_LocalScriptHost", call.cache_LocalScriptHost()),
            ("cache_LocalpipCacheDebug", call.cache_LocalpipCacheDebug()),
            ("cache_LoadedDataProfileID", call.cache_LoadedDataProfileID()),
            ("cache_LocalEventsHost", call.cache_LocalEventsHost()),
            ("cache_LocalDataProfileSulfurCount", call.cache_LocalDataProfileSulfurCount()),
            ("cache_LocalActiveModelHistory", call.cache_LocalActiveModelHistory()),
            ("AppSecretsFound_appid_developerconsole", call.cache_AppSecretsFound_appid_developerconsole()),
        ]

    def _paths(self) -> List[Tuple[str, Optional[str]]]:
        if self.scope == "all":
            return self._paths_all()
        if self.scope == "api":
            return self._paths_api()
        if self.scope == "output":
            return self._paths_output()
        if self.scope == "client":
            return self._paths_client()
        # fallback to all
        return self._paths_all()

    def preview(self) -> List[str]:
        """Return human-friendly preview list: 'method_name -> path'"""
        out = []
        for name, raw in self._paths():
            try:
                p = normalize_path(raw)
                out.append(f"{name} -> {p or '<empty>'}")
            except Exception:
                out.append(f"{name} -> <unreadable>")
        return out

    # ---------------------------------------------------------
    #      CLEARING LOGIC (unified, preserves original behaviours)
    #
    #  Behaviour summary (kept from original file):
    #   - 'all' and 'api': special method names cause a full directory wipe for that path.
    #   - 'output': normal truncation of files; truncates files inside directories (no special wipe by name).
    #   - 'client': treats each path as a file and attempts to truncate it (matches original behaviour).
    # ---------------------------------------------------------
    def clear_all(self) -> None:
        paths = self._paths()

        for i, (method_name, raw) in enumerate(paths):
            if not raw:
                print(f"[skip] empty path @ index {i} ({method_name})")
                continue

            path = normalize_path(raw)

            # --- name-based special behaviour (directory wipe) ---
            if self.scope in ("all", "api") and method_name in self.special_method_names:
                print(f"[special] Clearing ALL contents in: {path} (method: {method_name})")

                if path and path.exists() and path.is_dir():
                    for item in path.iterdir():
                        try:
                            remove_path(item)
                            print(f" - removed: {item}")
                        except Exception as e:
                            print(f" - failed removing {item}: {e}")
                else:
                    print(" - directory missing or not a dir, skipping")
                continue

            # --- client behaviour: attempt to truncate each entry as a file ---
            if self.scope == "client":
                if not path or not path.exists():
                    print(f"[skip] path missing: {path} (method: {method_name})")
                    continue

                print(f"[truncate file] {path} (method: {method_name})")
                truncate_file(path)
                continue

            # --- normal behaviour for 'all'/'output' (and fallback) ---
            if not path or not path.exists():
                print(f"[skip] path missing: {path} (method: {method_name})")
                continue

            if path.is_file():
                print(f"[truncate file] {path} (method: {method_name})")
                truncate_file(path)

            elif path.is_dir():
                print(f"[truncate dir contents] {path} (method: {method_name})")
                for child in path.iterdir():
                    if child.is_file():
                        truncate_file(child)
                        print(f" - truncated {child}")

            else:
                print(f"[skip] not a normal path: {path} (method: {method_name})")


def main():
    mgr = CacheManager(scope="all")
    mgr_client = CacheManager(scope="client")
    mgr_api = CacheManager(scope="api")
    mgr_output = CacheManager(scope="output")

    while True:
        clear_screen()
        banner()
        menu_main()
        c = ask("> ")

        if c == "1":      # Clear Cache
            clear_screen()
            banner()
            menu_cache()
            c2 = ask("> ")

            if c2 == "1":  # ALL cache
                show_paths("Will clear", mgr.preview())
                confirm = ask("Proceed? (y/n) ")
                if confirm.lower() == "y":
                    mgr.clear_all()
                ask("Press Enter...")

            if c2 == "2":  # CLIENT cache
                show_paths("Will clear", mgr_client.preview())
                confirm = ask("Proceed? (y/n) ")
                if confirm.lower() == "y":
                    mgr_client.clear_all()
                ask("Press Enter...")

            if c2 == "3":  # API cache
                show_paths("Will clear", mgr_api.preview())
                confirm = ask("Proceed? (y/n) ")
                if confirm.lower() == "y":
                    mgr_api.clear_all()
                ask("Press Enter...")

            if c2 == "4":  # OUTPUT cache
                show_paths("Will clear", mgr_output.preview())
                confirm = ask("Proceed? (y/n) ")
                if confirm.lower() == "y":
                    mgr_output.clear_all()
                ask("Press Enter...")

            elif c2 == "5":
                continue


        elif c == "2":  # Check for Secrets

            clear_screen()

            banner()

            scan_for_secrets()

            ask("Press Enter...")

        elif c == "3":
            clear_screen()
            banner()
            settings_mgr = SettingsManager()

            preview_lines = settings_mgr.preview()

            if not preview_lines:
                print("No settings found.")
                ask("Press Enter...")
                continue

            show_paths("Will reset the following settings:", preview_lines)
            confirm = ask("Reset these settings to defaults? (y/n) ")

            if confirm.lower() == "y":
                settings_mgr.reset()

            ask("Press Enter...")

        elif c == "4":  # Tests
            tm = TestManager()
            tm.run()
            ask("Press Enter...")

        elif c == "5":  # Exit
            break


if __name__ == "__main__":
    main()
