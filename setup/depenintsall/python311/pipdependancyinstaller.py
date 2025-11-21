#!/usr/bin/env python3
# pipdependancyinstaller.py
# Reworked: preserves function names and advanced behavior (DLL registration, MSVC installer, CUDA detection),
# but focuses flow on: ensure .venv exists, attempt imports, on failure print .venv-targeted pip install command(s),
# and pause to allow copying.

from __future__ import annotations
import os
import sys
import glob
import json
import time
import shutil
import subprocess
import importlib
import importlib.util
import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Dict, Any

# -----------------------------
# Debug / config read helpers (kept simple)
# -----------------------------
def _get_call_file_path():
    # lightweight fallback to support your original call pattern if VersionFiles available
    try:
        from extra_models.Sulfur.TrainingScript.Build import call_file_path
        return call_file_path.Call()
    except Exception:
        class _Dummy:
            def settings_extra_debug(self): return os.environ.get("PDI_DEBUG", "no")
            def settings_pip_fallback_amount(self): return None
        return _Dummy()

call = _get_call_file_path()
try:
    file_path_settings_extra_debug = call.settings_extra_debug()
    if os.path.isfile(file_path_settings_extra_debug):
        with open(file_path_settings_extra_debug, "r", encoding="utf-8", errors="ignore") as f:
            print_debug = f.readline().strip().lower() == "yes"
    else:
        print_debug = str(file_path_settings_extra_debug).lower() in ("1", "yes", "true")
except Exception:
    print_debug = False

def detect_gpu_backend():
    """
    Lightweight system-level GPU detection used BEFORE torch is installed.
    Returns: ("cuda", info) or ("rocm", info) or ("directml", info) or ("none", None)
    Uses quick checks (nvidia-smi, rocminfo) so it works when torch is not installed.
    """
    try:
        # NVIDIA / CUDA check (fast): look for nvidia-smi
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return ("cuda", None)
    except Exception:
        pass

    try:
        # ROCm check: rocminfo present?
        proc = subprocess.run(["rocminfo"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return ("rocm", None)
    except Exception:
        pass

    # DirectML: check for torch-directml presence in venv (unlikely pre-install)
    # Fallback: on Windows with WSL or AMD integrated GPUs, treat as none
    return ("none", None)


def detect_gpu_backend():
    """
    Lightweight system-level GPU detection used BEFORE torch is installed.
    Returns: ("cuda", info) or ("rocm", info) or ("directml", info) or ("none", None)
    Uses quick checks (nvidia-smi, rocminfo) so it works when torch is not installed.
    """
    try:
        # NVIDIA / CUDA check (fast): look for nvidia-smi
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return ("cuda", None)
    except Exception:
        pass

    try:
        # ROCm check: rocminfo present?
        proc = subprocess.run(["rocminfo"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return ("rocm", None)
    except Exception:
        pass

    # DirectML: check for torch-directml presence in venv (unlikely pre-install)
    # Fallback: on Windows with WSL or AMD integrated GPUs, treat as none
    return ("none", None)


def install_torch_gpu_or_cpu(venv_python: str, print_debug: bool = False):
    """
    Attempt to pip-install an appropriate torch wheel using the provided venv_python path.
    This function prefers GPU wheels when a GPU backend is detected, otherwise installs CPU-only torch.
    """
    backend, info = detect_gpu_backend()
    if print_debug:
        print(f"[DEBUG]: detect_gpu_backend -> {backend}, info={info}")

    # Basic command building
    def run_install(cmd_list):
        try:
            if print_debug:
                print(f"[DEBUG]: running install: {' '.join(cmd_list)}")
            res = subprocess.run(cmd_list, capture_output=True, text=True)
            if print_debug:
                print(f"[DEBUG]: pip returncode: {res.returncode}")
                if res.stdout:
                    print(f"[DEBUG]: stdout: {res.stdout[:1000]}")
                if res.stderr:
                    print(f"[DEBUG]: stderr: {res.stderr[:1000]}")
            return res.returncode == 0
        except Exception as e:
            if print_debug:
                print(f"[DEBUG]: pip install exception: {e}")
            return False

    # Try GPU-first if detected
    if backend == "cuda":
        # NOTE: cu124 is an example; adapt index-url if you want a different cuda version.
        cmd = [venv_python, "-m", "pip", "install",
               "torch", "torchvision", "torchaudio",
               "--index-url", "https://download.pytorch.org/whl/cu124"]
        if run_install(cmd):
            return True

    if backend == "rocm":
        cmd = [venv_python, "-m", "pip", "install",
               "torch", "torchvision", "torchaudio",
               "--index-url", "https://download.pytorch.org/whl/rocm"]
        if run_install(cmd):
            return True

    # Fall back to CPU wheel
    if print_debug:
        print("[DEBUG]: No appropriate GPU wheel installed or no GPU detected — installing CPU-only torch.")
    cpu_cmd = [venv_python, "-m", "pip", "install",
               "torch", "torchvision", "torchaudio",
               "--index-url", "https://download.pytorch.org/whl/cpu"]
    return run_install(cpu_cmd)



# -----------------------------
# Name canonicalization + filesystem detection helpers
# -----------------------------
def canonicalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name).lower()
    n = name.strip().lower().replace('_', '-').replace('.', '-')
    alias_map = {
        "sklearn": "scikit-learn",
        "sklearnex": "scikit-learn-intelex",
        "llama": "llama-cpp-python",
        "llama-cpp": "llama-cpp-python",
        "llama_cpp": "llama-cpp-python",
        "llama_cpp_python": "llama-cpp-python",
        "pygame": "pygame-ce",
        "pygame_gui": "pygame-gui",
        "tf_keras": "tf-keras",
    }
    return alias_map.get(n, n)

def _generate_name_candidates(canon_name: str) -> Iterable[str]:
    yield canon_name
    yield canon_name.replace('-', '_')
    if canon_name == "scikit-learn":
        yield "sklearn"
    if canon_name == "scikit-learn-intelex":
        yield "sklearnex"
        yield "sklearn_x"
    if canon_name == "llama-cpp-python":
        yield "llama_cpp"
        yield "llama"
    if canon_name.startswith("pygame"):
        yield "pygame"

def find_distinfo_by_name(dist_name_like: str) -> bool:
    try:
        canon = canonicalize_name(dist_name_like)
        candidates = []
        for c in _generate_name_candidates(canon):
            if c not in candidates: candidates.append(c)
        for p in sys.path:
            if not p: continue
            try:
                if not os.path.isdir(p): continue
                for name in candidates:
                    pat_a = os.path.join(p, f"{name}*.dist-info")
                    pat_b = os.path.join(p, f"{name}*.egg-info")
                    if glob.glob(pat_a) or glob.glob(pat_b):
                        return True
                    pkg_dir = os.path.join(p, name)
                    if os.path.isdir(pkg_dir):
                        return True
            except Exception:
                continue
        for name in candidates:
            try:
                if importlib.util.find_spec(name) is not None:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False

def distribution_installed(dist_name: str) -> bool:
    try:
        importlib_metadata.distribution(dist_name)
        return True
    except Exception:
        return False

def top_level_package_provided(pkg_name: str) -> bool:
    try:
        mapping = importlib_metadata.packages_distributions()
        return bool(mapping.get(pkg_name))
    except Exception:
        return False

# -----------------------------
# Import fallback helper (namespace fallback)
# -----------------------------
def _import_with_namespace_fallback(import_name: str, namespace_prefixes=None, debug: bool = False):
    try:
        return importlib.import_module(import_name)
    except Exception as e:
        if debug:
            print(f"| DEBUG: primary import('{import_name}') failed: {e} |")
    variants = {import_name, import_name.replace("-", "_"), import_name.replace("_", "-")}
    if namespace_prefixes is None:
        namespace_prefixes = ("google", "google.cloud", "azure", "aws", "tensorflow", "torch")
    for var in variants:
        for ns in namespace_prefixes:
            candidate = f"{ns}.{var}"
            try:
                if importlib.util.find_spec(candidate):
                    try:
                        return importlib.import_module(candidate)
                    except Exception as e:
                        if debug:
                            print(f"| DEBUG: import('{candidate}') failed: {e} |")
                else:
                    if importlib.util.find_spec(ns):
                        try:
                            ns_mod = importlib.import_module(ns)
                            if hasattr(ns_mod, var):
                                return getattr(ns_mod, var)
                        except Exception:
                            pass
            except Exception:
                continue
    return None

# -----------------------------
# Venv management + site-packages discovery
# -----------------------------
def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in [current_file, *current_file.parents]:
        if (parent / "SulfurAI.py").exists() or (parent / "requirements.txt").exists():
            return parent
    # fallback to three levels up
    return current_file.parents[3]


def _restrict_to_project_venv(project_root, venv_dir):
    project_root = str(project_root)
    venv_dir = str(venv_dir)

    safe_paths = []
    venv_lower = venv_dir.lower()
    root_lower = project_root.lower()

    for p in sys.path:
        pl = str(p).lower()

        # ALWAYS keep the script directory (sys.path[0])
        if p == sys.path[0]:
            safe_paths.append(p)
            continue

        # Keep stdlib (no site-packages, contains 'python3X')
        if "python3" in pl and "site-packages" not in pl:
            safe_paths.append(p)
            continue

        # Keep the venv and its site-packages
        if pl.startswith(venv_lower):
            safe_paths.append(p)
            continue

        # Keep anything inside the project root
        if pl.startswith(root_lower):
            safe_paths.append(p)
            continue

    sys.path[:] = safe_paths


project_root = _get_project_root()
VENV_DIR = project_root / ".venv"

# Strictly enforce that ONLY the project venv is used
_restrict_to_project_venv(project_root, VENV_DIR)
CACHE_JSON_PATH = os.path.join(str(project_root), "pip_install_failures.json")
PIP_CMD = [sys.executable, "-m", "pip"]
ORIGINAL_PYTHON = sys.executable

def _venv_site_packages_path() -> Optional[str]:
    try:
        venv = str(VENV_DIR)
        if os.name == "nt":
            path = os.path.join(venv, "Lib", "site-packages")
        else:
            pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            path = os.path.join(venv, "lib", pyver, "site-packages")
        return path
    except Exception:
        return None

def prepend_project_venv_site_packages(verbose=True):
    try:
        project_dir = Path(__file__).resolve().parent
        max_up = 6
        cur = project_dir
        venv_path = None
        for _ in range(max_up):
            candidate = cur / ".venv"
            if candidate.exists() and candidate.is_dir():
                venv_path = candidate
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if venv_path is None:
            env_venv = os.environ.get("VIRTUAL_ENV")
            if env_venv:
                venv_path = Path(env_venv)
        if venv_path is None:
            try:
                if getattr(sys, "base_prefix", None) and sys.base_prefix != sys.prefix:
                    venv_path = Path(sys.prefix)
            except Exception:
                pass
        if venv_path is None:
            if verbose and print_debug:
                print("DEBUG: No project .venv found (searched parents, VIRTUAL_ENV, sys.prefix).")
            return False
        venv_path = venv_path.resolve()
        win_site = venv_path / "Lib" / "site-packages"
        posix_site = venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        site_candidates = [win_site, posix_site]
        try:
            import sysconfig
            sc = sysconfig.get_paths().get("purelib")
            if sc:
                site_candidates.append(Path(sc))
        except Exception:
            pass
        site_path = None
        for p in site_candidates:
            try:
                if p and p.exists() and p.is_dir():
                    site_path = str(p); break
            except Exception:
                continue
        if not site_path:
            if verbose:
                print(f"DEBUG: Could not resolve site-packages for venv: {venv_path}")
            return False
        if site_path not in sys.path:
            sys.path.insert(0, site_path)
            if verbose and print_debug:
                print(f"DEBUG: Prepending venv site-packages to sys.path: {site_path}")
                print(f"DEBUG: sys.executable: {sys.executable}")
        else:
            if verbose and print_debug:
                print(f"DEBUG: venv site-packages already in sys.path: {site_path}")
        return True
    except Exception as e:
        if verbose and print_debug:
            print("DEBUG: prepend_project_venv_site_packages failed:", repr(e))
        return False

# Try to prepend early
prepend_project_venv_site_packages(verbose=True)

# -----------------------------
# Site-packages scan utility
# -----------------------------
def _scan_site_packages(site_path: str) -> Tuple[Set[str], Set[str]]:
    top_level_modules = set()
    dist_names = set()
    if not site_path or not os.path.isdir(site_path):
        return top_level_modules, dist_names
    try:
        for name in os.listdir(site_path):
            full = os.path.join(site_path, name)
            if name.endswith(".dist-info") or name.endswith(".egg-info"):
                dist_key = name.rsplit(".", 1)[0].lower()
                dist_names.add(dist_key)
                try:
                    tl = os.path.join(full, "top_level.txt")
                    if os.path.isfile(tl):
                        with open(tl, "r", encoding="utf-8", errors="ignore") as f:
                            for ln in f:
                                ln = ln.strip()
                                if ln: top_level_modules.add(ln)
                except Exception:
                    pass
                try:
                    metadata_file = os.path.join(full, "METADATA")
                    if os.path.isfile(metadata_file):
                        with open(metadata_file, "r", encoding="utf-8", errors="ignore") as mf:
                            for line in mf:
                                if line.lower().startswith("name:"):
                                    name_val = line.split(":", 1)[1].strip()
                                    if name_val:
                                        dist_names.add(name_val.lower())
                                    break
                except Exception:
                    pass
            elif os.path.isdir(full):
                top_level_modules.add(name)
            else:
                if name.endswith(".egg-info"):
                    dist_names.add(name.rsplit(".", 1)[0].lower())
    except Exception:
        pass
    return top_level_modules, dist_names

# -----------------------------
# Native DLL / shared lib registration helpers (DAAL / llama)
# -----------------------------
def ensure_native_dirs_registered(package_names, venv_site=None, print_debug=True) -> bool:
    try:
        added = set()
        if not venv_site:
            try:
                import sysconfig
                venv_site = sysconfig.get_paths().get("purelib")
            except Exception:
                venv_site = None
        if not venv_site or not os.path.isdir(venv_site):
            try:
                vexe = Path(sys.executable).resolve()
                root = vexe.parents[1]
                cand1 = os.path.join(str(root), "Lib", "site-packages")
                pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
                cand2 = os.path.join(str(root), "lib", pyver, "site-packages")
                if os.path.isdir(cand1):
                    venv_site = cand1
                elif os.path.isdir(cand2):
                    venv_site = cand2
            except Exception:
                venv_site = None
        if not venv_site or not os.path.isdir(venv_site):
            if print_debug:
                print("| DEBUG: ensure_native_dirs_registered: venv site-packages not found |")
            return False
        pkg_candidates = []
        for nm in package_names:
            if not nm: continue
            d = os.path.join(venv_site, nm)
            if os.path.isdir(d):
                pkg_candidates.append(d)
            d2 = os.path.join(venv_site, nm.replace("-", "_"))
            if os.path.isdir(d2) and d2 not in pkg_candidates:
                pkg_candidates.append(d2)
        try:
            for ent in os.listdir(venv_site):
                if ent.endswith(".dist-info"):
                    tl = os.path.join(venv_site, ent, "top_level.txt")
                    if os.path.isfile(tl):
                        try:
                            with open(tl, "r", encoding="utf-8", errors="ignore") as fh:
                                for ln in fh:
                                    ln = ln.strip()
                                    if not ln: continue
                                    cand_dir = os.path.join(venv_site, ln)
                                    if os.path.isdir(cand_dir) and cand_dir not in pkg_candidates:
                                        pkg_candidates.append(cand_dir)
                        except Exception:
                            continue
        except Exception:
            pass
        if print_debug:
            print(f"| DEBUG: ensure_native_dirs_registered: package candidates: {pkg_candidates} |")
        for pkg_dir in pkg_candidates:
            for root, dirs, files in os.walk(pkg_dir):
                for fn in files:
                    if fn.lower().endswith((".dll", ".pyd", ".so")):
                        parent = os.path.abspath(root)
                        if parent in added: continue
                        try:
                            if sys.platform.startswith("win"):
                                try:
                                    os.add_dll_directory(parent)
                                    if print_debug:
                                        print(f"| DEBUG: os.add_dll_directory({parent}) called |")
                                except Exception as e:
                                    os.environ["PATH"] = parent + os.pathsep + os.environ.get("PATH", "")
                                    if print_debug:
                                        print(f"| DEBUG: os.add_dll_directory failed, prepended {parent} to PATH: {e} |")
                            else:
                                os.environ["LD_LIBRARY_PATH"] = parent + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                                if print_debug:
                                    print(f"| DEBUG: prepended {parent} to LD_LIBRARY_PATH (best-effort) |")
                            added.add(parent)
                        except Exception as e:
                            if print_debug:
                                print(f"| DEBUG: failed to register native dir {parent}: {e} |")
        return bool(added)
    except Exception as e:
        if print_debug:
            print(f"| DEBUG: ensure_native_dirs_registered exception: {e} |")
        return False

def ensure_daal_libs_registered(print_debug: bool = False) -> bool:
    try:
        if os.name != "nt":
            if print_debug:
                print("| ensure_daal_libs_registered: non-Windows, skipping |")
            return False
        sitep = _venv_site_packages_path()
        if not sitep or not os.path.isdir(sitep):
            try:
                venv_root = Path(sys.executable).resolve().parents[1]
                cand_win = os.path.join(str(venv_root), "Lib", "site-packages")
                if os.path.isdir(cand_win):
                    sitep = cand_win
            except Exception:
                pass
        if not sitep or not os.path.isdir(sitep):
            if print_debug:
                print("| ensure_daal_libs_registered: could not determine site-packages |")
            return False
        if print_debug:
            print(f"| ensure_daal_libs_registered: scanning {sitep} for DAAL/pyd/dlls |")
        candidates = set()
        for root, dirs, files in os.walk(sitep):
            for fname in files:
                lf = fname.lower()
                if lf.startswith("_daal4py") and (lf.endswith(".pyd") or lf.endswith(".dll")):
                    candidates.add(root)
                elif "daal" in lf and (lf.endswith(".pyd") or lf.endswith(".dll")):
                    candidates.add(root)
                elif lf.endswith(".pyd") and "_daal" in lf:
                    candidates.add(root)
            for dname in dirs:
                dn = dname.lower()
                if dn == "daal" or dn.startswith("daal") or dn == "daal.libs":
                    candidates.add(os.path.join(root, dname))
        for name in ("daal", "daal4py", "scikit_learn_intelex", "scikit_learn_intelex-"):
            p = os.path.join(sitep, name)
            if os.path.isdir(p):
                for sub in ("lib", "libs", ""):
                    cand = os.path.join(p, sub) if sub else p
                    if os.path.isdir(cand):
                        candidates.add(cand)
        if not candidates:
            if print_debug:
                print("| ensure_daal_libs_registered: no candidate DLL dirs found |")
            return False
        registered_any = False
        for c in sorted(candidates):
            try:
                c_abs = os.path.abspath(c)
                if print_debug:
                    print(f"| ensure_daal_libs_registered: attempting add_dll_directory('{c_abs}') |")
                try:
                    os.add_dll_directory(c_abs)
                except Exception:
                    pass
                path_env = os.environ.get("PATH", "")
                if c_abs not in path_env:
                    os.environ["PATH"] = c_abs + os.pathsep + path_env
                registered_any = True
            except Exception as e:
                if print_debug:
                    print(f"| ensure_daal_libs_registered: failed for {c}: {e} |")
                continue
        if registered_any and print_debug:
            print("| ensure_daal_libs_registered: registered candidate DAAL dirs. |")
        return registered_any
    except Exception as e:
        if print_debug:
            print(f"| ensure_daal_libs_registered: unexpected error: {e} |")
        return False

def ensure_llama_dll_registered(print_debug=True) -> bool:
    try:
        sitep = _venv_site_packages_path()
        if not sitep or not os.path.isdir(sitep):
            try:
                venv_root = Path(sys.executable).resolve().parents[1]
                cand_win = os.path.join(str(venv_root), "Lib", "site-packages")
                pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
                cand_posix = os.path.join(str(venv_root), "lib", pyver, "site-packages")
                if os.path.isdir(cand_win):
                    sitep = cand_win
                elif os.path.isdir(cand_posix):
                    sitep = cand_posix
            except Exception:
                sitep = None
        if not sitep:
            if print_debug:
                print("| DEBUG: ensure_llama_dll_registered: could not determine venv site-packages |")
            return False
        if print_debug:
            print(f"| DEBUG: ensure_llama_dll_registered: examining site-packages: {sitep} |")
        candidates = []
        exact = os.path.join(sitep, "llama_cpp")
        if os.path.isdir(exact):
            candidates.append(exact)
        else:
            for ent in os.listdir(sitep):
                ent_path = os.path.join(sitep, ent)
                lower = ent.lower()
                if (os.path.isdir(ent_path) and (lower.startswith("llama") or lower.startswith("llama_cpp"))):
                    candidates.append(ent_path)
        if not candidates:
            if print_debug:
                print("| DEBUG: ensure_llama_dll_registered: no llama package directory found in venv site-packages |")
            return False
        added_dirs = set()
        for pkg_dir in candidates:
            for root, dirs, files in os.walk(pkg_dir):
                for fn in files:
                    if fn.lower().endswith((".dll", ".pyd", ".so")):
                        parent = os.path.abspath(root)
                        if parent not in added_dirs:
                            try:
                                if sys.platform.startswith("win"):
                                    try:
                                        os.add_dll_directory(parent)
                                        if print_debug:
                                            print(f"| DEBUG: os.add_dll_directory({parent}) registered for llama native libs |")
                                    except Exception as e:
                                        os.environ["PATH"] = parent + os.pathsep + os.environ.get("PATH", "")
                                        if print_debug:
                                            print(f"| DEBUG: os.add_dll_directory failed; prepended {parent} to PATH: {e} |")
                                else:
                                    os.environ["LD_LIBRARY_PATH"] = parent + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                                    if print_debug:
                                        print(f"| DEBUG: prepended {parent} to LD_LIBRARY_PATH (runtime) |")
                                added_dirs.add(parent)
                            except Exception as e:
                                if print_debug:
                                    print(f"| DEBUG: failed to register DLL dir {parent}: {e} |")
        if not added_dirs:
            if print_debug:
                print("| DEBUG: ensure_llama_dll_registered: no native DLL/.pyd files found under candidates |")
            return False
        # optional patching behavior omitted for simplicity (safe)
        return True
    except Exception as e:
        if print_debug:
            print("| DEBUG: ensure_llama_dll_registered exception:", repr(e))
        return False

# -----------------------------
# MSVC Build Tools helper (Windows)
# -----------------------------
def install_msvc_build_tools(print_debug=True):
    try:
        if shutil.which("cl.exe"):
            if print_debug:
                print("| MSVC already installed (cl.exe present) — skipping install. |")
            return True
        url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
        installer_path = os.path.join(os.getenv("TEMP") or ".", "vs_buildtools.exe")
        if not os.path.exists(installer_path):
            if print_debug:
                print(f"| Downloading Visual Studio Build Tools from {url} -> {installer_path} |")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, installer_path)
            except Exception as e:
                if print_debug:
                    print(f"| ERROR: failed to download MSVC installer: {e} |")
                return False
        cmd = [
            installer_path,
            "--quiet", "--wait", "--norestart", "--nocache",
            "--add", "Microsoft.VisualStudio.Workload.VCTools",
            "--includeRecommended"
        ]
        if print_debug:
            print("| Installing Visual Studio Build Tools (silent). This may take a while. |")
            print(f"| DEBUG: running: {cmd} |")
        result = subprocess.run(cmd)
        if result.returncode == 3010:
            if print_debug:
                print("| Visual Studio Build Tools installed but a reboot is required (code 3010). |")
            return 3010
        elif result.returncode == 0:
            if print_debug:
                print("| Visual Studio Build Tools installed successfully. |")
            return True
        else:
            if print_debug:
                print(f"| ERROR: Visual Studio Build Tools installer returned code {result.returncode} |")
            return False
    except Exception as e:
        if print_debug:
            print(f"| Exception during MSVC install: {repr(e)} |")
        return False

# -----------------------------
# Python/venv helpers
# -----------------------------
def get_python_for_venv_actions(print_debug=False) -> Optional[str]:
    try:
        if ORIGINAL_PYTHON and os.path.exists(ORIGINAL_PYTHON):
            return ORIGINAL_PYTHON
    except Exception:
        pass
    for name in ("python", "python3"):
        path = shutil.which(name)
        if path:
            if print_debug:
                print(f"| get_python_for_venv_actions: using fallback {path} |")
            return path
    try:
        if sys.executable and os.path.exists(sys.executable):
            return sys.executable
    except Exception:
        pass
    return None

def ensure_venv_and_reexec():
    global PIP_CMD
    in_venv = getattr(sys, "base_prefix", None) != getattr(sys, "prefix", None)
    try:
        if in_venv and os.path.dirname(sys.executable).startswith(str(VENV_DIR)):
            PIP_CMD = [sys.executable, "-m", "pip"]
            return
    except Exception:
        pass
    if not os.path.isdir(VENV_DIR):
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
            if print_debug:
                print(f"| Created virtualenv at {VENV_DIR} |")
        except Exception as e:
            print(f"| ERROR: Could not create venv: {e} |")
            return
    venv_python = os.path.join(str(VENV_DIR), "Scripts", "python.exe") if os.name == "nt" else os.path.join(str(VENV_DIR), "bin", "python")
    if not os.path.exists(venv_python):
        print("| ERROR: venv python not found after creation. Falling back to system python. |")
        return
    sys.executable = venv_python
    PIP_CMD = [venv_python, "-m", "pip"]
    if print_debug:
        print(f"| Using virtualenv python: {venv_python} |")

def repair_venv_pip(print_debug=False) -> bool:
    global PIP_CMD
    venv_python = os.path.join(str(VENV_DIR), "Scripts", "python.exe") if os.name == "nt" else os.path.join(str(VENV_DIR), "bin", "python")
    if os.path.exists(venv_python):
        try:
            subprocess.check_call([venv_python, "-m", "ensurepip", "--upgrade"])
        except Exception:
            pass
        try:
            subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            PIP_CMD = [venv_python, "-m", "pip"]
            if print_debug:
                print(f"| repair_venv_pip: pip upgraded for {venv_python} |")
            return True
        except Exception as e:
            if print_debug:
                print(f"| repair_venv_pip: pip upgrade failed for {venv_python}: {e} |")
    python_for_create = get_python_for_venv_actions(print_debug=print_debug)
    if not python_for_create:
        if print_debug:
            print("| repair_venv_pip: no python available to recreate venv |")
        return False
    try:
        if os.path.isdir(VENV_DIR):
            try:
                shutil.rmtree(VENV_DIR)
            except Exception:
                if print_debug:
                    print("| repair_venv_pip: could not remove existing .venv directory, attempting creation anyway |")
        subprocess.check_call([python_for_create, "-m", "venv", str(VENV_DIR)])
        venv_python = os.path.join(str(VENV_DIR), "Scripts", "python.exe") if os.name == "nt" else os.path.join(str(VENV_DIR), "bin", "python")
        subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        PIP_CMD = [venv_python, "-m", "pip"]
        if print_debug:
            print(f"| repair_venv_pip: recreated and upgraded venv using {python_for_create} -> {venv_python} |")
        return True
    except Exception as e:
        if print_debug:
            print(f"| repair_venv_pip: recreate-and-upgrade failed: {e} |")
        return False

# -----------------------------
# pip runner helper
# -----------------------------
def run_pip_capture(args, capture_output=True):
    full_cmd = (PIP_CMD if isinstance(PIP_CMD, list) else [sys.executable, "-m", "pip"]) + args
    try:
        if capture_output:
            proc = subprocess.run(full_cmd, capture_output=True, text=True)
        else:
            proc = subprocess.run(full_cmd)
        return proc
    except Exception as e:
        class _Fake:
            returncode = 1
            stdout = ""
            stderr = str(e)
        return _Fake()

# -----------------------------
# Failure cache
# -----------------------------
def save_failure_cache(d: Dict[str, Any]):
    try:
        with open(CACHE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass

def load_failure_cache() -> Dict[str, Dict[str, int]]:
    try:
        if not os.path.exists(CACHE_JSON_PATH):
            return {}
        with open(CACHE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        normalized = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if v is None:
                    normalized[k] = {"ts": None, "count": 0}
                elif isinstance(v, dict):
                    normalized[k] = {"ts": v.get("ts"), "count": int(v.get("count", 0))}
                elif isinstance(v, (list, tuple)):
                    ts_val, cnt = (v[0] if len(v) > 0 else None, int(v[1]) if len(v) > 1 else 0)
                    normalized[k] = {"ts": ts_val, "count": cnt}
                else:
                    try:
                        normalized[k] = {"ts": None, "count": int(v)}
                    except Exception:
                        normalized[k] = {"ts": None, "count": 0}
        else:
            normalized = {}
        save_failure_cache(normalized)
        return normalized
    except Exception:
        try:
            shutil.copy2(CACHE_JSON_PATH, CACHE_JSON_PATH + ".bak")
        except Exception:
            pass
        return {}

# -----------------------------
# Simple OS helpers
# -----------------------------
def ensure_windows_long_paths():
    if os.name != "nt":
        return
    try:
        import winreg
        key = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key, 0, winreg.KEY_READ) as reg_key:
            value, _ = winreg.QueryValueEx(reg_key, "LongPathsEnabled")
            if value == 1:
                print("| Windows long path support already enabled |")
                return
    except Exception:
        pass
    try:
        import winreg
        key = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key, 0, winreg.KEY_SET_VALUE) as reg_key:
            winreg.SetValueEx(reg_key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
        print("| Windows long path support has been enabled. Please restart your PC for it to take effect. |")
        time.sleep(10)
    except PermissionError:
        print("| ERROR: Windows long path support is not enabled. |")
        print("| Please run this script as Administrator once to enable it, or enable manually via Group Policy / Registry. |")
        time.sleep(10)
    except Exception as e:
        print(f"| WARNING: Could not enable long path support automatically: {e} |")
        time.sleep(10)

# -----------------------------
# High-level import check and install suggestion flow
# -----------------------------

def _add_global_path():
    """
    Create .pth files so the directory containing SulfurAI.py is added to sys.path
    for system site-packages and project/venv site-packages (best-effort).
    Safe/no-op if SulfurAI.py not found or write permission missing.
    """
    from pathlib import Path
    import sysconfig
    import site
    import os


    try:
        # Prefer the previously-determined project_root if available in this module
        pr = globals().get("project_root")
        if pr is None:
            # fallback: resolve relative to this script
            script_path = Path(__file__).resolve()
            for parent in (script_path, *script_path.parents):
                if (parent / "SulfurAI.py").exists():
                    pr = parent
                    break
        if pr is None:
            if print_debug:
                print("[DEBUG]: _add_global_path: could not determine project root (no SulfurAI.py found).")
            return

        sulfurai_path = Path(pr) / "SulfurAI.py"
        if not sulfurai_path.exists():
            if print_debug:
                print(f"[DEBUG]: Could not find SulfurAI.py at {sulfurai_path}")
            return

        if print_debug:
            print(f"[DEBUG]: Found SulfurAI.py at: {sulfurai_path}")

        # directory to add
        path_to_add = sulfurai_path.parent.resolve()

        targets = []

        # system site-packages (purelib)
        try:
            system_site = sysconfig.get_paths().get("purelib")
            if system_site:
                targets.append(Path(system_site) / "sulfurai.pth")
        except Exception:
            system_site = None

        # site.getsitepackages() may include virtualenv or other site-packages
        try:
            venv_sites = site.getsitepackages()
            for v in venv_sites:
                if not v:
                    continue
                # avoid duplicate of system_site
                if system_site and os.path.abspath(v) == os.path.abspath(system_site):
                    continue
                targets.append(Path(v) / "sulfurai.pth")
        except Exception:
            # if site.getsitepackages() failed (e.g. in some virtualenvs), try to add the project venv site-packages
            try:
                venv_site = globals().get("_venv_site_packages_path") and _venv_site_packages_path()
                if venv_site:
                    targets.append(Path(venv_site) / "sulfurai.pth")
            except Exception:
                pass

        # also attempt the project .venv site-packages (if VENV_DIR set)
        try:
            venv_dir = globals().get("VENV_DIR")
            if venv_dir:
                if os.name == "nt":
                    cand = Path(venv_dir) / "Lib" / "site-packages"
                else:
                    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
                    cand = Path(venv_dir) / "lib" / pyver / "site-packages"
                if cand.exists():
                    targets.append(cand / "sulfurai.pth")
        except Exception:
            pass

        # Deduplicate targets while preserving order
        seen = set()
        deduped = []
        for t in targets:
            try:
                key = str(t.resolve())
            except Exception:
                key = str(t)
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        targets = deduped

        # Write .pth files
        for pth_file in targets:
            try:
                pth_file.parent.mkdir(parents=True, exist_ok=True)
                with open(pth_file, "w", encoding="utf-8") as f:
                    f.write(str(path_to_add) + "\n")
                if print_debug:
                    print(f"[DEBUG]: Added {path_to_add} to {pth_file}")
            except PermissionError:
                if print_debug:
                    print(f"[DEBUG]: Could not write to {pth_file} (need admin/sudo?)")
            except Exception as e:
                if print_debug:
                    print(f"[DEBUG]: Failed to write {pth_file}: {e}")

        if print_debug:
            print("\n[DEBUG]: You can now `import SulfurAI` inside this venv AND outside with system Python!")
        print("\n| Python sys.path add-on succeeded. You're doubly good to go. |")
    except Exception as e:
        if print_debug:
            print(f"[DEBUG]: _add_global_path unexpected error: {e}")



class run:
    import importlib
    importlib.invalidate_caches()

    @staticmethod
    def is_pip_outdated() -> bool:
        try:
            installed = importlib_metadata.version("pip")
            latest = subprocess.check_output([sys.executable, "-m", "pip", "index", "versions", "pip"], stderr=subprocess.DEVNULL).decode()
            return installed not in latest
        except Exception:
            return False



    def run(self, print_debug: bool = False, RUN_INSTALLER: bool = True):
        """Main flow:
           - ensure .venv
           - check imports using MODULE_IMPORT_MAP
           - try helpful fixes (dll registration / repair pip) where appropriate
           - if missing, build pip install commands into .venv and print them
        """
        # ensure .venv exists and point PIP_CMD to it
        ensure_venv_and_reexec()
        # ensure site-packages prioritized
        prepend_project_venv_site_packages(verbose=print_debug)

        # mapping: module import name -> pip token + extras
        MODULE_IMPORT_MAP = {
            # Core ML / AI
            "sklearn": {"package": "scikit-learn", "imports": ["sklearn"], "extras": []},
            "sklearnex": {"package": "scikit-learn-intelex", "imports": ["sklearnex"], "extras": []},
            "torch": {"package": "torch", "imports": ["torch"], "extras": ["torchvision", "torchaudio"], "special": "torch_backend"},
            "torchvision": {"package": "torchvision", "imports": ["torchvision"], "extras": []},
            "torchaudio": {"package": "torchaudio", "imports": ["torchaudio"], "extras": []},
            "tensorflow": {"package": "tensorflow", "imports": ["tensorflow"], "extras": []},
            "tf_keras": {"package": "tf-keras", "imports": ["tf_keras", "keras"], "extras": []},
            "cupy": {"package": "cupy-cuda12x", "imports": ["cupy"], "extras": []},
            "xgboost": {"package": "xgboost", "imports": ["xgboost"], "extras": []},

            # Data processing / stats
            "numpy": {"package": "numpy", "imports": ["numpy"], "extras": []},
            "pandas": {"package": "pandas", "imports": ["pandas"], "extras": []},
            "textstat": {"package": "textstat", "imports": ["textstat"], "extras": []},

            # Natural Language Processing
            "nltk": {"package": "nltk", "imports": ["nltk"], "extras": []},
            "textblob": {"package": "textblob", "imports": ["textblob"], "extras": []},
            "spacy": {"package": "spacy", "imports": ["spacy"], "extras": []},
            "transformers": {"package": "transformers", "imports": ["transformers"], "extras": []},
            "sentencepiece": {"package": "sentencepiece", "imports": ["sentencepiece"], "extras": []},
            "safetensors": {"package": "safetensors", "imports": ["safetensors"], "extras": []},
            "fasttext": {"package": "fasttext-wheel", "imports": ["fasttext"], "extras": []},
            "langdetect": {"package": "langdetect", "imports": ["langdetect"], "extras": []},
            "langcodes": {"package": "langcodes", "imports": ["langcodes"], "extras": []},
            "pycountry": {"package": "pycountry", "imports": ["pycountry"], "extras": []},
            "better_profanity": {"package": "better-profanity", "imports": ["better_profanity"], "extras": []},
            "vaderSentiment": {"package": "vaderSentiment", "imports": ["vaderSentiment"], "extras": []},
            "language_tool_python": {"package": "language_tool_python", "imports": ["language_tool_python"],
                                     "extras": []},

            # GPU / compute acceleration
            "accelerate": {"package": "accelerate", "imports": ["accelerate"], "extras": []},
            "bitsandbytes": {"package": "bitsandbytes", "imports": ["bitsandbytes"], "extras": []},

            # Trends / web data
            "pytrends": {"package": "pytrends", "imports": ["pytrends"], "extras": []},

            # Utility / async / timeouts
            "func_timeout": {"package": "func_timeout", "imports": ["func_timeout"], "extras": []},
            "tqdm": {"package": "tqdm", "imports": ["tqdm"], "extras": []},

            # GUI / visual / fun
            "pygame": {"package": "pygame-ce", "imports": ["pygame"], "extras": []},
            "pygame_gui": {"package": "pygame-gui", "imports": ["pygame_gui"], "extras": []},
            "art": {"package": "art", "imports": ["art"], "extras": []},
            "emoji": {"package": "emoji", "imports": ["emoji"], "extras": []},

            # LLM / AI interfaces
            "llama_cpp": {"package": "llama-cpp-python", "imports": ["llama_cpp"], "extras": []},
            "hf_xet": {"package": "hf_xet", "imports": ["hf_xet"], "extras": []},
            "google-genai": {"package": "google-genai", "imports": ["google.genai", "google_genai"], "extras": []},

            # Security / crypto
            "cryptography": {"package": "cryptography", "imports": ["cryptography"], "extras": []},

            # Server Hosting
            "flask": {"package": "flask", "imports": ["flask"], "extras": []},
            "uvicorn": {"package": "uvicorn", "imports": ["uvicorn"], "extras": []},


            # Test

        }

        compatibility_rules = {
            "transformers": {
                "safe_range": ">=4.0.0,<5.0.0",
                "dependencies": {"huggingface-hub": "<1.0"},
                "latest_version": "4.57.1"
            },
            "huggingface-hub": {
                "safe_range": ">=0.30.0,<1.0",
                "dependencies": {},
                "latest_version": "0.30.0"
            },
            "peft": {
                "safe_range": ">=0.15.2,<1.0",
                "dependencies": {"huggingface-hub": ">=0.25.0"},
                "latest_version": "0.15.2"
            }
            # (Add other packages as needed)
        }

        def check_version_compatibility(print_debug=False):
            """
            Check installed package versions against the compatibility_rules.
            Returns a list of package specifiers that need to be fixed.
            """
            from packaging.specifiers import SpecifierSet
            fixes = []
            for pkg, rules in compatibility_rules.items():
                try:
                    curr_ver = importlib_metadata.version(pkg)
                except importlib_metadata.PackageNotFoundError:
                    # Skip if not installed
                    continue

                # Check the package's own safe range
                safe_spec = rules.get("safe_range", "")
                if safe_spec:
                    spec_set = SpecifierSet(safe_spec)
                    if curr_ver not in spec_set:
                        fixes.append(f"{pkg}{safe_spec}")
                        if print_debug:
                            print(f"| DEBUG: {pkg} version {curr_ver} not in {safe_spec} |")

                # Check each declared dependency range
                for dep, dep_spec in rules.get("dependencies", {}).items():
                    try:
                        dep_ver = importlib_metadata.version(dep)
                    except importlib_metadata.PackageNotFoundError:
                        # Skip if dependency not installed
                        continue
                    spec_set = SpecifierSet(dep_spec)
                    if dep_ver not in spec_set:
                        fixes.append(f"{dep}{dep_spec}")
                        if print_debug:
                            print(f"| DEBUG: dependency {dep} version {dep_ver} not in {dep_spec} |")

            return sorted(set(fixes))

        # Build list of modules to attempt (module name keys)
        modules_to_check = list(MODULE_IMPORT_MAP.keys())

        missing_pkgs: Set[str] = set()
        missing_imports: Dict[str, str] = {}

        # persistent failure tracking
        cache = load_failure_cache()

        # attempt to import each module, applying heuristics
        def fast_import_check(import_names, debug=False):
            import importlib.util
            for name in import_names:
                try:
                    spec = importlib.util.find_spec(name)
                except:
                    spec = None

                if spec is None:
                    if debug:
                        print(f"[DEBUG] find_spec failed for {name}")
                    continue

                # Module exists — treat as installed even if import fails
                return True

            return False

        missing_pkgs = set()
        missing_imports = {}

        for module_key in modules_to_check:
            entry = MODULE_IMPORT_MAP[module_key]
            pkg = entry["package"]
            imports = entry.get("imports", [module_key])
            extras = entry.get("extras", [])

            # Quick import check (venv-first due to earlier prepend/restrict)
            if fast_import_check(imports, debug=print_debug):
                continue

            # If module defines a special hook, run it BEFORE marking missing.
            special = entry.get("special")
            if special == "torch_backend":
                # Prepare venv python path
                venv_python = os.path.join(str(VENV_DIR), "Scripts", "python.exe") if os.name == "nt" \
                    else os.path.join(str(VENV_DIR), "bin", "python")
                try:
                    if print_debug:
                        print("| DEBUG: Running special installer for torch_backend |")
                    ok = install_torch_gpu_or_cpu(venv_python, print_debug=print_debug)
                    if ok:
                        # Re-check imports after attempting install
                        if fast_import_check(imports, debug=print_debug):
                            if print_debug:
                                print("| DEBUG: torch imports now available after special install |")
                            # also remove extras from missing list if they exist
                            continue
                except Exception as e:
                    if print_debug:
                        print(f"| DEBUG: special torch installer failed: {e} |")

            # Still missing — register
            missing_pkgs.add(pkg)
            for ex in extras:
                missing_pkgs.add(ex)
            missing_imports[module_key] = pkg

            # failure cache update
            ent = cache.get(module_key, {"ts": None, "count": 0})
            ent["count"] = ent.get("count", 0) + 1
            cache[module_key] = ent

        # Save updated cache
        try:
            save_failure_cache(cache)
        except Exception:
            pass

        # If nothing missing, report success
        if not missing_pkgs:
            print("| All required imports detected in the current .venv / interpreter. |")

            # ✅ Check for incompatible installed versions
            fix_list = check_version_compatibility(print_debug)
            if fix_list:
                # Prepare fix suggestions similar to missing package commands
                venv_python_win = os.path.join(str(VENV_DIR), "Scripts", "python.exe")
                venv_python_unix = os.path.join(str(VENV_DIR), "bin", "python")
                win_abs = os.path.abspath(venv_python_win)
                unix_abs = os.path.abspath(venv_python_unix)
                fixes = " ".join(fix_list)
                quoted_fixes = [
                    f'"{pkg}"' if any(op in pkg for op in "<>,=") else pkg
                    for pkg in fix_list
                ]
                fixes_str = " ".join(quoted_fixes)

                powershell_cmd = f'& "{win_abs}" -m pip install {fixes_str}'
                cmd_cmd = f'"{win_abs}" -m pip install {fixes_str}'
                bash_cmd = f'"{unix_abs}" -m pip install {fixes_str}'

                print("\n| Incompatible package versions detected. |")


                print(
                    "| Install using one of the following commands (run from project root / adjust path if needed): |\n")

                # ------------------------------------------------------------------------------------
                # OPTION 1 — Install via requirements.txt
                # ------------------------------------------------------------------------------------
                print("OPTION 1 — Install using requirements.txt (recommended)")
                print("--------------------------------------------------------")
                print("  Windows (PowerShell):")
                print(f"      .\\{VENV_DIR.name}\\Scripts\\Activate.ps1")
                print("      pip install -r requirements.txt\n")

                print("  Windows (CMD):")
                print(f"      .\\{VENV_DIR.name}\\Scripts\\activate.bat")
                print("      pip install -r requirements.txt\n")

                print("  Linux / macOS (Bash / Zsh):")
                print(f"      source ./{VENV_DIR.name}/bin/activate")
                print("      pip install -r requirements.txt\n")

                # ------------------------------------------------------------------------------------
                # OPTION 2 — Direct pip installation of only the required fixes
                # ------------------------------------------------------------------------------------
                print("OPTION 2 — Install only the required packages")
                print("--------------------------------------------------------")
                print("  Windows (CMD):")
                print(f"    {cmd_cmd}\n")

                print("  Windows (PowerShell):")
                print(f"    {powershell_cmd}\n")

                print("  Linux / macOS (Bash / Zsh):")
                print(f"    {bash_cmd}\n")

                # ------------------------------------------------------------------------------------
                # NOTES
                # ------------------------------------------------------------------------------------
                print("| Notes: |")
                print("  - Option 1 ensures a full, reproducible environment.")
                print("  - Option 2 installs only the missing/incompatible packages shown above.")
                print("  - In PowerShell you should use the call operator (&) before a quoted executable path.")
                print("  - CMD and Bash accept the quoted-path form shown above.")
                print("  - You may activate the virtualenv first, then run pip manually:")
                print(f"      source ./{VENV_DIR.name}/bin/activate  # (bash/zsh)")
                print(f"      .\\{VENV_DIR.name}\\Scripts\\Activate.ps1  # (PowerShell)\n")

                print("| Inspect commands before running. They will install into the project's .venv. |\n")

                import time
                time.sleep(100)
                exit()

            return []

        # Try some targeted repairs for specific popular failure modes:
        # - If llama present but import fails due to native libs -> attempt registration
        if "llama-cpp-python" in missing_pkgs or "llama_cpp" in missing_imports:
            try:
                ok = ensure_llama_dll_registered(print_debug=print_debug)
                if ok and print_debug:
                    print("| DEBUG: ensure_llama_dll_registered succeeded (retry imports may now work). |")
            except Exception:
                pass

        # - If sklearnex / daal issues detected -> register daal libs
        if "scikit-learn-intelex" in missing_pkgs or "sklearnex" in missing_imports:
            try:
                ok = ensure_daal_libs_registered(print_debug=print_debug)
                if ok and print_debug:
                    print("| DEBUG: ensure_daal_libs_registered succeeded (retry imports may now work). |")
            except Exception:
                pass

        # - Attempt to repair venv pip if pip errors observed (best-effort)
        try:
            repair_venv_pip(print_debug=print_debug)
        except Exception:
            pass

        # After targeted attempts, we still expect that user likely needs to pip-install packages.
        sorted_pkgs = sorted(missing_pkgs)

        # Build package list string
        pkg_list = " ".join(sorted_pkgs)

        # Resolve venv python paths (absolute) for both Windows and Unix forms
        venv_python_win = os.path.join(str(VENV_DIR), "Scripts", "python.exe")
        venv_python_unix = os.path.join(str(VENV_DIR), "bin", "python")
        venv_python_win_abs = os.path.abspath(venv_python_win)
        venv_python_unix_abs = os.path.abspath(venv_python_unix)

        # Prepare terminal-specific, copy-paste safe commands
        # PowerShell: use call operator (&) and quote path
        powershell_cmd = f'& "{venv_python_win_abs}" -m pip install {pkg_list}'
        # CMD: quoted executable path
        cmd_cmd = f'"{venv_python_win_abs}" -m pip install {pkg_list}'
        # Bash / Zsh: quoted unix path
        bash_cmd = f'"{venv_python_unix_abs}" -m pip install {pkg_list}'

        # Output user-friendly instructions (keep existing pipe-style formatting)
        print("\n| Incompatible package versions detected. |")

        print(
            "| Install using one of the following commands (run from project root / adjust path if needed): |\n")

        # ------------------------------------------------------------------------------------
        # OPTION 1 — Install via requirements.txt
        # ------------------------------------------------------------------------------------
        print("OPTION 1 — Install using requirements.txt (recommended)")
        print("--------------------------------------------------------")
        print("  Windows (PowerShell):")
        print(f"      .\\{VENV_DIR.name}\\Scripts\\Activate.ps1")
        print("      pip install -r requirements.txt\n")

        print("  Windows (CMD):")
        print(f"      .\\{VENV_DIR.name}\\Scripts\\activate.bat")
        print("      pip install -r requirements.txt\n")

        print("  Linux / macOS (Bash / Zsh):")
        print(f"      source ./{VENV_DIR.name}/bin/activate")
        print("      pip install -r requirements.txt\n")

        # ------------------------------------------------------------------------------------
        # OPTION 2 — Direct pip installation of only the required fixes
        # ------------------------------------------------------------------------------------
        print("OPTION 2 — Install only the required packages")
        print("--------------------------------------------------------")
        print("  Windows (CMD):")
        print(f"    {cmd_cmd}\n")

        print("  Windows (PowerShell):")
        print(f"    {powershell_cmd}\n")

        print("  Linux / macOS (Bash / Zsh):")
        print(f"    {bash_cmd}\n")

        # ------------------------------------------------------------------------------------
        # NOTES
        # ------------------------------------------------------------------------------------
        print("| Notes: |")
        print("  - Option 1 ensures a full, reproducible environment.")
        print("  - Option 2 installs only the missing/incompatible packages shown above.")
        print("  - In PowerShell you should use the call operator (&) before a quoted executable path.")
        print("  - CMD and Bash accept the quoted-path form shown above.")
        print("  - You may activate the virtualenv first, then run pip manually:")
        print(f"      source ./{VENV_DIR.name}/bin/activate  # (bash/zsh)")
        print(f"      .\\{VENV_DIR.name}\\Scripts\\Activate.ps1  # (PowerShell)\n")

        print("| Inspect commands before running. They will install into the project's .venv. |\n")

        # Pause briefly so user can copy the correct command
        import time
        try:
            time.sleep(8)
        except Exception:
            pass

        return sorted_pkgs

# -----------------------------
# Helper used by detection: last-chance venv installation detection
# -----------------------------
def _is_installed_in_venv(pkg_name: str, pip_token: Optional[str] = None) -> bool:
    # 1) distribution check
    try:
        if pip_token:
            if distribution_installed(pip_token) or find_distinfo_by_name(pip_token):
                return True
    except Exception:
        pass
    # 2) top-level mapping
    try:
        if top_level_package_provided(pkg_name):
            return True
    except Exception:
        pass
    # 3) importlib find_spec
    try:
        spec = importlib.util.find_spec(pkg_name)
        if spec is not None:
            return True
    except Exception:
        pass
    return False

# -----------------------------
# Example entrypoint
# -----------------------------
def init():
    # simple CLI usage: python pipdependancyinstaller.py
    runner = run()
    missing = runner.run(print_debug=print_debug, RUN_INSTALLER=True)
    if missing:
        print("\n| The installer detected the above missing packages. Run one of the printed commands to install them. |")
        import time
        time.sleep(100)
        exit()
    else:
        print("\n| No missing packages detected. You're good to go. |")
        try:  _add_global_path()
        except Exception as e:
            print(f"| DEBUG: _add_global_path failed: {e} |")

