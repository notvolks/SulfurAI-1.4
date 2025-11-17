import os, datetime, time
def who_imported_me():
    import inspect
    stack = inspect.stack()

    # skip first frame (this function itself)
    return [frame.filename for frame in stack[1:]]
def print_verti_list(items):  # Print items vertically
    for item in items:
        print(item)

def write_error(error, type):
    current_dir_e_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    folder_path_error_log = os.path.join(current_dir_e_log,'data', 'ErrorLogs')
    file_name_error_log = 'EasyLog.txt'
    file_path_error_log = os.path.join(folder_path_error_log, file_name_error_log)

    current_dir_db_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    folder_path_error_debug_log = os.path.join(current_dir_db_log,'data', 'ErrorLogs', 'logs')
    file_name_error_debug_log = 'error_debug.txt'
    file_path_error_debug_log = os.path.join(folder_path_error_debug_log, file_name_error_debug_log)

    current_dir_d_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    folder_path_error_list_d_log = os.path.join(current_dir_d_log,'data', 'ErrorLogs', 'logs')
    file_name_error_list_d_log = 'list_debug.txt'
    file_path_error_list_d_log = os.path.join(folder_path_error_list_d_log, file_name_error_list_d_log)

    if type == "debug_error":
        with open(file_path_error_list_d_log, "r") as file:
            lines_rd = file.readlines()
            if lines_rd:
                first_line_rd = lines_rd[0].strip()
                error_list_debug = first_line_rd.split(",")
            else:
                with open(file_path_error_list_d_log, "w") as file:
                    file.write(" ")

        with open(file_path_error_debug_log, "a") as file:
            error_message = f"{','.join(error_list_debug)}, {error}"
            file.write(f"{error_message}")

    if len(type) > 0:
        with open(file_path_error_log, "a") as file:
            error_msg_split = error_message.split(",")
            time_now = datetime.datetime.now()
            time_printed = time_now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"#########ERROR {time_printed}: {error_msg_split}\n")

def error(layout_type, error_type, error_message, error_code):  # Print errors
    if layout_type == "er1":
        extra = ""
        if error_type == "SCRIPT_TRACEBACK": extra = "This was an issue caused with the script-run. It will not be recorded, please send this full error log to the support page."
        text = [
            "[===========TRACEBACK===========]",
            f"[= ({error_type}) ~{error_message}~ =]",
            f"[= EXTRA: {extra} =]",
            "[===============================]",

        ]
        print_verti_list(text)
        print(f"Error code: {error_code}")
        print(f"Host scripts ID: {who_imported_me()}")
        write_error(f"Error Code ({error_code}) // No {error_type} ~{error_message} Found. **Solution - install new version** @host_file:{who_imported_me()}", "debug_error")
    elif layout_type == "er2":
        text = [
            f"{error_type} Failed...",
            f"#{error_message}#"
        ]
        print_verti_list(text)
        print(f"Error code: {error_code}")
        print(f"Host scripts ID: {who_imported_me()}")
        write_error(f"Error Code ({error_code}) // No {error_type} Failed. ~{error_message}. **Solution - install new version** @host_file:{who_imported_me()}", "debug_error")




def instant_shutdown(reason):
    try:
        exit()
    except (ImportError, SystemExit) as y:
        print(f"InstantShutDown-Initiated: {reason} Testing exit: Value:Exception// exit(sys):{y}")
        time.sleep(5)
        quit()


def brick_out(time2):
    time.sleep(time2)
    instant_shutdown("Timed out.")


# scripts/ai_renderer_sentences/error.py
import traceback
import linecache
import time
import uuid
import inspect
from typing import Optional, List

# Small ANSI helpers (optional)
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_CYAN = "\033[36m"


def who_imported_me(full_stack: bool = False, context: int = 3) -> str:
    """
    Return caller information. By default returns a short "who imported me" string:
      "<file>:<lineno> in <function>"
    If full_stack=True returns a formatted stack (like format_list) excluding this function.
    """
    if full_stack:
        # return formatted stack (exclude this frame)
        stack = traceback.extract_stack()[:-1]
        return "".join(traceback.format_list(stack))
    # default: return first meaningful caller outside this module
    stack = inspect.stack()
    # stack[0] is who_imported_me, stack[1] is the immediate caller; we want the first caller outside this module
    for frame_info in stack[1:]:
        module = inspect.getmodule(frame_info.frame)
        module_name = getattr(module, "__name__", None)
        # skip frames that come from this module file
        if module_name and module_name.startswith(__name__):
            continue
        filename = frame_info.filename
        lineno = frame_info.lineno
        func = frame_info.function
        return f"{filename}:{lineno} in {func}"
    # fallback: immediate caller
    if len(stack) > 1:
        fi = stack[1]
        return f"{fi.filename}:{fi.lineno} in {fi.function}"
    return "<unknown>"


def _format_frame_snippet(filename: str, lineno: int, context: int = 3, ansi: bool = False) -> str:
    """Return a snippet of source around lineno with the target line highlighted."""
    start = max(1, lineno - context)
    end = lineno + context
    lines: List[str] = []
    for n in range(start, end + 1):
        txt = linecache.getline(filename, n) or ""
        # rstrip to avoid double trailing whitespace in printed output
        if n == lineno:
            # highlight this line
            if ansi:
                line = f"{ANSI_RED}{ANSI_BOLD}-> {n:4d}: {txt.rstrip()}{ANSI_RESET}"
            else:
                line = f"-> {n:4d}: {txt.rstrip()}"
        else:
            line = f"   {n:4d}: {txt.rstrip()}"
        lines.append(line)
    return "\n".join(lines)


class SulfurError(Exception):
    """Error that formats a GitHub-style traceback with code snippets and includes who_imported_me (wim)."""

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        details: Optional[dict] = None,
        capture_stack: bool = True,
        include_wim: bool = True,
        wim_full_stack: bool = False,
    ):
        """
        :param message: human message
        :param code: stable machine code (e.g. "arch.start_failed")
        :param details: structured details for programmatic use
        :param capture_stack: whether to capture the stack now
        :param include_wim: whether to capture who_imported_me() and include in details
        :param wim_full_stack: if True `who_imported_me(full_stack=True)` is used
        """
        super().__init__(message)
        self.message = message or ""
        self.code = code or "internal.error"
        self.details = details.copy() if details else {}
        self.timestamp = time.time()
        self.request_id = str(uuid.uuid4())
        self._stack = None
        # capture stack summary now (exclude this __init__ frame)
        if capture_stack:
            self._stack = traceback.extract_stack()[:-1]

        # attach wim info if requested
        if include_wim:
            try:
                wim = who_imported_me(full_stack=wim_full_stack)
            except Exception:
                wim = "<wim-failure>"
            # expose both a short field and full text in details
            self.details.setdefault("wim", wim)
            # also include it in code if code was not set explicitly
            if code is None:
                # keep it compact for code field
                short = wim.splitlines()[0] if isinstance(wim, str) else str(wim)
                self.code = f"debug.imported_by:{short}"

    def format_github_traceback(self, *, context: int = 3, ansi: bool = False, markdown: bool = False) -> str:
        """
        Return a string formatted similarly to GitHub tracebacks with snippets.
        If markdown=True, returned text is wrapped in a fenced code block for GitHub.
        """
        frames = self._stack or []
        parts: List[str] = []
        parts.append("Traceback (most recent call last):")
        for fr in frames:
            filename = fr.filename
            lineno = fr.lineno
            func = fr.name
            parts.append(f'  File "{filename}", line {lineno}, in {func}')
            snippet = _format_frame_snippet(filename, lineno, context=context, ansi=ansi)
            parts.append(snippet)
        # finally include the exception line
        parts.append(f"{self.__class__.__name__}: {self.message} (code={self.code})")
        parts.append(f"[request_id: {self.request_id}]")
        # include wim summary (short) and full wim in details if present
        wim = self.details.get("wim")
        if wim:
            parts.append("")
            parts.append("Who imported me (wim):")
            # if multi-line (full stack), show a compact first line then full block
            if "\n" in wim:
                first = wim.splitlines()[0]
                parts.append(f"  {first}  (full stack below)")
                parts.append("")
                parts.append(wim)  # already formatted stack-like if full_stack used
            else:
                parts.append(f"  {wim}")

        body = "\n".join(parts)
        if markdown:
            return "```python\n" + body + "\n```"
        return body

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message} (code={self.code})"


def wrap_with_sulfur(msg: str, *, code: Optional[str] = None, details: Optional[dict] = None, debug_stack: bool = True) -> "SulfurError":
    """
    Create a SulfurError capturing the current stack and original exception (if any).
    Usage inside an except: `raise wrap_with_sulfur("msg") from e`
    """
    import sys

    err = SulfurError(msg, code=code, details=details, capture_stack=debug_stack, include_wim=True)
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_value is not None:
        err.__cause__ = exc_value
    return err

