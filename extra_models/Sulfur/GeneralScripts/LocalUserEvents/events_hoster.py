### needs to:
# write to the cache (if same event, clear cache)
# read from the cache


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

def write_event(event_text):
    """
    Writes event_text to file_path.
    If event_text already exists in the file, the file is cleared first.
    """
    file_path = call.cache_LocalEventsHost()
    try:

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    if event_text in content and not event_text == "event_RanSulfurViaMain":
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")
    else:

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(event_text + "\n")


def read_events():
    """
    Reads a file and returns a list of lines.
    Returns None if the file is empty or doesn't exist.
    """
    try:
        file_path = call.cache_LocalEventsHost()
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            return lines if lines else None
    except Exception:
        return None

# python
def calculate_events_average():
    # Calculates the percentage of non-sulfur events relative to the total user events.
    events = read_events()
    total_pre = len(events) if events else 0
    count_sulfur = ["event_RanSulfurViaMain", "event_RanSulfurViaAPI"]
    total_sulfur = sum(events.count(thing) for thing in count_sulfur)
    try:
        non_sulfur = total_pre - total_sulfur
        percentage = (non_sulfur / total_pre) * 100 if total_pre > 0 else 0
    except ZeroDivisionError:
        percentage = 0
    return percentage


