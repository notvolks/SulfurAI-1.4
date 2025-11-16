import os


def _get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()
call = _get_call_file_path()


def parse_config_file(file_path):
    config = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '====' not in line:
                continue  # skip empty or invalid lines

            parts = line.split('====')
            key = parts[0].strip()
            values = parts[1:]

            config[key] = values[0] if len(values) == 1 else [v.strip() for v in values]

    return config


def parse_all_configs_in_directory(directory_path):
    config_data = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            config = parse_config_file(file_path)
            config_data[filename] = config

    return config_data

def extract_model_names_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("name===="):
                parts = line.strip().split("====")
                if len(parts) > 1:
                    return parts[1].strip()
    return None


def list_all_models_in_directory(directory_path):
    model_names = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            model_name = extract_model_names_from_file(file_path)
            if model_name:
                model_names.append(model_name)

    return model_names

def get_active_model_from_file():
    active_model = call.active_model()
    with open(active_model, "r", encoding="utf-8", errors="ignore") as file:
        model_name = file.read().strip()
        if model_name == "": return None
        value, timestamp = model_name.strip().split("@", 1)
        return value,timestamp

def add_active_model(str):
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    active_model = call.active_model()
    cache_LocalActiveModelHistory = call.cache_LocalActiveModelHistory()
    with open(active_model, "w", encoding="utf-8", errors="ignore") as file:
        file.write(str + "@" + timestamp)
    with open(cache_LocalActiveModelHistory, "a", encoding="utf-8", errors="ignore") as file:
        file.write(f"{str}@{timestamp}\n")


def get_active_model_history():
    cache_file = call.cache_LocalActiveModelHistory()
    history = []

    if not os.path.exists(cache_file):
        return history  # empty list if file doesn't exist

    with open(cache_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or "@" not in line:
                continue  # skip empty or invalid lines

            model, timestamp = line.split("@", 1)
            history.append({"model": model, "timestamp": timestamp})

    return history
