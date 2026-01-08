
from scripts.ai_renderer_sentences.error import SulfurError
raise SulfurError(message=f"FILE_WAS_DEPRECATED_TRAIN_MANUALLY!")




######################## deprecated!

import os, subprocess, sys
import importlib, random, time
import importlib.metadata
from contextlib import contextmanager
import contextlib

########## Debug Items
TOS = [
    "--------------------------------------------------------------------------------------------------",
    "By using this application you agree to the Terms of Service listed in the project files.",
    "If you do not consent, stop using our services.",
    "If you cannot find it, install a new version OR look in the root folder for 'Terms of Service.txt' .",
    "--------------------------------------------------------------------------------------------------",
]

def print_verti_list(list_):
    for item in list_:
        print(item)

print_verti_list(TOS)


@contextmanager
def suppress_output():
    # Open OS null device for writing
    with open(os.devnull, 'w') as devnull:
        # Save original stdout and stderr file descriptors
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



def get_call_file_path():
    import call_file_path
    return call_file_path.Call()

call = get_call_file_path()

def install(packages):
    if isinstance(packages, str):
        packages = [packages]

    for pkg in packages:
        try:
            if pkg == "pygame-ce":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame-ce", "--upgrade"])
                print("pygame-ce installed successfully!")
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"{pkg} installed successfully!")
        except (PermissionError, subprocess.CalledProcessError) as error:
            print(f"An error occurred while installing {pkg}: {error}")

def get_installed_packages():
    package_map = {}
    for dist in importlib.metadata.distributions():
        try:
            package_name = dist.metadata["Name"]
            top_level = dist.read_text("top_level.txt")
            if top_level:
                for module in top_level.splitlines():
                    package_map[module] = package_name
        except KeyError:
            continue
    return package_map

print("-------Preparing PIP libraries.")

def safe_import(module_name, package_name=None, extra_packages=None):
    MODULE_TO_PACKAGE_MAP = {
        "sklearn": "scikit-learn",
        "pygame": "pygame-ce",
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
        "beautifulsoup4": "bs4",
    }
    automatic_restart_failsafe = 0
    file_path_limit = call.settings_pip_fallback_amount()
    with open(file_path_limit, "r", encoding="utf-8", errors="ignore") as file:
        automatic_restart_limit = int(file.readline().strip() or 1)

    pkg = package_name or module_name
    while automatic_restart_failsafe < automatic_restart_limit:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            print(f"------- {pkg} not found. Installing...")
            install([pkg] + (extra_packages or []))
            try:
                target = MODULE_TO_PACKAGE_MAP.get(module_name, module_name)
                return importlib.import_module(target)
            except ImportError:
                automatic_restart_failsafe += 1
                print(f"Error while importing {module_name} after installation. "
                      f"Attempt {automatic_restart_failsafe}/{automatic_restart_limit}. "
                      f"Restart Sulfur if this persists.")
                if automatic_restart_failsafe >= automatic_restart_limit:
                    print(f"Failed to import {module_name} after multiple attempts. This could be a fake error - check previous print statements to ensure.")
                    return None

modules = [
    ("sklearn", "scikit-learn"),
    ("pygame", "pygame-ce"),
    ("pygame_gui",),
    ("language_tool_python",),
    ("langdetect",),
    ("tqdm",),
    ("numpy",),
    ("nltk",),
    ("pandas",),
    ("transformers",),
    ("torch", None, ["torchvision", "torchaudio"]),
    ("faker",),
    ("tensorflow",),
]

for mod in modules:
    safe_import(*mod)

print("-------All custom libraries are installed. ")
from faker import Faker
import nltk
from nltk.corpus import names, wordnet as wn

try:
    nltk.data.find('corpora/names')
except LookupError:
    nltk.download('names', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

########################## Logic Items

fake = Faker()

def get_random_faker_language():
    supported_langs = [
        'en_US', 'en_GB', 'de_DE', 'fr_FR', 'it_IT', 'es_ES', 'pt_BR', 'nl_NL', 'ru_RU', 'ja_JP', 'zh_CN'
    ]
    while supported_langs:
        lang = random.choice(supported_langs)
        try:
            Faker(lang)
            return lang
        except:
            supported_langs.remove(lang)
    return random.choice(['en_GB', 'en_US'])

natural_toggle = True  # Add this line near the top of your script (global scope)

def generate_sentence(style, language=None):
    global language_code, natural_toggle
    language_code = "en_GB"

    if style in ("premiumlang", "pl"):
        language = language or get_random_faker_language()
        language_code = language
        if extra_debug_autotrainer == "yes": print(f"Selected language for Faker: {language}")
        try:
            fake_local = Faker(language)
            return fake_local.sentence(), language_code
        except Exception:
            print(f"[!] Faker does not support '{language}', falling back to English.")
            choice = random.choice(['en_GB', 'en_US'])
            return Faker(choice).sentence(), choice

    subject = random.choice(names.words())
    verb = ""
    fake_local = fake

    def get_compatible_verbs(noun):
        noun_synsets = wn.synsets(noun, pos=wn.NOUN)
        compatible_verbs = []
        for synset in noun_synsets:
            for lemma in synset.lemmas():
                for d in lemma.derivationally_related_forms():
                    if d.synset().pos() == 'v':
                        compatible_verbs.append(d.name())
        return list(set(compatible_verbs))

    def add_complementary_elements(verb, obj):
        adjectives = [adj.name().split('.')[0] for adj in wn.all_synsets('a') if len(adj.name().split('.')[0]) <= 8]
        adverbs = [adv.name().split('.')[0] for adv in wn.all_synsets('r') if len(adv.name().split('.')[0]) <= 8]
        obj_adj = random.choice(adjectives) if random.random() > 0.5 else ""
        verb_adv = random.choice(adverbs) if random.random() > 0.5 else ""

        if random.random() > 0.7:
            verb = f"was {verb} by"
        if obj_adj:
            obj = f"{obj_adj} {obj}"
        return verb, obj

    objects = [n.name().split('.')[0] for n in wn.all_synsets('n') if len(n.name().split('.')[0]) <= 8]
    obj = random.choice(objects)

    if style in ("natural", "n"):
        # Alternate between en_GB and en_US
        language_code = 'en_GB' if natural_toggle else 'en_US'
        natural_toggle = not natural_toggle
        fake_local = Faker(language_code)

        verbs = [v.name().split('.')[0] for v in wn.all_synsets('v') if len(v.name().split('.')[0]) <= 8]
        verb = random.choice(verbs)

    elif style in ("legacy", "l"):
        return fake.sentence(), random.choice(['en_GB', 'en_US'])

    else:
        verbs = get_compatible_verbs(obj)
        verb = random.choice(verbs) if verbs else "did"
        verb, obj = add_complementary_elements(verb, obj)

    prefix_options = [
        "",
        "Yesterday, ",
        f"On {fake_local.day_of_week()}, ",
        f"This morning at {fake_local.time(pattern='%H:%M')}, ",
    ]
    prefix = random.choice(prefix_options)

    sentence = f"{prefix}{subject} {verb} the {obj}."
    return sentence.capitalize(), language_code

######################## Menu Items

id_run_train = 0
file_path_autotrainer_debug = call.settings_auto_trainer_extra_debug()

with open(file_path_autotrainer_debug, "r", encoding="utf-8", errors="ignore") as file:
    extra_debug_autotrainer = ",".join(file.readlines()).strip().lower()

print("Presenting Menu.....")
print("--------------------Select the trainer methods.--------------------")
print("######SentenceGenerationType######")
print("- Legacy: Provides basic sentences that are quick and easy to process that only supports UK english. Not recommended for large datasets.")
print("- Natural: Provides more complex, coherent sentences that supports US/UK english abbreviation. Good for large datasets. (BETA)")
print("- PremiumLang: Provides ultra coherent sentences that support *multiple* languages. Good for large context models/features but is extremely slow. (BETA)")

while True:
    legacy_options = input("Select the sentence generation type:").strip().lower()
    options = ["natural", "n", "legacy", "l", "premiumlang", "pl"]
    if legacy_options in options:
        break
    else:
        print("Must select 'Legacy' , 'Natural' or 'PremiumLang'.")

while True:
    language = ""
    id_run_train += 1
    print(f"######################---------Running Trainer on loop **{id_run_train}**")

    if extra_debug_autotrainer == "yes":
        print(f"###################--------- SulfurAI is starting...")

    if extra_debug_autotrainer == "yes":
        print(f"################--------- SulfurAI is generating a sentence...")

    match legacy_options:
        case "legacy" | "l":
            sentence, language = fake.sentence(), random.choice(['en_GB', 'en_US'])
        case "natural" | "n":
            sentence, language = generate_sentence("natural")
        case "premiumlang" | "pl":
            sentence, language = generate_sentence("premiumlang")
        case _:
            sentence = fake.sentence()

    if extra_debug_autotrainer == "yes":
        print(f"################--------- SulfurAI generated a sentence...")

    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','..'))
    folder_path_input = os.path.join(current_dir, 'DATA')
    file_name_input = 'Input.txt'
    file_path_input = os.path.join(folder_path_input, file_name_input)

    with open(file_path_input, "w", encoding="utf-8", errors="ignore") as file:
        file.write(str(sentence))

    try:
        if extra_debug_autotrainer == "yes":
            print(f"###################--------- SulfurAI is processing...")

        stdout = None if extra_debug_autotrainer == "yes" else subprocess.DEVNULL
        stderr = None if extra_debug_autotrainer == "yes" else subprocess.DEVNULL

        try:
            #with suppress_output():
                sys.path.insert(0, current_dir)
                SulfurAI = importlib.import_module("SulfurAI")
                sys.path.pop(0)
                SulfurAI.run_via_trainer(language)
        except subprocess.CalledProcessError as e:
            print(f"Error running SulfurAI.py: {e}")
        except Exception as e:
            print(f"Unexpected error in SulfurAI run: {e}")

    except Exception as e:
        print(e)