###################TO ADD NEW SETTINGS:
#-FIRST, CREATE A NEW IMAGE FOR THE BUTTON AND THE SCREEN.
#-THEN, ADD IT TO THE PATHS , UI_SPECS + UI_KEYS + DEFAULT_MAP + [IF NM == ___ + IF "MGRS" IN LOCALS:___  + IF KEY == ___ + ELIF MGR ==: ___] ->> highlighted in red (if needed to be input) AND SCREEN_CFG WITH AN INPUT BUTTON IF NEEDED (OFF/ON)
#-THEN, CREATE A BUTTON SPEC IN BTN_SPECS
#-THEN, CREATE A FILE PATH IN DATA/SETTINGS, ADD IT TO CALL_FILE_PATH.PY AND ADD IT TO THE PATHS AFOREMENTIONED.



import os
import sys
import subprocess
from importlib.metadata import PackageNotFoundError
from extra_models.Sulfur.TrainingScript.Build import call_file_path
import pygame
from extra_models.Sulfur.GeneralScripts.LocalUserEvents import events_hoster
events_hoster.write_event("event_OpenedSettings")

# ── install helper ──
def install(pkg):
    cmd = [sys.executable, "-m", "pip", "install", pkg]
    if pkg == "pygame-ce":
        cmd.append("--upgrade")
    try:
        subprocess.check_call(cmd)
        print(f"{pkg} installed successfully!")
    except Exception as e:
        print(f"Failed to install {pkg}: {e}")

# ── ensure pygame & pygame_gui ──
try:
    import pygame_gui
except ImportError:
    install("pygame_gui")
    import pygame_gui

try:
    pygame.init()
except Exception:
    install("pygame-ce")
    import pygame
    pygame.init()

# ── renderer call ──
call = call_file_path.Call()

# ── Settings paths & defaults ──
paths = {
    "extra":      (call.settings_extra_debug(),     "no"),
    "backup":     (call.settings_backup(),          "yes"),
    "input":      (call.input_limit(),              "50"),
    "input_bypass": (call.settings_input_process_limit(), "yes"),
    "days_ago":   (call.settings_ui_days_ago(),     "5"),
    "days_apart": (call.settings_ui_days_apart(),   "5"),
    "weeks_ago":   (call.settings_ui_weeks_ago(),   "5"),
    "weeks_apart": (call.settings_ui_weeks_apart(), "5"),
    "months_ago":   (call.settings_ui_months_ago(),   "5"),
    "months_apart": (call.settings_ui_months_apart(), "5"),
    "years_ago":   (call.settings_ui_years_ago(),   "5"),
    "years_apart": (call.settings_ui_years_apart(), "1"),
    "autotrainer": (call.settings_auto_trainer_extra_debug(),     "yes"),
    "auto_trainer_delay": (call.settings_auto_trainer_delay(), "0"),
    "python_pip": (call.settings_pip_fallback_amount(),     "3"),
    "extra_output": (call.settings_ui_write_to_seperate_output(),     "yes"),
    "training_data": (call.settings_save_training_data(),     "yes"),
    "auto_render_dp": (call.settings_auto_render_dp(),     "yes"),
    "debug_dp": (call.settings_debug_dp(),     "yes"),
}

def ensure(fp, default):
    if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
        with open(fp, "w", encoding="utf-8", errors="ignore") as f:
            f.write(default)

for key, (fp, df) in paths.items():
    ensure(fp, df)

# ── Print TOS ──
def pv(lines):
    for line in lines:
        print(line)
pv([
    "By using this application you agree to the Terms of Service listed in the project files.",
    "If you cannot find it, install a new version."
])
print("-------SulfurAI Settings requires the extension PYGAME COMMUNITY EDITION.-------")

# ── Read settings ──
def read(fp, df):
    try:
        return open(fp, "r", encoding="utf-8", errors="ignore").read().strip()
    except:
        return df

values = {k: read(fp, df) for k, (fp, df) in paths.items()}

# ── Pygame + UI init ──
SCREEN = (1080, 720)
screen = pygame.display.set_mode(SCREEN)
clock = pygame.time.Clock()

# Separate manager for input‐limit screen
manager_input = pygame_gui.UIManager(SCREEN)
manager_pip_python = pygame_gui.UIManager(SCREEN)
manager_ui_extra_delay = pygame_gui.UIManager(SCREEN)
# Managers for the eight UI fields (days/weeks/months/years)
ui_keys = [
    "days_ago", "days_apart",
    "weeks_ago", "weeks_apart",
    "months_ago", "months_apart",
    "years_ago", "years_apart",
    "python_pip", "auto_trainer_delay",

]
mgrs = {k: pygame_gui.UIManager(SCREEN) for k in ui_keys}

# ── Create text inputs ──
ui_specs = {
    "input":      ((500, 350), (100, 50)),
    "days_ago":   ((400, 175), (100, 50)),
    "days_apart": ((825, 175), (100, 50)),
    "weeks_ago":   ((400, 260), (100, 50)),
    "weeks_apart": ((825, 260), (100, 50)),
    "months_ago":   ((400, 350), (100, 50)),
    "months_apart": ((850, 350), (100, 50)),
    "years_ago":   ((400, 425), (100, 50)),
    "years_apart": ((825, 425), (100, 50)),
    "python_pip": ((370, 130), (100, 50)),
    "auto_trainer_delay": ((530, 270), (100, 50)),
}

texts = {}
for key, (pos, size) in ui_specs.items():
    if key == "input":
        mgr = manager_input
    elif key == "python_pip":
        mgr = manager_pip_python
    elif key == "auto_trainer_delay":
        mgr = manager_ui_extra_delay
    else:
        mgr = mgrs[key]
    te = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(pos, size),
        manager=mgr,
        object_id="#main_text_entry"
    )
    te.set_text(values[key])
    texts[key] = te

# ── Button class + loader ──
class Button:
    def __init__(self, img, pos, scale):
        surf = pygame.image.load(img).convert_alpha()
        w, h = surf.get_size()
        self.image = pygame.transform.smoothscale(surf, (int(w * scale), int(h * scale)))
        self.rect = self.image.get_rect(topleft=pos)
        self.pressed = False

    def draw(self, window):
        window.blit(self.image, self.rect)

    def touched(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def is_pressed(self):
        down = pygame.mouse.get_pressed()[0]
        if self.rect.collidepoint(pygame.mouse.get_pos()) and down and not self.pressed:
            self.pressed = True
            return True
        if not down:
            self.pressed = False
        return False

    def set_alpha(self, a):
        self.image.set_alpha(a)

btn_specs = {
    "debug":  ("settings/images/ex_dbg_button.jpeg",   (250, 200), 0.15),
    "backup": ("settings/images/ex_backup_button.jpeg", (500, 200), 0.15),
    "input":  ("settings/images/ex_input_button.jpeg",  (750, 200), 0.15),
    "user":   ("settings/images/ex_user_insight.jpeg",  (250, 350), 0.15),
    "autotrainer":   ("settings/images/ex_AutoTrainer.jpeg",  (500, 350), 0.15),
    "save_changes":   ("settings/images/ex_save_changes.jpeg",  (850, 590), 0.15),
    "python_pip":   ("settings/images/ex_python_pip_button.jpeg",  (750, 350), 0.15),
    "training_data":   ("settings/images/ex_TrainingData_button.jpeg",  (250, 500), 0.15),
    "auto_render_dp":   ("settings/images/ex_advancedllmsettings_button.jpeg",  (500, 500), 0.15),



}

buttons = {k: Button(*v) for k, v in btn_specs.items()}


# ── Screens config ──
screen_cfg = {
    "debug": {
        "path_key":     ["extra", "extra_output","debug_dp"],
        "screen_btn":   "settings/images/ex_dbg_screen.jpeg",
        "off_btn":      "settings/images/ex_dbg_off.jpeg",
        "on_btn":       "settings/images/ex_dbg_on.jpeg",
        "x_btn":        "settings/images/ex_dbg_x.jpeg",
        "pos_off":      [(425,150), (550,180), (530,300)],
        "scale_main":   0.6,
        "scale_toggle": 0.15,
        "pos_x":        (900, 0),
    },
    "backup": {
        "path_key":    "backup",
        "screen_btn":  "settings/images/ex_dbg_screen_backup.jpeg",
        "off_btn":     "settings/images/ex_dbg_off.jpeg",
        "on_btn":      "settings/images/ex_dbg_on.jpeg",
        "x_btn":       "settings/images/ex_dbg_x.jpeg",
        "pos_off":     (425, 200),
    },
    "input": {
        "path_key": "input",
        "path_key": "input_bypass",
        "screen_btn": "settings/images/ex_input_screen.jpeg",
        "off_btn": "settings/images/ex_dbg_off.jpeg",
        "on_btn": "settings/images/ex_dbg_on.jpeg",
        "x_btn": "settings/images/ex_dbg_x.jpeg",
        "pos_off": (620, 190),
    },
    "user": {
        "path_key":    None,
        "screen_btn":  "settings/images/ex_user_insight_screen.jpeg",
        "off_btn":     "settings/images/ex_dbg_off.jpeg",
        "on_btn":      "settings/images/ex_dbg_on.jpeg",
        "x_btn":       "settings/images/ex_dbg_x.jpeg",
        "pos_off":     (620, 190),
    },

    "autotrainer": {
        "path_key":     "autotrainer",
        "screen_btn":   "settings/images/ex_AutoTrainer_screen.jpeg",
        "off_btn":      "settings/images/ex_dbg_off.jpeg",
        "on_btn":       "settings/images/ex_dbg_on.jpeg",
        "x_btn":        "settings/images/ex_dbg_x.jpeg",
        "pos_off":      (600, 170),
        "scale_main":   0.6,
        "scale_toggle": 0.15,
        "pos_x":        (900, 0),
    },

    "python_pip": {
        "path_key": "python_pip",
        "screen_btn": "settings/images/ex_python_pip_screen.jpeg",
        "off_btn": "settings/images/ex_dbg_off.jpeg",
        "on_btn": "settings/images/ex_dbg_on.jpeg",
        "x_btn": "settings/images/ex_dbg_x.jpeg",
        "pos_off": (9000, 9000), ##essentially hides the button because its not needed
        "scale_main": 0.6,
        "scale_toggle": 0.15,
        "pos_x": (900, 0),
    },

    "training_data": {
        "path_key": "training_data",
        "screen_btn": "settings/images/ex_TrainingData_screen.jpeg",
        "off_btn": "settings/images/ex_dbg_off.jpeg",
        "on_btn": "settings/images/ex_dbg_on.jpeg",
        "x_btn": "settings/images/ex_dbg_x.jpeg",
        "pos_off": (400, 200),
        "scale_main": 0.6,
        "scale_toggle": 0.15,
        "pos_x": (900, 0),
    },

    "auto_render_dp": {
        "path_key": "auto_render_dp",
        "screen_btn": "settings/images/ex_advancedllmsettings_screen.jpeg",
        "off_btn": "settings/images/ex_dbg_off.jpeg",
        "on_btn": "settings/images/ex_dbg_on.jpeg",
        "x_btn": "settings/images/ex_dbg_x.jpeg",
        "pos_off": (800, 150),
        "scale_main": 0.6,
        "scale_toggle": 0.15,
        "pos_x": (900, 0),
    },


}

for name, cfg in screen_cfg.items():
    toggle_scale = cfg.get("scale_toggle", 0.15)
    pos_x        = cfg.get("pos_x", (900, 0))

    # build the “screen” and “x” buttons as before
    screen_btn = Button(cfg["screen_btn"], cfg.get("pos_main", (55,50)), cfg.get("scale_main", .6))
    x_btn      = Button(cfg["x_btn"],       pos_x,                               0.25)

    # now handle off/on toggles potentially as lists:
    offs = cfg["pos_off"]
    if isinstance(cfg["path_key"], list):
        # we have N settings to toggle → build N off/on buttons
        off_buttons = []
        on_buttons  = []
        for pos in offs:
            off_buttons.append(Button(cfg["off_btn"], pos, toggle_scale))
            on_buttons .append(Button(cfg["on_btn"],  pos, toggle_scale))
    else:
        off_buttons = Button(cfg["off_btn"], offs, toggle_scale)
        on_buttons  = Button(cfg["on_btn"],  offs, toggle_scale)

    cfg.update({
        "screen": screen_btn,
        "x":      x_btn,
        "off":    off_buttons,
        "on":     on_buttons,
    })

show = {name: False for name in screen_cfg}

backdrop = pygame.transform.smoothscale(
    pygame.image.load("settings/images/backdrop.jpeg"),
    (SCREEN[0] + 42, SCREEN[1] + 25)
)
mouse_img = pygame.image.load("settings/images/mousehitbox.jpeg")
mouse_img.set_alpha(0)

# ── Main loop ──
while True:
    screen.blit(backdrop, (-25, -10))
    mx, my = pygame.mouse.get_pos()

    # main buttons
    modal_active = any(show.values())
    active_screens = {name: False for name in screen_cfg}
    for nm, btn in buttons.items():
        btn.draw(screen)
        btn.set_alpha(200 if btn.touched() else 255)
        if btn.is_pressed() and not any(active_screens.values()) and not modal_active:
            show[nm] = True
            if nm == "save_changes":
                print("Saved changes.")
                show[nm] = False
            if nm == "python_pip":
                show["python_pip"] = True
            if nm == "auto_trainer_delay":
                show["autotrainer"] = True




    # events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # route text‐entry events
        if (
            event.type == pygame.USEREVENT
            and event.user_type == pygame_gui.UI_TEXT_ENTRY_CHANGED
            and event.ui_object_id == "#main_text_entry"
        ):
            mgr = event.ui_element.ui_manager
            # identify which field
            key = None
            if mgr is manager_input:
                key = "input"
            elif mgr is manager_pip_python:
                key = "python_pip"
            elif mgr is manager_ui_extra_delay:
                key = "auto_trainer_delay"
            else:
                for k, m in mgrs.items():
                    if mgr is m:
                        key = k
                        break

            # parse or default
            try:
                val = int(event.ui_element.get_text())
            except ValueError:
                default_map = {
                    "input":       50,
                    "days_ago":     5,
                    "days_apart":   5,
                    "weeks_ago":    5,
                    "weeks_apart":  5,
                    "months_ago":   5,
                    "months_apart": 5,
                    "years_ago":    5,
                    "years_apart":  1,
                    "python_pip": 3,
                    "auto_trainer_delay": 0,
                }
                val = default_map.get(key, 5)
                print(f"Your input is not an integer. Auto set to {val}.")
            print(f"Input changed to {val}")

            # savef
            if key is not None:
                fp, _ = paths[key]

                # Validation logic
                if key == "python_pip" and val < 3:
                    print("Value for python_pip cannot be less than 3. Setting to 3.")
                    val = 3
                elif key == "auto_trainer_delay" and val < 0:
                    print("Value for auto_trainer_delay cannot be less than 0. Setting to 0.")
                    val = 0
                elif val < 1:
                    print("Value cannot be less than 1. Setting to 1.")  # Generic fallback, customize if needed
                    val = 1

                # Write to file
                with open(fp, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(str(val))

        # pygame_gui needs its managers to process all events
        if "mgrs" in locals():
            if event.type != pygame.USEREVENT:
                for m in mgrs.values():
                    m.process_events(event)
                manager_input.process_events(event)
                manager_pip_python.process_events(event)
                manager_ui_extra_delay.process_events(event)


    # draw active screens
    for nm, cfg in screen_cfg.items():
        if not show[nm]:
            continue

        cfg["screen"].draw(screen)
        cfg["x"].draw(screen)

        # if we have a list of settings to toggle:
        if isinstance(cfg["path_key"], list):
            for idx, key in enumerate(cfg["path_key"]):
                fp, _ = paths[key]
                data = open(fp, "r").read().strip()
                # pick the correct off/on widget
                toggle = cfg["off"][idx] if data == "no" else cfg["on"][idx]
                toggle.set_alpha(200 if toggle.touched() else 255)
                toggle.draw(screen)

                if toggle.is_pressed():
                    with open(fp, "w") as f:
                        f.write("yes" if data == "no" else "no")

        # otherwise your old single‐toggle logic:
        elif cfg["path_key"] is not None:
            fp, _ = paths[cfg["path_key"]]
            data = open(fp, "r").read().strip()
            toggle = cfg["off"] if data == "no" else cfg["on"]
            toggle.set_alpha(200 if toggle.touched() else 255)
            toggle.draw(screen)
            if toggle.is_pressed():
                with open(fp, "w") as f:
                    f.write("yes" if data == "no" else "no")

        if nm == "input":

            manager_input.draw_ui(screen)
            manager_input.update(0)
        elif nm == "user":
            for m in mgrs.values():
                m.draw_ui(screen)
                m.update(0)

        elif nm == "python_pip":

            manager_pip_python.draw_ui(screen)
            manager_pip_python.update(0)


        elif nm == "autotrainer":

            fp, _ = paths["autotrainer"]
            manager_ui_extra_delay.draw_ui(screen)
            manager_ui_extra_delay.update(0)

        if cfg['x'].is_pressed():
            show[nm] = False


    screen.blit(mouse_img, (mx, my))
    pygame.display.update()
    clock.tick(120)
