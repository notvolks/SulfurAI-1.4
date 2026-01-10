################ Welcome to SulfurAI-APP!
### Functions here should be modified with proper intent.
### This python script was written in the Holly format. To find out how it works go into setup/HollyFormat/ReadMe.txt
### This python script is designed to host all SulfurAI API functions for python and run via the __main__ tag.


# ---------------Naviging the full app structure:

# apps/:SulfurAppAssets/:
#>> cache /:caches for tabs and other temp data
#>> scripts / verifictation /: oAUTH and verification scripts
#>> scripts / essential / ui / pages /: UI page classes
### LAYOUT:


# ---------------GOING DOWN!
##### -Importing base level items.
##### -Setting up the set-up page and oAUTH scripts (app backend).
##### -Setting up the UI and oAUTH scripts (frontend-ui).


# ---------------SECTIONS:
# 1) SET-UP
# 2) SCRIPTS
# 3) UI
# 4) MAIN



# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       S E T - U P                                                                                      |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


#please keep imports in length order
import SulfurAI
import os
import json
import threading
import datetime
import pathlib
import time
import base64
import flet as ft
import requests
import secrets
import shutil
import keyring
from pathlib import Path
from apps.SulfurAppAssets.globalvar import global_var
from apps.SulfurAppAssets.scripts.verification import sulfuroauth

CACHE_BASE = Path(os.path.dirname(__file__)) / "apps" / "SulfurAppAssets" / "cache" / "tabs"
CACHE_BASE.mkdir(parents=True, exist_ok=True)

BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()


# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       ESSENTIAL SCRIPTS                                                                                      |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


from apps.SulfurAppAssets.scripts.essential.ui import tabs_script
TABS_INDEX_PATH = CACHE_BASE / "tabs_index.json"


# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       UI                                                                                      |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


class PageBase:
    def __init__(self, title: str):
        self.title = title

    def content(self):
        raise NotImplementedError


# ----------------------------------------------------------------------Tabs UI------------------------------------------------------------------------------
from apps.SulfurAppAssets.scripts.essential.ui.pages.SetupPage import SetupPage
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPage import ChannelPage
from apps.SulfurAppAssets.scripts.essential.ui.pages.FileExplorerTabs import FileExplorerTabs






def load_cached_tabs(pages_dict: dict, file_tabs_obj: FileExplorerTabs, switch_func, app_page=None):
    """
    Scans CACHE_BASE for folders and re-creates tabs.
    Uses tabs_index.json to restore creation order and titles where possible.
    """
    try:
        if not CACHE_BASE.exists():
            return

        idx = tabs_script._load_tabs_index()
        valid_ids = {f.name for f in CACHE_BASE.iterdir() if f.is_dir()}
        order = [tid for tid in idx.get("order", []) if tid in valid_ids]
        tabs_script._save_tabs_index({"order": order, "last_active_tab": idx.get("last_active_tab")})
        order = idx.get("order", [])
        processed = set()
        idx = tabs_script._load_tabs_index()



        for tab_id in order:
            folder = CACHE_BASE / tab_id
            processed.add(tab_id)
            if not folder.exists() or not folder.is_dir():
                continue

            state = tabs_script.read_tab_state(tab_id)

            if not state:
                continue

            try:
                page_type = state.get("page") or state.get("page_type") or "SetupPage"



                if page_type == "ChannelPage" and state.get("signed_in") and state.get("keyring_name"):

                    keyname = state.get("keyring_name")
                    try:
                        token_data = tabs_script.restore_tokens_from_keyring(keyname)
                    except Exception as e:
                        token_data = None
                        print(f"restore_tokens_from_keyring error for {tab_id}: {e}")

                    if token_data:

                        try:
                            tmp_setup = SetupPage(
                                "tmp",
                                app_page=app_page,
                                pages_dict=pages_dict,
                                file_tabs=file_tabs_obj,
                                switch_func=switch_func,
                                tab_id=tab_id
                            )

                            platform_inner = tmp_setup.platform_inner


                            page_obj = ChannelPage(
                                state.get("title") or "Channel",
                                token_data=token_data,
                                user_info={"sub": state.get("user_id")},
                                app_page=app_page,
                                tab_id=tab_id,
                                keyring_name=keyname,
                                platform_inner=platform_inner
                            )

                            pages_dict[tab_id] = page_obj
                            file_tabs_obj.add_tab(tab_id, page_obj, activate=False)
                        except Exception as e:
                            print(f"Failed to recreate ChannelPage for {tab_id}: {e}")

                            page_obj = None
                    else:

                        page_obj = None

                    if page_obj is None:
                        try:
                            page_obj = SetupPage(state.get("title") or "Setup", app_page=app_page,
                                                 pages_dict=pages_dict, file_tabs=file_tabs_obj,
                                                 switch_func=switch_func, tab_id=tab_id)
                            pages_dict[tab_id] = page_obj
                            file_tabs_obj.add_tab(tab_id, page_obj, activate=False)
                        except Exception as e:
                            print(f"Failed to add fallback SetupPage for {tab_id}: {e}")

                else:

                    try:
                        page_obj = SetupPage(state.get("title") or "Setup", app_page=app_page,
                                             pages_dict=pages_dict, file_tabs=file_tabs_obj,
                                             switch_func=switch_func, tab_id=tab_id)
                        pages_dict[tab_id] = page_obj
                        file_tabs_obj.add_tab(tab_id, page_obj, activate=False)
                    except Exception as e:
                        print(f"Failed to recreate tab {tab_id}: {e}")

            except Exception as e:
                print(f"Error processing cached tab {tab_id}: {e}")


        for folder in CACHE_BASE.iterdir():
            if not folder.is_dir():
                continue
            tid = folder.name
            if tid in processed:
                continue
            state = tabs_script.read_tab_state(tid) or {}
            try:
                page_obj = SetupPage(state.get("title") or "Setup", app_page=app_page,
                                     pages_dict=pages_dict, file_tabs=file_tabs_obj,
                                     switch_func=switch_func, tab_id=tid)
                pages_dict[tid] = page_obj
                file_tabs_obj.add_tab(tid, page_obj, activate=False)
            except Exception as e:
                print(f"Failed to add unindexed tab {tid}: {e}")

        # restore last active tab if valid, else activate first available
        try:
            last = idx.get("last_active_tab")
            if last and last in pages_dict and last in file_tabs_obj.tabs:
                file_tabs_obj.switch_tab(last)
            elif file_tabs_obj.order:
                file_tabs_obj.switch_tab(file_tabs_obj.order[0])
        except Exception as e:
            print(f"Failed to restore last active tab: {e}")

    except Exception as e:
        print(f"load_cached_tabs fatal error: {e}")



# ----------------------------------------------------------------------Pages------------------------------------------------------------------------------




# ----------------------------------------------------------------------Host & App------------------------------------------------------------------------------
class PageHost:
    def __init__(self):
        self.container = ft.Column(expand=True, alignment=ft.MainAxisAlignment.CENTER,
                                   horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def show(self, page_obj: PageBase):
        if page_obj is None:
            self.container.controls = [
                ft.Container(
                    content=ft.Text("No tab open", color=FG, text_align="center"),
                    padding=16, bgcolor=BG, alignment=ft.alignment.center
                )
            ]
        else:
            # IMPORTANT: no scroll, no column, no wrapper
            self.container.controls = [page_obj.content()]

        try:
            self.container.update()
        except:
            pass


# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       MAIN                                                                                      |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


def main(page: ft.Page):
    import platform
    import ctypes
    import time

    def make_process_dpi_aware_windows():
        try:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
            try:
                ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
            except Exception:
                pass
        except Exception:
            pass

    if platform.system() == "Windows":
        make_process_dpi_aware_windows()

    page.title = "SulfurAI"
    page.bgcolor = BG

    page.window_resizable = False
    page.window_width = UI_WIDTH
    page.window_height = UI_HEIGHT

    try:
        page.window_min_width = UI_WIDTH
        page.window_max_width = UI_WIDTH
        page.window_min_height = UI_HEIGHT
        page.window_max_height = UI_HEIGHT
    except Exception:
        pass

    try:
        win = getattr(page, "window", None)
        if win is not None:
            try:
                win.resizable = False
            except Exception:
                pass
            try:
                win.width = UI_WIDTH
                win.height = UI_HEIGHT
            except Exception:
                pass
            try:
                win.min_width = UI_WIDTH
                win.max_width = UI_WIDTH
                win.min_height = UI_HEIGHT
                win.max_height = UI_HEIGHT
            except Exception:
                pass
            try:
                if hasattr(win, "title_bar_hidden"):
                    win.title_bar_hidden = False
            except Exception:
                pass
    except Exception:
        pass

    try:
        page.window_center()
    except Exception:
        pass

    page.data = {"scale": 1.0}
    page.on_resize = None

    pages = {}
    host = PageHost()

    def switch(tab_id):
        if tab_id is None:
            host.show(None)
            return
        host.show(pages.get(tab_id))

    next_tab_index = {"i": 1}

    def close_tab(tab_id):
        if tab_id in pages:
            del pages[tab_id]
        tabs_script.delete_tab_cache(tab_id)

    def create_new_tab(preferred_name=None):
        idx = next_tab_index["i"]
        next_tab_index["i"] += 1
        name = preferred_name or f"Setup"
        tab_id = f"tab_{secrets.token_hex(4)}"
        pages[tab_id] = SetupPage(
            name,
            app_page=page,
            pages_dict=pages,
            file_tabs=file_tabs,
            switch_func=switch,
            tab_id=tab_id
        )
        file_tabs.add_tab(tab_id, pages[tab_id])
        tabs_script.set_last_active_tab(tab_id)
        tabs_script.write_tab_state(tab_id,
                        {"page": "SetupPage", "signed_in": False, "user_id": None, "keyring_name": None, "title": name})
        tabs_script.register_tab_in_index(tab_id, title=name, page_type="SetupPage")

    file_tabs = FileExplorerTabs(
        on_switch=switch,
        on_new_tab=lambda: create_new_tab(),
        on_close_tab=close_tab,
        app_page=page
    )

    main_area = ft.Column([file_tabs.build(), host.container], expand=True, alignment=ft.MainAxisAlignment.CENTER)

    layout = ft.Column([main_area], expand=True)

    page.add(layout)

    load_cached_tabs(pages, file_tabs, switch, app_page=page)

    try:
        last = tabs_script.get_last_active_tab()
        if last and last in pages and last in file_tabs.tabs:
            file_tabs.switch_tab(last)
        else:
            # fallback: first existing tab
            if file_tabs.order:
                file_tabs.switch_tab(file_tabs.order[0])
    except Exception as e:
        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"Error restoring last active tab: {e}")

    if len(pages) == 0:
        create_new_tab("Setup")


if __name__ == "__main__":
    ft.app(target=main)
