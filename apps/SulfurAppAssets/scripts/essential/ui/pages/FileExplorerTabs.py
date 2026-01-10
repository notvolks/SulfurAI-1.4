from apps.SulfurAppAssets.scripts.verification import sulfuroauth
import os
import json
import flet as ft
from pathlib import Path
import shutil
import keyring
base = Path(__file__).resolve().parents[6]
CACHE_BASE = Path(base / "apps" / "SulfurAppAssets" / "cache" / "tabs")
CACHE_BASE.mkdir(parents=True, exist_ok=True)
from apps.SulfurAppAssets.globalvar import global_var
from apps.SulfurAppAssets.scripts.essential.ui import tabs_script
BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()


class PageBase:
    def __init__(self, title: str):
        self.title = title

    def content(self):
        raise NotImplementedError

#-----------------------------------------------------------------

class FileExplorerTabs:
    def __init__(self, on_switch, on_new_tab, on_close_tab, app_page=None):
        self.app_page = app_page

        self.tabs = {}
        self.order = []
        self.active = None
        self.on_switch = on_switch
        self.on_new_tab = on_new_tab
        self.on_close_tab = on_close_tab
        self.row = ft.Row(spacing=6, vertical_alignment="center", wrap=False)

    # -------------------ESSENTIALS----------------------
    def replace_tab(self, tab_id: str, page_obj: PageBase):
        """
        Replace the Page object for an existing tab and rebuild its button with the new title.
        """
        if tab_id not in self.tabs:
            return self.add_tab(tab_id, page_obj)
        # replace page_obj
        try:

            new_btn = self._make_tab_button(tab_id, page_obj)
            # replace stored page/button
            self.tabs[tab_id]["page"] = page_obj
            self.tabs[tab_id]["button"] = new_btn
            # keep order intact
            self._refresh()
        except Exception as e:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"replace_tab error: {e}")

    # ------------------UI----------------------

    def _make_tab_button(self, tab_id: str, page_obj: PageBase):
        lbl = ft.Text(
            page_obj.title,
            size=int(tabs_script.s(16, self.app_page)),
            weight="w700",
            color=FG,
            text_align="center"
        )

        close_btn = ft.TextButton(
            "✕",
            width=int(tabs_script.sh(28, self.app_page)),
            height=int(tabs_script.sv(28, self.app_page)),
            on_click=lambda e, tid=tab_id: self.close_tab(tid),
            style=ft.ButtonStyle(text_style=ft.TextStyle(color=FG))
        )

        return ft.Container(
            content=ft.Row(
                [lbl, ft.Container(width=int(tabs_script.s(10, self.app_page))), close_btn],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            padding=ft.padding.symmetric(
                horizontal=int(tabs_script.sh(18, self.app_page)),
                vertical=int(tabs_script.sv(10, self.app_page))
            ),
            border_radius=int(tabs_script.s(14, self.app_page)),
            bgcolor=BG,
            border=ft.border.all(1, ORANGE),
            on_click=lambda e, tid=tab_id: self.switch_tab(tid)
        )

    def add_tab(self, tab_id: str, page_obj: PageBase, activate: bool = True):
        """
        Add a new tab. If activate is False, the tab is added but not made active (useful during startup).
        """
        if tab_id in self.tabs:

            if activate:
                self.switch_tab(tab_id)
            return

        btn = self._make_tab_button(tab_id, page_obj)
        self.tabs[tab_id] = {"page": page_obj, "button": btn}
        self.order.append(tab_id)

        if activate:
            self.active = tab_id

            try:
                tabs_script.set_last_active_tab(tab_id)
            except Exception:
                pass
            self._refresh()
            self.on_switch(tab_id)
        else:

            self._refresh()

    def switch_tab(self, tab_id: str):
        if tab_id not in self.tabs:
            return
        self.active = tab_id

        try:
            tabs_script.set_last_active_tab(tab_id)
        except Exception:
            pass

        self._refresh()
        self.on_switch(tab_id)

    def close_tab(self, tab_id: str):
        if tab_id not in self.tabs:
            return
        self.on_close_tab(tab_id)
        idx = self.order.index(tab_id)
        del self.order[idx]
        del self.tabs[tab_id]
        if self.active == tab_id:
            if len(self.order) > 0:
                new_active = self.order[max(0, idx - 1)]
                self.active = new_active
                self.on_switch(new_active)
            else:
                self.active = None
                self.on_switch(None)
        self._refresh()

    def _refresh(self):
        controls = []
        for tid in self.order:
            btn = self.tabs[tid]["button"]
            if tid == self.active:
                btn.bgcolor = BG
                btn.border = ft.border.all(2, ORANGE)
            else:
                btn.bgcolor = BG
                btn.border = ft.border.all(1, ORANGE)
            controls.append(btn)

        add_btn = ft.Container(
            content=ft.Row([ft.Text("+", size=int(tabs_script.s(16, self.app_page)), weight="bold", color=ORANGE),
                            ft.Container(width=int(tabs_script.s(6, self.app_page)))],
                           alignment=ft.MainAxisAlignment.CENTER),
            padding=ft.padding.symmetric(horizontal=int(tabs_script.s(10, self.app_page)), vertical=int(tabs_script.s(6, self.app_page))),
            border_radius=int(tabs_script.s(12, self.app_page)),
            border=ft.border.all(1, ORANGE),
            bgcolor=BG,
            on_click=lambda e: self.on_new_tab()
        )
        controls.append(add_btn)

        self.row.controls = controls
        try:
            self.row.update()
        except Exception:
            pass

    def build(self):
        top = ft.Container(
            content=self.row,
            padding=ft.padding.symmetric(horizontal=int(tabs_script.s(12, self.app_page)),
                                         vertical=int(tabs_script.s(8, self.app_page)))
            ,
            bgcolor=BG,
            border=ft.border.only(bottom=ft.BorderSide(1, ORANGE))
        )
        return top