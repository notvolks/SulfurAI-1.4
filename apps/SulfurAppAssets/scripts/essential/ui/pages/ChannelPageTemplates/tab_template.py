import flet as ft
from flet.plotly_chart import PlotlyChart
from pathlib import Path
import shutil
import keyring
import traceback
base = Path(__file__).resolve().parents[7]
CACHE_BASE = Path(base / "apps" / "SulfurAppAssets" / "cache" / "tabs")
CACHE_BASE.mkdir(parents=True, exist_ok=True)
from apps.SulfurAppAssets.globalvar import global_var
from apps.SulfurAppAssets.scripts.essential.ui import tabs_script
BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()


def tab_template_simple(
    tabs: list[tuple[str, ft.Control]],
    page=None,
    width=None,
    height=None,
):
    """
    A minimal tabs container.
    - Top row: rounded tab buttons
    - Bottom: active tab content
    - Host container has no border or background (invisible)
    - NO UI for adding tabs
    """

    if not tabs:
        return ft.Container(
            content=ft.Column(controls=[]),
            width=width,
            height=height
        )

    active_index = 0

    content_area = ft.Column(
        controls=[tabs[0][1]],
        expand=True,
        scroll="auto"
    )

    tab_buttons = []

    def select_tab(idx: int):
        nonlocal active_index
        if 0 <= idx < len(tabs):
            active_index = idx
            content_area.controls = [tabs[idx][1]]
            for i, btn in enumerate(tab_buttons):
                btn.border = ft.border.all(2, ORANGE) if i == active_index else ft.border.all(1, ORANGE)
            if page and getattr(page, "update", None):
                page.update()

    for i, (label, _) in enumerate(tabs):
        btn = ft.Container(
            content=ft.Text(label, color=FG),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border=ft.border.all(1, ORANGE),
            border_radius=8,               # rounded corners
            bgcolor=BG,
            on_click=lambda e, idx=i: select_tab(idx),
        )
        tab_buttons.append(btn)

    # highlight the first
    tab_buttons[0].border = ft.border.all(2, ORANGE)

    tab_row = ft.Row(
        controls=tab_buttons,
        spacing=6
    )

    host = ft.Container(
        content=ft.Column(
            controls=[
                tab_row,
                ft.Container(height=6),
                content_area
            ],
            expand=True
        ),
        width=width,
        height=height,

        # invisible host container
        bgcolor=None,
        border=None,
        padding=0,
    )

    host.select = select_tab
    host.content_area = content_area

    return host

def tab_template_advanced(
    *,
    initial_tabs: list[tuple[str, ft.Control]] | None = None,
    tab_content_template: ft.Control = None,
    page=None,
    width=None,
    height=None,
    on_tabs_changed=None,   # 🔑 ADD THIS
):

    """
    Advanced tabs host:

    - shows initial UI tabs
    - UI "+" button creates new tabs (NOT the calling code)
    - all new tabs use tab_content_template
    - content is a real container
    - tabs have close buttons
    - tab history stored in local list
    """

    initial_tabs = list(initial_tabs or [])

    tabs = {}
    order = []
    active = None
    history = []
    counter = 1

    content_area = ft.Column(controls=[], expand=True, scroll="auto")
    tab_row = ft.Row(spacing=6, wrap=False, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _notify_tabs_changed():
        if callable(on_tabs_changed):
            try:
                on_tabs_changed()
            except Exception as e:
                print("on_tabs_changed failed:", e)


    def _apply_styles():
        for tid in order:
            btn = tabs[tid]["button"]
            if tid == active:
                btn.border = ft.border.all(2, ORANGE)
            else:
                btn.border = ft.border.all(1, ORANGE)

    def select_tab(tab_id: str, ev=None):
        nonlocal active
        if tab_id not in tabs:
            return

        active = tab_id
        content_area.controls = [tabs[tab_id]["control"]]

        _apply_styles()

        if page and getattr(page, "update", None):
            page.update()

    def close_tab(tab_id: str):
        nonlocal active

        if tab_id not in tabs:
            return

        idx = order.index(tab_id)
        del order[idx]
        del tabs[tab_id]

        if active == tab_id:
            if order:
                new_active = order[max(0, idx - 1)]
                active = new_active
                content_area.controls = [tabs[new_active]["control"]]
            else:
                active = None
                content_area.controls = [ft.Container(content=ft.Text("No tabs"), padding=8)]

        _rebuild_row()
        _apply_styles()

        if page and getattr(page, "update", None):
            page.update()
        _notify_tabs_changed()

    def _make_tab_button(tab_id: str, title: str):
        lbl = ft.Text(title, color=FG)

        close_btn = ft.TextButton(
            "✕",
            on_click=lambda e, tid=tab_id: close_tab(tid),
            style=ft.ButtonStyle(text_style=ft.TextStyle(color=FG))
        )

        row = ft.Row([lbl, close_btn], alignment=ft.MainAxisAlignment.CENTER)

        return ft.Container(
            content=row,
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=8,
            bgcolor=BG,
            border=ft.border.all(1, ORANGE),
            on_click=lambda e, tid=tab_id: select_tab(tid),
        )

    def add_tab_from_ui(e=None):
        nonlocal counter, active

        if tab_content_template is None:
            # If no template given, fallback
            new_content = ft.Container(content=ft.Text("New tab"), padding=12)
        else:
            # Clone template
            new_content = ft.Container(
                content=tab_content_template.content,
                padding=tab_content_template.padding
            )

        new_id = f"Video {counter}"
        counter += 1

        btn = _make_tab_button(new_id, new_id)

        tabs[new_id] = {"button": btn, "control": new_content}
        order.append(new_id)
        history.append(new_id)

        _rebuild_row()
        select_tab(new_id)

        _notify_tabs_changed()


    add_btn = ft.Container(
        content=ft.Text("+", color=ORANGE),
        padding=ft.padding.symmetric(horizontal=10, vertical=6),
        border_radius=8,
        border=ft.border.all(1, ORANGE),
        bgcolor=BG,
        on_click=add_tab_from_ui
    )

    def _rebuild_row():
        tab_row.controls = [tabs[t]["button"] for t in order] + [add_btn]

    # Load initial UI tabs
    for label, ctrl in initial_tabs:
        btn = _make_tab_button(label, label)
        tabs[label] = {"button": btn, "control": ctrl}
        order.append(label)
        history.append(label)

    if order:
        active = order[0]
        content_area.controls = [tabs[active]["control"]]

    _rebuild_row()
    _apply_styles()

    host = ft.Container(
        content=ft.Column(
            controls=[
                tab_row,
                ft.Container(height=6),
                content_area
            ],
            expand=True
        ),
        width=width,
        height=height,
        bgcolor=None,
        border=None,
        padding=0
    )

    def add_tab(title: str, control: ft.Control | None = None):
        nonlocal counter, active

        # choose a unique internal id / key for the tab
        new_id = str(title or f"Video {counter}")
        if new_id in tabs:
            # avoid clobbering existing tab id by suffixing the counter
            new_id = f"{new_id} {counter}"
            counter += 1

        # Build a control if none provided (mirror add_tab_from_ui behaviour)
        if control is None:
            if tab_content_template is None:
                control = ft.Container(content=ft.Text("New tab"), padding=12)
            else:
                # Note: we intentionally create a new Container wrapping the template's content.
                # If the template reuses the same inner controls, you may want to clone them more
                # explicitly to avoid shared-control identity — see comments below.
                control = ft.Container(
                    content=tab_content_template.content,
                    padding=getattr(tab_content_template, "padding", 12)
                )

        # create button and register internal structures
        btn = _make_tab_button(new_id, title or new_id)
        tabs[new_id] = {"button": btn, "control": control}
        order.append(new_id)
        history.append(new_id)

        # re-render tab row (and make the new tab active)
        _rebuild_row()
        select_tab(new_id)

        # notify any listeners and return the new id
        _notify_tabs_changed()
        return new_id

    # expose the add_tab API so ChannelPage can call it
    host.add_tab = add_tab

    host.get_history = lambda: list(history)
    host.close_tab = close_tab
    host.select_tab = select_tab

    def _get_tabs_state():
        """
        Return list of tabs state entries. Each entry is:
          {"title": <tab id>, "content": {"ideas": "...", "format_ideas": "...", "sound_ideas": "..."}}
        We traverse the stored control tree and collect up to 3 TextField values in traversal order.
        """
        state = []

        for tid in order:
            # default blanks
            content_data = {"ideas": "", "format_ideas": "", "sound_ideas": ""}
            try:
                ctrl = tabs.get(tid, {}).get("control")
                if ctrl is not None:
                    # recursively collect TextField values (in traversal order)
                    def extract_all(c, out=None):
                        if out is None:
                            out = []
                        try:
                            # If it's a TextField, get its value
                            if isinstance(c, ft.TextField):
                                v = getattr(c, "value", "") or ""
                                out.append(v)
                                return out
                            # dive into Container.content if present
                            if hasattr(c, "content") and c.content is not None:
                                extract_all(c.content, out)
                            # dive into controls if present
                            if hasattr(c, "controls") and c.controls:
                                for ch in c.controls:
                                    extract_all(ch, out)
                        except Exception:
                            pass
                        return out

                    vals = extract_all(ctrl) or []
                    content_data["ideas"] = vals[0] if len(vals) > 0 else ""
                    content_data["format_ideas"] = vals[1] if len(vals) > 1 else ""
                    content_data["sound_ideas"] = vals[2] if len(vals) > 2 else ""
            except Exception:
                pass

            state.append({
                "title": tid,
                "content": content_data,
            })

        return state

    host.get_tabs_state = _get_tabs_state

    return host