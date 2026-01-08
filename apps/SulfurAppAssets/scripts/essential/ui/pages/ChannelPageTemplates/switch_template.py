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


def switch_template(
    label: str,
    *,
    value: bool = False,
    on_change=None,
    page=None,
    width: int | None = None,
    # sizing uses tabs_script so it matches the rest of your UI
):
    """
    Returns a single-row control: [ Label | Switch ]
    - label: text on the left
    - value: initial boolean state
    - on_change: optional callback called as on_change(new_value, ev) when toggled
    - width: optional overall width (the row/container)
    The switch is rectangular with slightly rounded corners and uses ORANGE for the "on" appearance.
    """

    # sizing that matches the rest of your UI
    track_w = int(tabs_script.sh(64, page))
    track_h = int(tabs_script.sv(30, page))
    thumb_w = int(tabs_script.sh(26, page))
    thumb_h = int(tabs_script.sv(26, page))
    pad = max(2, (track_h - thumb_h) // 2)

    # computed left positions for thumb
    left_off = pad
    left_on = track_w - thumb_w - pad

    # state holder so closures can mutate
    state = {"value": bool(value)}

    # left label container
    label_ctrl = ft.Container(
        content=ft.Text(label, color=FG, size=int(tabs_script.s(14, page))),
        padding=ft.padding.symmetric(horizontal=8, vertical=6),
        bgcolor=None,
    )

    # track content (Text will be replaced on toggle)
    track_text = ft.Text("ON" if state["value"] else "OFF",
                         color=BG if state["value"] else FG,
                         size=int(tabs_script.s(12, page)))

    track = ft.Container(
        content=ft.Row(controls=[ft.Container(expand=True, content=track_text)], alignment=ft.MainAxisAlignment.CENTER),
        width=track_w,
        height=track_h,
        padding=0,
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(6, page)),  # slightly rounded corners
        bgcolor=ORANGE if state["value"] else BG,
    )

    # thumb (the sliding knob)
    thumb = ft.Container(
        width=thumb_w,
        height=thumb_h,
        border_radius=int(tabs_script.s(6, page)),
        border=ft.border.all(1, ORANGE),
        bgcolor=BG,  # contrasts the track
    )

    # stack to position thumb over track
    switch_stack = ft.Stack(
        controls=[
            track,
            ft.Container(
                content=thumb,
                top=pad,
                left=(left_on if state["value"] else left_off),
                expand=False,
            ),
        ],
        width=track_w,
        height=track_h,
        expand=False,
    )

    # click handler toggles state and updates visuals
    def _safe_update():
        try:
            if page and getattr(page, "update", None):
                page.update()
                return
        except Exception:
            pass
        try:
            thumb.update()
        except Exception:
            pass
        try:
            track.update()
        except Exception:
            pass

    def _toggle(ev=None):
        # flip
        state["value"] = not state["value"]

        # update track bg + inner text color
        try:
            track.bgcolor = ORANGE if state["value"] else BG
            # update text color so it contrasts appropriately
            track_text.value = "ON" if state["value"] else "OFF"
            track_text.color = BG if state["value"] else FG
        except Exception:
            pass

        # reposition thumb container inside the Stack by replacing the positioned container
        try:
            # remove existing thumb positioned container and re-add at new left
            # find positioned container (the one that has content == thumb)
            for c in list(switch_stack.controls):
                if getattr(c, "content", None) is thumb:
                    switch_stack.controls.remove(c)
                    break
            switch_stack.controls.append(
                ft.Container(
                    content=thumb,
                    top=pad,
                    left=(left_on if state["value"] else left_off),
                    expand=False,
                )
            )
        except Exception:
            pass

        _safe_update()

        # call callback if provided
        try:
            if callable(on_change):
                # prefer giving both value and event for flexibility
                on_change(state["value"], ev)
        except Exception:
            pass

    # allow clicking the track or thumb to toggle
    # set on_click on both track and thumb positioned container
    # track.on_click set on the track container itself
    track.on_click = _toggle

    # make a host container for the stack so we can attach click too (safe)
    host_stack_container = ft.Container(
        content=switch_stack,
        padding=ft.padding.symmetric(horizontal=2, vertical=2),
        bgcolor=None,
        on_click=_toggle,  # clicking anywhere on the widget toggles
    )

    # build the row: label on left, switch on right
    row = ft.Row(
        controls=[
            ft.Container(content=label_ctrl, expand=True, bgcolor=None),
            host_stack_container,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=int(tabs_script.s(8, page)),
    )

    # wrap in outer container to match your other templates' look (border optional)
    outer = ft.Container(
        content=row,
        width=int(tabs_script.sh(width, page)) if width else None,
        padding=ft.padding.symmetric(horizontal=6, vertical=6),
        bgcolor=None,
    )

    # expose a small API on the returned container to query/change state programmatically
    outer.get_value = lambda: state["value"]
    def _set_value(v, ev=None):
        if bool(v) == state["value"]:
            return
        _toggle(ev)
    outer.set_value = _set_value
    outer.on_change = on_change  # keep reference

    return outer
