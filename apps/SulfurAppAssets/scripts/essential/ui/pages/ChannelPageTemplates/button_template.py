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


def button_template(
        title: str,
        base: ft.Container,
        width: int = None,
        height: int = None,
        page=None,
        host: ft.Stack | None = None,
        placement: str = "center",  # "center","top","bottom","top_left","top_right","bottom_left","bottom_right"
        x_offset: int | None = None,  # optional absolute x offset (px) relative to host top-left
        y_offset: int | None = None,  # optional absolute y offset (px) relative to host top-left
):
    """
    Robust popup-button (extended):
    - Inserts the supplied `base` Container into `host` (an ft.Stack).
    - `placement` controls alignment inside the host Stack (alignment-based positioning).
    - If x_offset or y_offset is provided the popup will be placed using absolute
      top/left coordinates inside the host (positioned), which overrides `placement`.
    - Uses host.update() when host is mounted; otherwise falls back to e.page.update()
      so the popup is visible even when the host Stack wasn't attached yet.
    - Child elements in `base` can access the close function via base.data['close_popup']
    """
    btn_w = width or int(tabs_script.sh(200, page))
    btn_h = height or int(tabs_script.sv(42, page))

    # Initialize data dict if not present
    if not hasattr(base, 'data') or base.data is None:
        base.data = {}

    btn = ft.Container(
        content=ft.Text(title, size=int(tabs_script.s(14, page)), color=FG),
        alignment=ft.alignment.center,
        width=btn_w,
        height=btn_h,
        padding=ft.padding.symmetric(
            horizontal=int(tabs_script.s(8, page)),
            vertical=int(tabs_script.sv(6, page)),
        ),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(8, page)),
    )

    def _find_ancestor_stack(ctrl):
        p = getattr(ctrl, "parent", None)
        while p:
            try:
                if getattr(p, "__class__", type(p)).__name__ == "Stack":
                    return p
            except Exception:
                pass
            p = getattr(p, "parent", None)
        return None

    def _safe_update(target, ev):
        """
        Try to update `target` if mounted; otherwise fall back to ev.page.update()
        (ev will be None only in extremely rare programmatic calls).
        """
        try:
            if getattr(target, "page", None) is not None:
                target.update()
                return
        except Exception:
            pass
        try:
            if ev is not None and getattr(ev, "page", None) is not None:
                ev.page.update()
                return
        except Exception:
            pass
        try:
            p = target
            while p is not None:
                if getattr(p, "page", None) is not None:
                    p.update()
                    return
                p = getattr(p, "parent", None)
        except Exception:
            pass

    def _on_click(e):
        nonlocal host
        # Resolve host (prefer explicit host; fallback to nearest ancestor Stack)
        if host is None:
            host = _find_ancestor_stack(e.control)
        if host is None:
            # no sensible place to attach popup; bail out safely
            return

        # prevent double-insert
        if getattr(btn, "_popup_attached", False):
            return

        # close handler (will remove popup_container when invoked)
        def _close_popup(ev):
            try:
                if popup_container in host.controls:
                    host.controls.remove(popup_container)
                btn._popup_attached = False
                _safe_update(host, ev)
            except Exception:
                pass

        # Store close function in base.data so child elements can access it
        if isinstance(base.data, dict):
            base.data['close_popup'] = _close_popup
        else:
            base.data = {'close_popup': _close_popup}

        # small, version-safe close button using Text "×"
        close_btn = ft.Container(
            content=ft.Text("×", size=int(tabs_script.s(16, page)), weight="w700", color=FG),
            width=int(tabs_script.s(28, page)),
            height=int(tabs_script.s(28, page)),
            alignment=ft.alignment.center,
            border=ft.border.all(1, ORANGE),
            border_radius=int(tabs_script.s(8, page)),
            on_click=_close_popup,
        )

        popup_stack = ft.Stack(
            controls=[
                base,
                ft.Container(
                    content=close_btn,
                    top=0,
                    right=0,
                    padding=ft.padding.all(int(tabs_script.s(8, page))),
                ),
            ],
            expand=False,
        )

        # Alignment map for placement-based positioning inside host Stack
        _alignment_map = {
            "center": ft.alignment.center,
            "top": ft.alignment.top_center,
            "bottom": ft.alignment.bottom_center,
            "top_left": ft.alignment.top_left,
            "top_right": ft.alignment.top_right,
            "bottom_left": ft.alignment.bottom_left,
            "bottom_right": ft.alignment.bottom_right,
        }

        # If offsets provided, place absolutely using top/left (positioned child).
        # Otherwise use alignment + expand=True (original behavior).
        if x_offset is not None or y_offset is not None:
            # use absolute positioning relative to host top-left
            popup_container = ft.Container(
                content=popup_stack,
                top=(y_offset if y_offset is not None else 0),
                left=(x_offset if x_offset is not None else 0),
                # positioned children should not expand (they float)
                expand=False,
            )
        else:
            # alignment-based overlay that expands to the host and aligns inner content
            popup_container = ft.Container(
                content=popup_stack,
                alignment=_alignment_map.get(placement, ft.alignment.center),
                expand=True,
            )

        try:
            host.controls.append(popup_container)
            btn._popup_attached = True
            btn._popup_container_ref = popup_container  # Store reference for external close
            btn._host_ref = host  # Store host reference
            # prefer host.update(); fallback to event page update so popup shows immediately
            _safe_update(host, e)
        except Exception:
            pass

    btn.on_click = _on_click

    # Expose a method to programmatically close the popup (same as X button)
    def close_popup():
        """Close the popup if it's currently open - same behavior as X button"""
        if hasattr(btn, '_popup_attached') and btn._popup_attached:
            try:
                if hasattr(btn, '_host_ref') and hasattr(btn, '_popup_container_ref'):
                    host_ref = btn._host_ref
                    popup_container = btn._popup_container_ref
                    if popup_container in host_ref.controls:
                        host_ref.controls.remove(popup_container)
                    btn._popup_attached = False
                    _safe_update(host_ref, None)
            except Exception:
                pass

    btn.close_popup = close_popup
    return btn