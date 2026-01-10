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


# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       UI TEMPLATES                                                                                     |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

def _size_s(x, page):

    return int(tabs_script.s(x, page))

def create_scrollable_panel(title: str, content_widgets: list, page=None,
                            width=None, height=None):
    """
    Returns a styled ft.Container that contains a Column with scroll='auto'.
    `content_widgets` must be new instances (the builder does not copy).
    """
    title_text = ft.Text(title, weight="bold", color=FG, size=_size_s(14, page))
    column = ft.Column(
        controls=[title_text, ft.Container(height=_size_s(8, page))] + content_widgets,
        spacing=int(tabs_script.sv(8, page)),
        horizontal_alignment=ft.CrossAxisAlignment.START,
        alignment=ft.MainAxisAlignment.START,
        scroll="auto",
    )
    return ft.Container(
        content=column,
        padding=ft.padding.all(int(tabs_script.s(14, page))),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(12, page)),
        bgcolor=BG,
        width=width or int(tabs_script.sh(420, page)),
        height=height or int(tabs_script.sv(220, page)),
    )

def create_text_block(title: str, text: str, page=None, width=None, height=None):
    t = ft.Text(title, weight="bold", color=FG, size=_size_s(14, page))
    p = ft.Text(text, color=FG, size=_size_s(12, page))
    return ft.Container(
        content=ft.Column([t, ft.Container(height=int(tabs_script.sv(6, page))), p], spacing=8),
        padding=ft.padding.all(int(tabs_script.s(14, page))),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(12, page)),
        bgcolor=BG,
        width=width or int(tabs_script.sh(420, page)),
        height=height or int(tabs_script.sv(220, page)),
    )


def make_dashboard_lines(num_lines: int, page=None):
    out = []
    for i in range(num_lines):
        out.append(ft.Text(
            f"Test line {i + 1}: Lorem ipsum dolor sit amet, consectetur adipiscing elit — filler content.",
            size=int(tabs_script.s(12, page)),
            color=FG,
        ))
        out.append(ft.Container(height=int(tabs_script.sv(6, page))))
    return out
