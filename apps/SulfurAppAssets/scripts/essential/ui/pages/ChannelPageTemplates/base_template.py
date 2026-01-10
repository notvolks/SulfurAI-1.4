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


def base_template(
    *,
    children=None,
    width=None,
    height=None,
    padding=None,
    bgcolor=BG,
    border_color=ORANGE,
    border_width=1,
    border_radius=12,
    page=None,
):
    """
    Generic, non-scrollable layout container.

    - Supports multiple children
    - Content-agnostic
    - No internal UI logic
    - Safe inside a scrollable parent Column
    """

    if children is None:
        children = []

    inner = ft.Column(
        controls=list(children),
        spacing=int(tabs_script.sv(6, page)),
        horizontal_alignment=ft.CrossAxisAlignment.START,
        expand=True,
    )

    return ft.Container(
        content=inner,
        width=width,
        height=height,
        padding=padding or ft.padding.all(int(tabs_script.s(12, page))),
        bgcolor=bgcolor,
        border=ft.border.all(border_width, border_color),
        border_radius=int(tabs_script.s(border_radius, page)),
    )