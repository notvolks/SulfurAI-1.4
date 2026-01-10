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


def graph(
    data=None,
    title=None,
    graph_type="line",
    width=None,
    height=220,
    x_axis_title=None,
    y_axis_title=None,
    show_axis_title_on_hover=True,
    include_toggle=False,
):
    """
    Flet-only graph renderer with tooltips + axis hover overlay applied both
    on initial render and after in-place re-renders (switch / set_data).
    """

    import traceback

    # --------------------------------------------------
    # Environment + config
    # --------------------------------------------------
    try:
        hover_supported = ft.Page().platform not in ("android", "ios")
    except Exception:
        hover_supported = True

    _cycle_map = {"line": "bar", "bar": "pie", "pie": "line"}

    # --------------------------------------------------
    # Try chart classes (guarded)
    # --------------------------------------------------
    try:
        LineChart = ft.LineChart
        LineChartData = ft.LineChartData
        LineChartDataPoint = ft.LineChartDataPoint
        ChartAxis = ft.ChartAxis
        ChartAxisLabel = ft.ChartAxisLabel
        PieChart = ft.PieChart
        PieChartSection = ft.PieChartSection
        BarChart = ft.BarChart
        BarChartGroup = ft.BarChartGroup
        BarChartRod = ft.BarChartRod
    except Exception:
        LineChart = LineChartData = LineChartDataPoint = None
        ChartAxis = ChartAxisLabel = None
        PieChart = PieChartSection = None
        BarChart = BarChartGroup = BarChartRod = None

    points = list(data or [])

    # --------------------------------------------------
    # Overlay for axis hover
    # --------------------------------------------------
    overlay_text = ft.Text("", size=10, color=FG)
    overlay_container = ft.Container(
        content=overlay_text,
        padding=ft.padding.symmetric(horizontal=8, vertical=6),
        bgcolor="#222222",
        border_radius=6,
        visible=False,
    )

    def _show_overlay(text):
        try:
            overlay_text.value = text
            overlay_container.visible = True
            if hasattr(overlay_container, "update"):
                overlay_container.update()
        except Exception:
            pass

    def _hide_overlay():
        try:
            overlay_container.visible = False
            if hasattr(overlay_container, "update"):
                overlay_container.update()
        except Exception:
            pass

    # --------------------------------------------------
    # Helper: axis labels with hover handlers (reusable)
    # --------------------------------------------------
    def _make_axis_labels(values, axis_name):
        out = []
        for v in values:
            try:
                lbl = ft.Text(str(round(v, 2)), size=10, color=FG)
                if show_axis_title_on_hover and hover_supported:
                    # closure to capture v and axis_name
                    def _on_enter(ev, value=v, a_name=axis_name):
                        try:
                            axis_t = x_axis_title if a_name == "x" else y_axis_title
                            axis_t = axis_t or ("x" if a_name == "x" else "y")
                            _show_overlay(f"{axis_t}: {value}")
                        except Exception:
                            pass

                    def _on_leave(ev):
                        _hide_overlay()

                    try:
                        lbl.on_hover = lambda ev: _on_enter(ev)
                        # some Flet versions may expose on_unhover or similar; guard it
                        try:
                            lbl.on_unhover = lambda ev: _on_leave(ev)
                        except Exception:
                            pass
                    except Exception:
                        pass

                # wrap in ChartAxisLabel if available
                try:
                    out.append(ChartAxisLabel(value=v, label=lbl))
                except Exception:
                    out.append(lbl)
            except Exception:
                pass
        return out

    # --------------------------------------------------
    # Helper: make a LineChartData (with tooltip support)
    # --------------------------------------------------
    def _make_series(pts, gtype):
        data_points = []
        for p in pts:
            try:
                x, y = p
            except Exception:
                continue
            # build tooltip text
            try:
                xt = x_axis_title or "x"
                yt = y_axis_title or "y"
                tooltip = f"{xt}: {x}\n{yt}: {y}"
            except Exception:
                tooltip = f"{x}, {y}"
            try:
                dp = LineChartDataPoint(x, y, tooltip=tooltip)
            except Exception:
                # older versions may not accept tooltip kwarg
                try:
                    dp = LineChartDataPoint(x, y)
                except Exception:
                    continue
            data_points.append(dp)
        try:
            return LineChartData(
                data_points=data_points,
                stroke_width=6 if gtype == "line" else 2,
                curved=False,
                color=ORANGE,
            )
        except Exception:
            return None

    # --------------------------------------------------
    # Centralized chart builder used both initially and on rebuild
    # --------------------------------------------------
    def _build_chart(gtype, pts):
        # compute min/max safely
        xs = [p[0] for p in pts] if pts else []
        ys = [p[1] for p in pts] if pts else []
        min_x, max_x = (min(xs), max(xs)) if xs else (0, 1)
        min_y, max_y = (min(ys), max(ys)) if ys else (0, 1)

        # PIE
        if gtype == "pie" and PieChart and PieChartSection:
            sections = []
            for label, value in pts:
                try:
                    v = float(value)
                except Exception:
                    continue
                try:
                    xt = x_axis_title or "label"
                    yt = y_axis_title or "value"
                    tooltip = f"{xt}: {label}\n{yt}: {v}"
                except Exception:
                    tooltip = f"{label}: {v}"
                try:
                    sections.append(PieChartSection(value=v, title=str(label), color=ORANGE, tooltip=tooltip))
                except Exception:
                    # fallback if tooltip not supported
                    sections.append(PieChartSection(value=v, title=str(label), color=ORANGE))
            return PieChart(sections=sections, sections_space=2, center_space_radius=50, expand=True) if sections else ft.Text("No data", size=12, color=FG)

        # LINE
        if LineChart and LineChartData and LineChartDataPoint and gtype == "line":
            series = _make_series(pts, gtype)
            if series is None:
                return ft.Text("Chart not supported", size=12, color=FG)
            return LineChart(
                data_series=[series],
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                left_axis=ChartAxis(labels=_make_axis_labels([min_y, max_y], "y")) if ChartAxis else None,
                bottom_axis=ChartAxis(labels=_make_axis_labels([min_x, max_x], "x")) if ChartAxis else None,
                expand=True,
            )

        # BAR
        if gtype == "bar" and BarChart and BarChartGroup and BarChartRod:
            groups = []
            max_y_v = 0.0
            for idx, (label, value) in enumerate(pts):
                try:
                    v = float(value)
                except Exception:
                    continue
                max_y_v = max(max_y_v, v)
                # tooltip text
                try:
                    xt = x_axis_title or "label"
                    yt = y_axis_title or "value"
                    tooltip = f"{xt}: {label}\n{yt}: {v}"
                except Exception:
                    tooltip = f"{label}: {v}"
                # create rod and set tooltip where possible
                try:
                    rod = BarChartRod(from_y=0, to_y=v, width=18, color=ORANGE, border_radius=4)
                    try:
                        setattr(rod, "tooltip", tooltip)
                    except Exception:
                        pass
                except Exception:
                    try:
                        rod = BarChartRod(from_y=0, to_y=v, width=18, color=ORANGE)
                    except Exception:
                        rod = None
                groups.append(BarChartGroup(x=idx, bar_rods=[rod] if rod is not None else []))
            return (
                BarChart(
                    bar_groups=groups,
                    min_y=0,
                    max_y=(max_y_v * 1.1) if max_y_v else 1,
                    left_axis=ChartAxis(labels=_make_axis_labels([0, max_y_v], "y")) if ChartAxis else None,
                    bottom_axis=ChartAxis(labels=[ChartAxisLabel(value=i, label=ft.Text(str(pts[i][0]), size=10)) for i in range(len(pts))]) if ChartAxis else None,
                    expand=True,
                )
                if groups
                else ft.Text("No data", size=12, color=FG)
            )

        # fallback
        return ft.Text("Chart not supported", size=12, color=FG)

    # --------------------------------------------------
    # initial chart build (uses centralized builder)
    # --------------------------------------------------
    try:
        content_chart = _build_chart(graph_type, points)
    except Exception:
        content_chart = ft.Text("Chart error", size=12, color=FG)

    # --------------------------------------------------
    # header + optional toggle
    # --------------------------------------------------
    toggle_btn = None
    if include_toggle:
        toggle_btn = ft.Container(
            content=ft.Text("Switch", size=12, color=FG),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border=ft.border.all(1, ORANGE),
            border_radius=int(tabs_script.s(8, None)),
            on_click=lambda e: _switch_graph(None),
        )

    title_ctrl = ft.Text(title, weight="bold", size=12, color=FG) if title else None
    header_controls = ([title_ctrl] if title_ctrl else []) + [ft.Container(expand=True)]
    if toggle_btn:
        header_controls.append(toggle_btn)
    header_row = ft.Row(controls=header_controls, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=0)

    inner_col = ft.Column(controls=[header_row, content_chart, overlay_container], spacing=6, expand=True)

    container = ft.Container(
        content=inner_col,
        padding=ft.padding.all(int(tabs_script.s(12, None))),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(12, None)),
        bgcolor=BG,
        width=width,
        height=height,
    )

    # --------------------------------------------------
    # in-place re-render helpers (use same builder)
    # --------------------------------------------------
    def _set_data(new_data):
        nonlocal points
        points = list(new_data or [])
        try:
            new_chart = _build_chart(graph_type, points)
            inner_col.controls[1] = new_chart
            # refresh any helper refs (container._graph_funcs preserved)
            try:
                if hasattr(container, "update"):
                    container.update()
            except Exception:
                pass
        except Exception:
            print("DEBUG: graph._set_data failed:", traceback.format_exc())

    def _switch_graph(new_type):
        nonlocal graph_type
        graph_type = new_type or _cycle_map.get(graph_type, "line")
        _set_data(points)

    container._graph_funcs = {"set_data": _set_data, "switch_graph": _switch_graph}

    return container



def graph_switch(
    data=None,
    title="Subscriber Growth",
    width=None,
    height=int(tabs_script.sv(220, None)),
    x_axis_title="Time",
    y_axis_title="Value",
    initial_type="line",
):
    """
    Container wrapper that:
    - hosts graph() inside it
    - places the switch button INSIDE the widget (top-right)
    - does NOT modify graph() internals
    - respects width / height
    """

    # build graph normally (NO toggle inside graph)
    inner_graph = graph(
        data=data,
        title=title,
        graph_type=initial_type,
        width=None,
        height=None,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        include_toggle=False,
    )


    def _on_switch(e):

        try:
            funcs = getattr(inner_graph, "_graph_funcs", None)
            if funcs and "switch_graph" in funcs:
                funcs["switch_graph"](None)

        except Exception as ex:
            print("DEBUG: graph_switch -> switch failed:", ex, traceback.format_exc())

        if e.page:
            e.page.update()


    switch_btn = ft.Container(
        content=ft.Text("Switch", size=12, color=FG),
        padding=ft.padding.symmetric(horizontal=10, vertical=6),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(8, None)),
        on_click=_on_switch,
    )

    # overlay switch inside widget (top-right)
    overlay = ft.Stack(
        controls=[
            inner_graph,
            ft.Container(
                content=switch_btn,
                alignment=ft.alignment.top_right,
                padding=ft.padding.all(8),
            ),
        ],
        expand=True,
    )

    # single stable outer container
    wrapper = ft.Container(
        content=overlay,
        width=width,
        height=height,
        padding=0,
        bgcolor=None,
    )

    # expose graph funcs at wrapper level (important)
    wrapper._graph_funcs = getattr(inner_graph, "_graph_funcs", {})

    return wrapper