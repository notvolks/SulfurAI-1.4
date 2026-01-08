from apps.SulfurAppAssets.scripts.verification import sulfuroauth
import os
import json
import flet as ft
from pathlib import Path
import shutil
import keyring
base = Path(__file__).resolve().parents[5]
CACHE_BASE = Path(base / "apps" / "SulfurAppAssets" / "cache" / "tabs")
CACHE_BASE.mkdir(parents=True, exist_ok=True)
from apps.SulfurAppAssets.globalvar import global_var
BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()



# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       ESSENTIAL SCRIPTS                                                                                      |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

TABS_INDEX_PATH = CACHE_BASE / "tabs_index.json"


def _load_tabs_index():
    try:
        if not TABS_INDEX_PATH.exists():
            return {"order": [], "meta": {}}
        with open(TABS_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Failed loading tabs_index:", e)
        return {"order": [], "meta": {}}


def _save_tabs_index(index_obj):
    try:
        with open(TABS_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(index_obj, f, indent=2)
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"Failed saving tabs_index: {e}")


def register_tab_in_index(tab_id: str, title: str = None, page_type: str = None):
    idx = _load_tabs_index()
    if tab_id not in idx["order"]:
        idx["order"].append(tab_id)
    if title or page_type:
        meta = idx.get("meta", {})
        ent = meta.get(tab_id, {})
        if title:
            ent["title"] = title
        if page_type:
            ent["page"] = page_type
        meta[tab_id] = ent
        idx["meta"] = meta
    _save_tabs_index(idx)


def unregister_tab_from_index(tab_id: str):
    idx = _load_tabs_index()
    if tab_id in idx.get("order", []):
        idx["order"].remove(tab_id)
    meta = idx.get("meta", {})
    if tab_id in meta:
        del meta[tab_id]
        idx["meta"] = meta
    _save_tabs_index(idx)


def set_last_active_tab(tab_id: str):
    idx = _load_tabs_index()
    idx["last_active_tab"] = tab_id
    _save_tabs_index(idx)


def get_last_active_tab() -> str | None:
    idx = _load_tabs_index()
    return idx.get("last_active_tab")


def _tab_folder_for(tab_id: str) -> Path:
    return CACHE_BASE / tab_id


def write_tab_state(tab_id: str, state: dict):
    folder = _tab_folder_for(tab_id)
    folder.mkdir(parents=True, exist_ok=True)
    try:

        title = state.get("title")
        page_type = state.get("page") or state.get("page_type")
        if title or page_type:
            register_tab_in_index(tab_id, title=title, page_type=page_type)
        with open(folder / "tab_state.txt", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"write_tab_state error: {e}")


def read_tab_state(tab_id: str) -> dict | None:
    folder = _tab_folder_for(tab_id)
    fpath = folder / "tab_state.txt"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def update_tab_state(tab_id: str, patch: dict):
    """
    Read existing state for tab_id, apply patch (shallow merge), and write back.
    This keeps existing keys (like keyring_name) while updating inner_tab or inner_data.
    """
    try:
        state = read_tab_state(tab_id) or {}
        state.update(patch or {})
        write_tab_state(tab_id, state)
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"update_tab_state error: {e}")


def delete_tab_cache(tab_id: str):
    folder = _tab_folder_for(tab_id)
    if folder.exists():
        try:
            shutil.rmtree(folder)
        except Exception as e:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"delete_tab_cache error: {e}")

    try:
        unregister_tab_from_index(tab_id)
    except Exception as e:

        from scripts.ai_renderer_sentences.error import SulfurError
        raise SulfurError(message=f"unregister_tab_from_index error: {e}")


def restore_tokens_from_keyring(keyring_name: str) -> dict | None:
    """
    Returns the JSON object stored in keyring for the provided key name, or None.
    (This only reads; it does not write anything to disk.)
    """
    try:
        raw = keyring.get_password(sulfuroauth.KEYRING_SERVICE, keyring_name)
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def compute_ui_scale(page):
    """
    Compute a single uniform scale factor for the whole UI:
      scale = min(window_width / base_width, window_height / base_height)

    This uses the page.width / page.height values that your code already relies on,
    stores helpful derived values on page.data and returns the uniform scale.
    """
    try:
        if page is None:
            return 1.0

        w = float(page.width or UI_BASE_WIDTH)
        h = float(page.height or UI_BASE_HEIGHT)

        sx = w / UI_BASE_WIDTH
        sy = h / UI_BASE_HEIGHT

        scale = min(sx, sy)

        lock_min_scale = min(
            UI_LOCK_MIN_WIDTH / UI_BASE_WIDTH,
            UI_LOCK_MIN_HEIGHT / UI_BASE_HEIGHT
        )
        scale = max(scale, lock_min_scale)
        scale = max(scale, UI_MIN_SCALE)
        scale = min(scale, UI_MAX_SCALE)

        page.data["sx"] = sx
        page.data["sy"] = sy
        page.data["win_width"] = w
        page.data["win_height"] = h
        page.data["scale"] = scale

        return scale

    except Exception:
        return 1.0


def get_scale_from_page(app_page):
    try:
        return float(app_page.data.get("scale", 1.0))
    except Exception:
        return 1.0


def s(value, app_page=None):
    """
    Uniform scale for sizes/fonts: always uses the single computed scale factor.
    Use everywhere for widths, heights and font sizes.
    """
    if not isinstance(value, (int, float)):
        return value
    scale = get_scale_from_page(app_page) if app_page else 1.0
    return value * scale


def sh(value, app_page=None):
    if not isinstance(value, (int, float)):
        return value
    return s(value, app_page)


def sv(value, app_page=None):
    if not isinstance(value, (int, float)):
        return value
    return s(value, app_page)