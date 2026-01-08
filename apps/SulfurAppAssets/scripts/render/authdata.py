"""
sulfur_youtube_gemini.py

Provides:
- fetch_channel_metrics(token_data, metrics, start_date, end_date, dimension='day')
    -> returns dict of metric -> {'points': [(index, value), ...], 'x_labels': [date_str,...], 'raw_rows': [...]}

- metrics_to_graph_points(metrics_payload, metric_name)
    -> convenience adapter returning [(x,y), ...] for a single metric suitable for graph.py

- chat_gemini(prompt, system_message=None, api_key=None, model='gemini-2.5-flash',
              temperature=None, top_p=None, top_k=None)
    -> calls Gemini 2.5 Flash (via google.genai SDK if available) and returns the raw response object.

Notes:
- token_data should be the dict or string that SulfurApp provides via tabs_script.restore_tokens_from_keyring,
  i.e. a dict containing "provider_token" (the Google OAuth token from Supabase) or a bare access token string.
"""

from typing import List, Dict, Any, Optional, Tuple
import requests
import os
import datetime
import json

# ---------------------------
# Helpers: token normalization
# ---------------------------
def _get_access_token(token_data: Any) -> str:
    """
    Normalize token_data to an OAuth access token string.

    Accepts:
      - dict with "provider_token" (Supabase Google OAuth token) - PREFERRED
      - dict with "access_token" (fallback for compatibility)
      - raw string access_token

    Raises ValueError on invalid input.
    """
    if token_data is None:
        raise ValueError("token_data is None")

    # If it's already a string, assume it's the access token
    if isinstance(token_data, str):
        return token_data

    if isinstance(token_data, dict):
        # PRIMARY: Look for provider_token (Supabase stores Google OAuth token here)
        if "provider_token" in token_data and token_data["provider_token"]:
            return token_data["provider_token"]

        # FALLBACK: Try other common token keys
        for key in ("access_token", "token", "accessToken"):
            if key in token_data and token_data[key]:
                return token_data[key]

        # Also check nested forms
        if "credentials" in token_data and isinstance(token_data["credentials"], dict):
            cred = token_data["credentials"]
            if "provider_token" in cred:
                return cred["provider_token"]
            if "access_token" in cred:
                return cred["access_token"]

    raise ValueError(
        "Could not extract access token from token_data. "
        "Expected str or dict with 'provider_token' (Supabase) or 'access_token'. "
        f"Received type: {type(token_data)}, keys: {token_data.keys() if isinstance(token_data, dict) else 'N/A'}"
    )

# ---------------------------
# YouTube Analytics function
# ---------------------------
def fetch_channel_metrics(
    token_data: Any,
    metrics: List[str],
    start_date: str,
    end_date: str,
    dimension: str = "day",
    timezone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetches one or more metrics from the YouTube Analytics API for the authenticated channel.

    Returns per-metric dict with:
      - "points": [(1, value), (2, value), ...]         # numeric-indexed points (line-friendly)
      - "label_points": [("YYYY-MM-DD", value), ...]     # label-first points (bar/pie-friendly)
      - "x_labels": ["YYYY-MM-DD", ...]                  # labels matching the points order
      - "raw_rows": rows                                 # original response rows

    Use:
      payload = fetch_channel_metrics(...)
      line_points = payload["views"]["points"]
      bar_points  = payload["views"]["label_points"]
    """
    access_token = _get_access_token(token_data)
    url = "https://youtubeanalytics.googleapis.com/v2/reports"

    if not metrics or not isinstance(metrics, (list, tuple)):
        raise ValueError("metrics must be a non-empty list of metric names")

    metrics_str = ",".join(metrics)

    params = {
        "ids": "channel==MINE",
        "startDate": start_date,
        "endDate": end_date,
        "metrics": metrics_str
    }
    if dimension:
        params["dimensions"] = dimension
    if timezone:
        params["timezone"] = timezone

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        msg = f"YouTube Analytics API request failed: {e}\nResponse: {resp.status_code} {resp.text}"
        raise RuntimeError(msg)

    data = resp.json()
    col_headers = [h.get("name") for h in data.get("columnHeaders", [])] or []
    rows = data.get("rows", []) or []

    header_to_index = {name: idx for idx, name in enumerate(col_headers)}
    dimension_col = col_headers[0] if col_headers else None

    result: Dict[str, Any] = {}
    for m in metrics:
        result[m] = {"points": [], "label_points": [], "x_labels": [], "raw_rows": rows}

    if not rows:
        # no data: return empty structure with zeroed series length 0
        return result

    # Build x_labels from first column (dimension values)
    x_labels = [str(r[0]) for r in rows]

    # For each metric, locate column index and build both series
    for m in metrics:
        if m not in header_to_index:
            # attempt fuzzy match
            matched = None
            for h in header_to_index:
                if h.endswith(m) or m.endswith(h) or (h.lower() == m.lower()):
                    matched = h
                    break
            if matched:
                idx = header_to_index[matched]
            else:
                # no column: produce zero series
                points_zero = [(i + 1, 0.0) for i in range(len(rows))]
                label_points_zero = [(x_labels[i], 0.0) for i in range(len(rows))]
                result[m]["points"] = points_zero
                result[m]["label_points"] = label_points_zero
                result[m]["x_labels"] = x_labels
                continue
        else:
            idx = header_to_index[m]

        numeric_points = []
        labelled_points = []
        for i, row in enumerate(rows):
            try:
                raw_val = row[idx]
            except Exception:
                raw_val = 0
            try:
                val = float(raw_val)
            except Exception:
                try:
                    val = float(str(raw_val).replace(",", "").strip())
                except Exception:
                    val = 0.0
            numeric_points.append((i + 1, val))              # numeric x for line charts
            labelled_points.append((x_labels[i], val))       # (label, value) for bar/pie

        result[m]["points"] = numeric_points
        result[m]["label_points"] = labelled_points
        result[m]["x_labels"] = x_labels
        result[m]["raw_rows"] = rows

    return result

# ---------------------------
# Small adapter to return a single-metric points list for graph.py
# ---------------------------
def metrics_to_graph_points(metrics_payload: Dict[str, Any], metric_name: str) -> List[Tuple[float, float]]:
    """
    Convert the metrics_payload returned by fetch_channel_metrics into a simple
    list of (x, y) tuples for a single metric_name.

    Example:
      metrics_payload = fetch_channel_metrics(...)
      points = metrics_to_graph_points(metrics_payload, "views")
      # points -> [(1, 123.0), (2, 200.0), ...]

    If metric not found -> returns empty list.
    """
    if not metrics_payload or metric_name not in metrics_payload:
        return []
    return metrics_payload[metric_name].get("points", [])

# ---------------------------
# Optional helper: aggregate daily points into weekly groups
# ---------------------------
def aggregate_points_weekly(points: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Accepts points [(1,val), (2,val), ...] where index increments daily.
    Returns weekly aggregated points: every 7 days summed into one point.
    """
    if not points:
        return []
    out = []
    current_sum = 0.0
    current_count = 0
    week_index = 1
    for i, (idx, val) in enumerate(points, start=1):
        current_sum += val
        current_count += 1
        if current_count == 7:
            out.append((week_index, current_sum))
            week_index += 1
            current_sum = 0.0
            current_count = 0
    # leftover days -> include as final partial week
    if current_count > 0:
        out.append((week_index, current_sum))
    return out

# ---------------------------
# Gemini 2.5 Flash chat function
# ---------------------------
def chat_gemini(
    prompt: str,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_raw: bool = True,
) -> Any:
    """
    Chat with Gemini 2.5 Flash.

    Uses the `google.genai` SDK if available. If the SDK is not installed or cannot be used,
    this function raises a RuntimeError with guidance.

    Args:
      prompt: user text prompt (required)
      system_message: optional system instruction (higher priority)
      api_key: optional API key; if provided the function will set GOOGLE_API_KEY env var before client init
      model: model name (default 'gemini-2.5-flash')
      temperature, top_p, top_k: generation control parameters (optional)
      return_raw: if True, return SDK response object; if False, return textual content (best-effort)

    Returns:
      Raw SDK response object (preferred) or text (if return_raw=False).

    Example usage:
      resp = chat_gemini("Summarise my analytics", system_message="You are a helpful assistant.", api_key=os.environ.get('GOOGLE_API_KEY'))
    """
    if api_key:
        # many SDKs read GOOGLE_API_KEY or similar env var — set it for convenience
        os.environ["GOOGLE_API_KEY"] = api_key

    # Try using google.genai SDK
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            "google.genai SDK not available or failed to import. "
            "Install the official SDK (pip install google-genai) and authenticate (set GOOGLE_API_KEY or use application default credentials). "
            f"Import error: {e}"
        )

    try:
        client = genai.Client()  # reads env var or credentials as configured

        # build contents according to SDK's expected types
        contents = []
        if system_message:
            # types.Content and types.Part allow structured role+parts
            try:
                contents.append(types.Content(role="system", parts=[types.Part.from_text(system_message)]))
            except Exception:
                # fallback to using dict shape which some SDK versions accept
                contents.append({"role": "system", "text": system_message})

        # user message
        try:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(prompt)]))
        except Exception:
            contents.append({"role": "user", "text": prompt})

        # build generation config (sdk-specific class)
        config = types.GenerateContentConfig()
        if temperature is not None:
            try:
                config.temperature = float(temperature)
            except Exception:
                pass
        if top_p is not None:
            try:
                config.top_p = float(top_p)
            except Exception:
                pass
        if top_k is not None:
            try:
                config.top_k = int(top_k)
            except Exception:
                pass

        # Call model
        # Note: some SDK versions accept `model=` and `contents=` while others use different signatures.
        response = client.models.generate_content(model=model, contents=contents, config=config)

        if return_raw:
            return response
        else:
            # extract plain text if possible
            try:
                # many SDK responses expose `.text` or `.candidates[0].content[0].text`
                if hasattr(response, "text"):
                    return response.text
                # try nested structure
                resp_dict = json.loads(response.json()) if hasattr(response, "json") else None
            except Exception:
                resp_dict = None

            # fallback: attempt to stringify
            try:
                return str(response)
            except Exception:
                return response

    except Exception as e:
        raise RuntimeError(f"Failed to call Gemini via google.genai: {e}")