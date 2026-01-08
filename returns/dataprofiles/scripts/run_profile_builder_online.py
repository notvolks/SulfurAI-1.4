# run_profile_builder_online_patched.py
# Patched single-file version with robust OpenAI and Gemini generator normalization.
from typing import Optional, Callable
import os

def generate_gemini_text(
        prompt: str,
        model: str,
        api_key: str,
        system_prompt=None,
        n: int = 1,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> dict:
    """
    Generates using Gemini (via google-genai SDK) using the same input signatures
    as run_profile_builder. Prints debug info and returns the raw dict response.
    """
    import os
    import json
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize GenAI client -> {e}")
        return {"error": str(e)}

    # Build config
    try:
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
    except Exception as e:
        print(f"WARNING: Couldn’t build config with types.GenerateContentConfig -> {e}")
        config = None

    # Build contents list
    if system_prompt is not None:
        try:
            sp_str = json.dumps(system_prompt, ensure_ascii=False, indent=2)
        except Exception:
            sp_str = str(system_prompt)
        contents = [sp_str, prompt]
    else:
        contents = [prompt]


    for idx, c in enumerate(contents):
        preview = repr(c[:200]) if isinstance(c, str) else f"<{type(c).__name__}>"


    # Make API call
    try:
        if config:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        else:
            resp = client.models.generate_content(
                model=model,
                contents=contents
            )

    except Exception as e:
        print(f"ERROR: Exception during generate_content -> {e}")
        return {"error": str(e)}

    # Convert to dict (if supported)
    try:
        resp_dict = resp.to_json_dict()
    except Exception:
        try:
            resp_dict = resp.to_dict()
        except Exception:
            resp_dict = {"raw": str(resp)}
            print("DEBUG: Could not convert response object to dict, returning raw string.")

    # Build output dict
    result = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "raw_response": resp_dict
    }

    return result



