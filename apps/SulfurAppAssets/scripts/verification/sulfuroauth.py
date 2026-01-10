# Google / YouTube imports (kept for compatibility where still used)

import datetime
import base64
import keyring
from dotenv import load_dotenv, set_key
import os, json
import requests
import secrets
import hashlib
from urllib.parse import urlencode, parse_qs, urlparse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

# load .env early
load_dotenv()

# --- Theme / constants ---
BG = "#121212"
FG = "#FFFFFF"
ORANGE = "#FFA64D"

ACCOUNTS_INDEX_PATH = os.path.expanduser("~/.sulfur/youtube_accounts.json")
# change service name to reflect Supabase
KEYRING_SERVICE = "sulfur-supabase"

# Supabase configuration (from environment)
supabase_url = os.getenv("SUPABASE_URL", "https://wnqmiengjapawpnjlrcp.supabase.co")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY",
                              "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InducW1pZW5namFwYXdwbmpscmNwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ0ODQ5NDQsImV4cCI6MjA4MDA2MDk0NH0.5hNvHBmbNbOICKWY5Ju1qi6gs-3fjI5rfOEag8RqR9U")
supabase_auth_base = supabase_url.rstrip("/") + "/auth/v1"
redirect_uri = os.getenv("SUPABASE_REDIRECT_URI", "http://127.0.0.1:8765/callback")

# scopes you want to request via Supabase -> Google
GOOGLE_SCOPES = os.getenv(
    "GOOGLE_SCOPES",
    "openid email profile https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/yt-analytics.readonly https://www.googleapis.com/auth/generative-language.retriever https://www.googleapis.com/auth/youtube.force-ssl https://www.googleapis.com/auth/cloud-platform"
).split()

ENV_FILE = ".env"
APP_NAME = "SulfurAI"

# ensure index dir exists
os.makedirs(os.path.dirname(ACCOUNTS_INDEX_PATH), exist_ok=True)


# ----------------- index/keyring helpers -----------------
def _save_tokens_safely(keyname: str, token_data: dict):
    from pathlib import Path
    try:
        keyring.set_password(KEYRING_SERVICE, keyname, json.dumps(token_data))
        return True, None
    except Exception:
        try:
            keyring.delete_password(KEYRING_SERVICE, keyname)
        except Exception:
            pass
        try:
            keyring.set_password(KEYRING_SERVICE, keyname, json.dumps(token_data))
            return True, None
        except Exception:
            pass
    try:
        creds_dir = Path(os.path.expanduser("~")) / ".sulfur" / "secure_credentials"
        creds_dir.mkdir(parents=True, exist_ok=True)
        file_path = creds_dir / f"{keyname}.json"
        tmp = file_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(token_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(file_path)
        try:
            file_path.chmod(0o600)
        except Exception:
            pass
        minimal = {
            "provider_token": token_data.get("provider_token"),
            "provider_refresh_token": token_data.get("provider_refresh_token"),
            "stored_in_file": str(file_path),
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z"
        }
        try:
            keyring.set_password(KEYRING_SERVICE, keyname, json.dumps(minimal))
        except Exception:
            pass
        return True, f"saved_to_file:{file_path}"
    except Exception as e:
        return False, f"save_failed: {e}"


def _save_refresh_token(refresh_keyname: str, refresh_token: str):
    """Save refresh token to a separate keyring entry."""
    try:
        keyring.set_password(KEYRING_SERVICE, refresh_keyname, refresh_token)
        return True
    except Exception:
        # Fallback to file storage
        try:
            from pathlib import Path
            creds_dir = Path(os.path.expanduser("~")) / ".sulfur" / "secure_credentials"
            creds_dir.mkdir(parents=True, exist_ok=True)
            file_path = creds_dir / f"{refresh_keyname}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(refresh_token)
                f.flush()
                os.fsync(f.fileno())
            try:
                file_path.chmod(0o600)
            except Exception:
                pass
            return True
        except Exception:
            return False


def _load_refresh_token(refresh_keyname: str):
    """Load refresh token from keyring or file."""
    try:
        token = keyring.get_password(KEYRING_SERVICE, refresh_keyname)
        if token:
            return token
    except Exception:
        pass

    # Try file fallback
    try:
        from pathlib import Path
        creds_dir = Path(os.path.expanduser("~")) / ".sulfur" / "secure_credentials"
        file_path = creds_dir / f"{refresh_keyname}.txt"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass

    return None


def refresh_access_token(refresh_keyname: str, access_keyname: str):
    """
    Use Supabase refresh token to refresh the session.
    Also manually refreshes Google provider token using provider_refresh_token.
    For Desktop app OAuth clients, only client_id is needed (no client_secret).
    Returns (True, new_expiry_timestamp) on success or (False, error_string) on failure.
    """
    print(f"[DEBUG] Starting token refresh for access_keyname: {access_keyname}, refresh_keyname: {refresh_keyname}")

    refresh_token = _load_refresh_token(refresh_keyname)
    if not refresh_token:
        print("[DEBUG] ERROR: No refresh token found")
        return False, "No refresh token found"

    print(f"[DEBUG] Loaded refresh token (first 20 chars): {refresh_token[:20]}...")

    try:
        # Load existing token data to get provider_refresh_token
        existing_data = {}
        try:
            print(f"[DEBUG] Attempting to load existing token data from keyring: {access_keyname}")
            existing_json = keyring.get_password(KEYRING_SERVICE, access_keyname)
            if existing_json:
                existing_data = json.loads(existing_json)
                print(f"[DEBUG] Loaded existing token data. Keys present: {list(existing_data.keys())}")
            else:
                print("[DEBUG] No existing token data found in keyring")
        except Exception as e:
            print(f"[DEBUG] Error loading existing token data: {e}")

        # Step 1: Refresh Supabase session
        print("[DEBUG] STEP 1: Refreshing Supabase session")
        token_url = f"{supabase_auth_base}/token?grant_type=refresh_token"
        headers = {
            "Content-Type": "application/json",
            "apikey": supabase_anon_key,
            "Authorization": f"Bearer {supabase_anon_key}",
        }
        body = {"refresh_token": refresh_token}

        print(f"[DEBUG] Supabase token URL: {token_url}")
        print(f"[DEBUG] Sending Supabase refresh request...")
        response = requests.post(token_url, json=body, headers=headers)

        print(f"[DEBUG] Supabase response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[DEBUG] ERROR: Supabase token refresh failed: {response.text}")
            return False, f"Token refresh failed: {response.status_code} {response.text}"

        new_token_data = response.json()
        print(f"[DEBUG] Supabase refresh successful. New token data keys: {list(new_token_data.keys())}")

        # Step 2: Refresh Google provider token manually (Supabase doesn't do this)
        provider_refresh_token = existing_data.get("provider_refresh_token")
        if provider_refresh_token:
            print("[DEBUG] STEP 2: Refreshing Google provider token via Supabase Edge Function")
            try:
                # Call Supabase Edge Function to refresh provider token securely
                # The client secret is stored on Supabase backend, not in your local code
                edge_function_url = f"{supabase_url}/functions/v1/refresh-provider-token"
                edge_function_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {supabase_anon_key}",
                }
                edge_function_body = {
                    "provider_refresh_token": provider_refresh_token
                }

                print(f"[DEBUG] Calling Supabase Edge Function: {edge_function_url}")
                edge_response = requests.post(edge_function_url, json=edge_function_body, headers=edge_function_headers)

                print(f"[DEBUG] Edge Function response status: {edge_response.status_code}")
                if edge_response.status_code == 200:
                    edge_data = edge_response.json()
                    print(f"[DEBUG] ✓ Provider token refreshed via Edge Function!")
                    new_provider_token = edge_data.get("provider_token")
                    if new_provider_token:
                        print(f"[DEBUG] New provider_token received (first 20 chars): {new_provider_token[:20]}...")
                        new_token_data["provider_token"] = new_provider_token
                        new_token_data["provider_refresh_token"] = provider_refresh_token
                        print("[DEBUG] Provider tokens updated in new_token_data")
                    else:
                        print("[DEBUG] WARNING: No provider_token in Edge Function response")
                        new_token_data["provider_token"] = existing_data.get("provider_token")
                        new_token_data["provider_refresh_token"] = provider_refresh_token
                else:
                    print(f"[DEBUG] ERROR: Edge Function failed: {edge_response.text}")
                    print("[DEBUG] This may mean:")
                    print("[DEBUG]   → The Edge Function is not deployed")
                    print("[DEBUG]   → The GOOGLE_CLIENT_SECRET is not set in Supabase")
                    print("[DEBUG]   → The provider_refresh_token is invalid")
                    # Fallback to old tokens
                    new_token_data["provider_token"] = existing_data.get("provider_token")
                    new_token_data["provider_refresh_token"] = provider_refresh_token
                    print("[DEBUG] Falling back to old provider tokens")
            except Exception as e:
                print(f"[DEBUG] ERROR: Exception calling Edge Function: {type(e).__name__}: {e}")
                new_token_data["provider_token"] = existing_data.get("provider_token")
                new_token_data["provider_refresh_token"] = provider_refresh_token
                print("[DEBUG] Falling back to old provider tokens due to exception")
        else:
            print("[DEBUG] STEP 2 SKIPPED: No provider_refresh_token found in existing data")
            # Preserve old provider tokens if they exist
            if "provider_token" in existing_data:
                new_token_data["provider_token"] = existing_data.get("provider_token")
                print("[DEBUG] Preserving old provider_token from existing data")
            if "provider_refresh_token" in existing_data:
                new_token_data["provider_refresh_token"] = existing_data.get("provider_refresh_token")
                print("[DEBUG] Preserving old provider_refresh_token from existing data")

        # Step 3: Save refreshed token data
        print(f"[DEBUG] STEP 3: Saving refreshed token data to keyring: {access_keyname}")
        print(f"[DEBUG] Token data keys being saved: {list(new_token_data.keys())}")
        ok, note = _save_tokens_safely(access_keyname, new_token_data)
        if not ok:
            print(f"[DEBUG] ERROR: Failed to save tokens: {note}")
            return False, f"Failed to save refreshed token: {note}"
        print(f"[DEBUG] Tokens saved successfully. Note: {note}")

        # Calculate new expiry time (expires_in is in seconds)
        expires_in = new_token_data.get("expires_in", 3600)
        expiry_timestamp = datetime.datetime.utcnow().timestamp() + expires_in
        print(f"[DEBUG] Calculated new expiry timestamp: {expiry_timestamp} (expires_in: {expires_in}s)")

        # Update Supabase refresh token if a new one was provided (token rotation)
        new_refresh = new_token_data.get("refresh_token")
        if new_refresh:
            print(f"[DEBUG] New Supabase refresh_token received (first 20 chars): {new_refresh[:20]}...")
            print(f"[DEBUG] Saving new refresh token to: {refresh_keyname}")
            _save_refresh_token(refresh_keyname, new_refresh)
            print("[DEBUG] New refresh token saved")
        else:
            print("[DEBUG] No new refresh_token in response (token rotation not performed)")

        print(f"[DEBUG] ✓ Token refresh completed successfully. New expiry: {expiry_timestamp}")
        return True, expiry_timestamp

    except Exception as e:
        print(f"[DEBUG] ERROR: Token refresh failed with exception: {type(e).__name__}: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False, f"Token refresh error: {e}"


def load_accounts_index():
    if not os.path.exists(ACCOUNTS_INDEX_PATH):
        return []
    try:
        with open(ACCOUNTS_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_accounts_index(accounts):
    with open(ACCOUNTS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)


def add_or_update_account_index(entry):
    accounts = load_accounts_index()
    found = False
    for i, e in enumerate(accounts):
        if e.get("keyring_username") == entry.get("keyring_username"):
            accounts[i] = entry
            found = True
            break
    if not found:
        accounts.append(entry)
    save_accounts_index(accounts)


def remove_account_index_by_keyname(keyname):
    accounts = load_accounts_index()
    accounts = [a for a in accounts if a.get("keyring_username") != keyname]
    save_accounts_index(accounts)


# ----------------- PKCE helpers -----------------
def _base64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> (str, str):
    """Return (code_verifier, code_challenge)"""
    code_verifier = _base64url_encode(secrets.token_bytes(32))
    m = hashlib.sha256()
    m.update(code_verifier.encode("ascii"))
    code_challenge = _base64url_encode(m.digest())
    return code_verifier, code_challenge


class _AuthCallbackHandler(BaseHTTPRequestHandler):
    """Temporary HTTP handler to capture the OAuth redirect (code & state)."""

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        # Attach data to server instance for retrieval
        self.server.auth_code = qs.get("code", [None])[0]
        self.server.auth_error = qs.get("error", [None])[0]
        self.server.state = qs.get("state", [None])[0]

        # Respond with a simple HTML page
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h3>You may close this window and return to the app.</h3></body></html>")

    def log_message(self, format, *args):
        # quiet logging
        return


# ----------------- Supabase OAuth flow (desktop PKCE) -----------------
def supabase_desktop_oauth(key_name: str = "default"):
    """
    PKCE desktop OAuth via Supabase (Google). Returns (True, token_data) or (False, error_str).
    Debug prints are verbose so paste their output here if it still fails.
    """
    try:
        # return cached if present
        # REMOVED these lines:
        # existing = keyring.get_password(KEYRING_SERVICE, key_name)
        # if existing:
        #     try:
        #         tok = json.loads(existing)
        #         return True, tok
        #     except Exception:
        #         pass

        supabase_auth_base = supabase_url.rstrip('/') + '/auth/v1'
        code_verifier, code_challenge = generate_pkce_pair()
        scopes_str = " ".join(GOOGLE_SCOPES)

        # Build authorize URL; do NOT supply state (let Supabase handle it).
        params = {
            "provider": "google",
            "redirect_to": redirect_uri,
            "scopes": scopes_str,
            "access_type": "offline",
            "prompt": "consent",
            "response_type": "code",  # <-- request an authorization code
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{supabase_auth_base}/authorize?{urlencode(params)}"

        # DEBUG: print the authorize URL so you can inspect/copy it
        print("OPENING AUTHORIZE URL:\n", auth_url)

        webbrowser.open(auth_url)

        # start local HTTP server to capture callback
        parsed = urlparse(redirect_uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80

        server = HTTPServer((host, port), _AuthCallbackHandler)
        server.timeout = 300
        print("Waiting for OAuth callback on", host, port)
        while getattr(server, "auth_code", None) is None and getattr(server, "auth_error", None) is None:
            server.handle_request()

        # Debug: show what the handler received
        print("CALLBACK PATH RECEIVED:", getattr(server, "auth_code", None), getattr(server, "auth_error", None),
              getattr(server, "state", None))

        if getattr(server, "auth_error", None):
            return False, f"Authorization error: {server.auth_error}"

        code = getattr(server, "auth_code", None)
        if not code:
            return False, "No authorization code received"

        token_url = f"{supabase_auth_base}/token?grant_type=pkce"
        token_body = {
            "auth_code": code,  # auth_code (not 'code')
            "code_verifier": code_verifier,
        }

        headers = {
            "Content-Type": "application/json",
            "apikey": supabase_anon_key,
            "Authorization": f"Bearer {supabase_anon_key}",
        }

        print("TOKEN EXCHANGE (pkce/json) -> URL:", token_url)
        print("TOKEN EXCHANGE (pkce/json) -> HEADERS:",
              {k: ("<redacted>" if k in ("apikey", "Authorization") else v) for k, v in headers.items()})
        print("TOKEN EXCHANGE (pkce/json) -> BODY:", token_body)

        token_resp = requests.post(token_url, json=token_body, headers=headers)

        print("TOKEN EXCHANGE RESPONSE STATUS:", token_resp.status_code)
        print("TOKEN EXCHANGE RESPONSE BODY:", token_resp.text)

        if token_resp.status_code != 200:
            return False, f"Token exchange failed: {token_resp.status_code} {token_resp.text}"

        token_data = token_resp.json()

        # Save token/session JSON to keyring
        ok, note = _save_tokens_safely(key_name, token_data)
        if not ok:
            return False, f"OAuth succeeded but storing tokens failed: {note}"
        return True, token_data

    except Exception as e:
        return False, f"OAuth failed: {type(e).__name__}: {e}"


def get_supabase_user_info(supabase_url, access_token: str):
    """
    Return (True, user_json) on success or (False, error_string) on failure.
    """
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(f"{supabase_url.rstrip('/')}/auth/v1/user", headers=headers)
        if r.status_code == 200:
            return True, r.json()
        return False, f"Userinfo failed: {r.status_code} {r.text}"
    except Exception as e:
        return False, f"Userinfo error: {e}"