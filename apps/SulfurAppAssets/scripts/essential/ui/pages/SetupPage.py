from apps.SulfurAppAssets.scripts.verification import sulfuroauth
import os
import json
import flet as ft
from pathlib import Path
import shutil
import keyring
import secrets
import datetime
import webbrowser

base = Path(__file__).resolve().parents[6]
CACHE_BASE = Path(base / "apps" / "SulfurAppAssets" / "cache" / "tabs")

CACHE_BASE.mkdir(parents=True, exist_ok=True)
from apps.SulfurAppAssets.globalvar import global_var
from apps.SulfurAppAssets.scripts.essential.ui import tabs_script

BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()
def _get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()

call = _get_call_file_path()

class PageBase:
    def __init__(self, title: str):
        self.title = title

    def content(self):
        raise NotImplementedError

    # -----------------------------------------------------------------


from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPage import ChannelPage


class SetupPage(PageBase):
    def __init__(self, title, app_page=None, pages_dict=None, file_tabs=None, switch_func=None, tab_id=None,
                 platform_inner=None, logo_path=None):

        super().__init__(title)

        # references so this page can swap itself for ChannelPage after auth
        self.app_page = app_page
        self.pages_dict = pages_dict
        self.file_tabs = file_tabs
        self.switch_func = switch_func
        self.tab_id = tab_id
        self.logo_path = logo_path  # Path to local image file for branding

        # --- default: platform is selected ---
        self.platform_selected = True

        # status text
        self.oauth_status = ft.Text(
            "Not connected",
            color="#aaaaaa",
            size=12,
            text_align="center",
            visible=True
        )

        # API key field (styled to match theme)
        self.api_key_field = ft.TextField(
            label="Google API Key",
            hint_text="Enter your API key",
            width=int(tabs_script.sh(366, self.app_page)),
            password=True,
            can_reveal_password=True,
            on_change=self._on_api_key_change,
            border_color=ORANGE,
            focused_border_color=ORANGE,
            text_size=16,
            label_style=ft.TextStyle(color=FG),
            color=FG,
            height=56,
        )

        # Continue button (styled with orange accent)
        self.continue_button = ft.ElevatedButton(
            "Continue",
            on_click=self._oauth_click,
            style=ft.ButtonStyle(
                bgcolor=ORANGE,
                color="#ffffff",
                shape=ft.RoundedRectangleBorder(radius=4),
                padding=ft.padding.symmetric(horizontal=24, vertical=12),
            ),
            width=int(tabs_script.sh(366, self.app_page)),
            height=40,
            disabled=True,  # Disabled until API key is valid
        )

        self.platform_inner = ft.Container(
            content=ft.Text(
                "YouTube Channel",
                color=FG,
                size=int(tabs_script.s(16, self.app_page)),
                text_align="center"
            ),
            padding=ft.padding.symmetric(
                horizontal=int(tabs_script.sh(22, self.app_page)),
                vertical=int(tabs_script.sv(14, self.app_page))
            ),
            border=ft.border.all(2, ORANGE),
            border_radius=int(tabs_script.s(16, self.app_page)),
            bgcolor="#0f0f0f",
            alignment=ft.alignment.center,
            on_click=self._platform_click,
            width=int(tabs_script.sh(320, self.app_page)),
        )

    # ---------------- platform toggle ----------------
    def _platform_click(self, e):
        # toggle
        self.platform_selected = not self.platform_selected

        if self.platform_selected:
            self.platform_inner.border = ft.border.all(2, ORANGE)
            self.platform_inner.bgcolor = "#0f0f0f"
        else:
            self.platform_inner.border = ft.border.all(1, ORANGE)
            self.platform_inner.bgcolor = BG

        try:
            e.page.update()
        except:
            pass

    # ---------------- OAuth ----------------
    def _oauth_click(self, e):
        self.oauth_status.value = "Opening browser..."
        self.oauth_status.color = "#ffffaa"
        self.oauth_status.visible = True
        try:
            e.page.update()
        except:
            pass

        # Prepare a unique keyname to store tokens under
        key_name = f"auth_{secrets.token_urlsafe(6)}"

        # <-- pass key_name into sulfuroauth so it stores under a unique keyring entry -->
        success, info = sulfuroauth.supabase_desktop_oauth(key_name)

        if not success:
            self.oauth_status.value = f"OAuth failed: {info}"
            print(f"OAuth failed: {info}")
            self.oauth_status.color = "#ff8888"
            self.oauth_status.visible = True
            try:
                e.page.update()
            except:
                pass
            return

        token_data = info

        # attempt to get userinfo for friendly label
        sup_ok, sup_info = sulfuroauth.get_supabase_user_info(sulfuroauth.supabase_url, token_data.get("access_token"))
        if sup_ok:
            title = sup_info.get("email") or sup_info.get("id") or "Supabase user"
        else:
            title = "Supabase user"
            sup_info = {}

        # persist metadata into accounts index (including youtube scope flags and keyring name)
        metadata = {
            "channel_id": sup_info.get("id") or key_name,
            "title": title,
            "keyring_username": key_name,
            "youtube_scopes": ["youtube.readonly", "youtube.upload"],
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        try:
            sulfuroauth.add_or_update_account_index(metadata)
        except Exception:
            pass

        # update/ write the tab cache so the tab is restorable
        try:
            # securely store Google API key under a separate keyring entry (service: sulfur-google)
            api_key_name = f"sulfur-google-api-key-{self.tab_id}"
            try:
                # store the API key; do NOT print the key anywhere
                keyring.set_password("sulfur-google", api_key_name, (self.api_key_field.value or "").strip())
            except Exception:
                # if keyring storage fails, fall back to None but do not leak the key
                api_key_name = None

            # Extract and store refresh token separately
            refresh_token = token_data.get("provider_refresh_token")
            refresh_keyname = None
            if refresh_token:
                refresh_keyname = f"{key_name}_refresh"
                sulfuroauth._save_refresh_token(refresh_keyname, refresh_token)

            # Calculate access token expiry timestamp
            expires_in = token_data.get("expires_in", 3600)  # default 1 hour
            expiry_timestamp = datetime.datetime.utcnow().timestamp() + expires_in

            # prefer storing non-sensitive identifiers only
            write_state = {
                "page": "ChannelPage",
                "signed_in": True,
                "user_id": sup_info.get("id") or sup_info.get("sub"),
                "keyring_name": key_name,
                "token_expiry": expiry_timestamp,
            }
            if refresh_keyname:
                write_state["refresh_keyring_name"] = refresh_keyname
            if api_key_name:
                write_state["google_api_key_name"] = api_key_name

            tabs_script.write_tab_state(self.tab_id, write_state)
        except Exception:
            pass

        self.oauth_status.value = f"Connected as: {title}"
        self.oauth_status.color = "#aaffaa"  # Success green (matching old script)
        self.oauth_status.visible = True

        try:
            e.page.update()
        except:
            pass

        # --- Replace this tab's page with the Channel page and rename the tab ---
        try:
            if self.pages_dict is not None and self.file_tabs is not None and self.switch_func is not None and self.tab_id is not None:
                # create channel page and replace in pages dict
                channel_page = ChannelPage(
                    "User's Channel",
                    token_data=token_data,
                    user_info=sup_info,
                    app_page=self.app_page,
                    tab_id=self.tab_id,
                    keyring_name=key_name,
                    platform_inner=self.platform_inner
                )

                # replace in pages dict and ask the file tabs widget to replace button/Page
                try:
                    self.pages_dict[self.tab_id] = channel_page
                    # use new safe API to replace button & page
                    try:
                        self.file_tabs.replace_tab(self.tab_id, channel_page)
                    except Exception:
                        pass
                    # update the cache/state with new page type & title
                    try:
                        update_state = {
                            "page": "ChannelPage",
                            "signed_in": True,
                            "user_id": sup_info.get("id") or sup_info.get("sub"),
                            "keyring_name": key_name,
                            "title": channel_page.title,
                            "token_expiry": expiry_timestamp,
                        }
                        if refresh_keyname:
                            update_state["refresh_keyring_name"] = refresh_keyname
                        # include google_api_key_name if we saved it
                        try:
                            if 'api_key_name' in locals() and api_key_name:
                                update_state["google_api_key_name"] = api_key_name
                        except Exception:
                            pass

                        tabs_script.update_tab_state(self.tab_id, update_state)
                        tabs_script.register_tab_in_index(self.tab_id, title=channel_page.title,
                                                          page_type="ChannelPage")
                    except Exception:
                        pass

                    # switch to the same tab id to render the new page
                    self.switch_func(self.tab_id)
                    # change window/page title
                    try:
                        self.app_page.title = channel_page.title
                        self.app_page.update()
                    except Exception:
                        pass
                except Exception as e:
                    from scripts.ai_renderer_sentences.error import SulfurError
                    raise SulfurError(message=f"Error replacing tab after oauth: {e}")


        except Exception:
            # if replacement fails, ignore but tokens are stored
            pass

    def _on_api_key_change(self, e):
        """
        Basic format check: starts with 'AIza' and at least 35 characters.
        Enables the Continue button only when the check passes.
        """
        try:
            key_text = (self.api_key_field.value or "").strip()
        except Exception:
            key_text = ""

        valid = key_text.startswith("AIza") and len(key_text) >= 35

        try:
            self.continue_button.disabled = not valid
        except Exception:
            pass

        # refresh UI
        try:
            e.page.update()
        except Exception:
            pass

    @staticmethod
    def check_and_refresh_token(tab_id):
        """
        Check if the access token has expired and refresh it if needed.
        Call this method periodically or before making API requests.
        Returns True if token is valid/refreshed, False if refresh failed.
        """
        try:
            # Load tab state
            state = tabs_script.read_tab_state(tab_id)
            if not state:
                return False

            token_expiry = state.get("token_expiry")
            keyring_name = state.get("keyring_name")
            refresh_keyring_name = state.get("refresh_keyring_name")

            # If no refresh token, can't refresh
            if not refresh_keyring_name:
                return True  # Continue without exceptions

            # Check if token has expired (add 5 minute buffer)
            current_time = datetime.datetime.utcnow().timestamp()
            buffer_seconds = 300  # 5 minutes

            if token_expiry and current_time >= (token_expiry - buffer_seconds):
                # Token expired or about to expire, refresh it
                success, result = sulfuroauth.refresh_access_token(refresh_keyring_name, keyring_name)

                if success:
                    # Update the expiry time in cache
                    new_expiry = result
                    state["token_expiry"] = new_expiry
                    tabs_script.write_tab_state(tab_id, state)
                    return True
                else:
                    print(f"Token refresh failed for tab {tab_id}: {result}")
                    return False

            return True  # Token still valid

        except Exception as e:
            print(f"Error checking/refreshing token for tab {tab_id}: {e}")
            return False

    # ---------------- Save (Hop in) ----------------
    def _save_click(self, e):
        try:
            e.page.snack_bar = ft.SnackBar(
                ft.Text("Proceeding — Jumping in..."),
                bgcolor="#111"
            )
            e.page.snack_bar.open = True
            e.page.update()
        except:
            pass

    # ---------------- Build UI ----------------
    def content(self):

        EXTERNALAPP_sulfurapp_logo = call.EXTERNALAPP_sulfurapp_logo()
        logo_content = ft.Image(
            src=EXTERNALAPP_sulfurapp_logo,
            width=120,
            height=115,
            fit=ft.ImageFit.FILL,
        )


        # Google-style sign-in card
        sign_in_card = ft.Container(
            content=ft.Column(
                [
                    # Logo / Brand area (placeholder for image or text)
                    ft.Container(
                        content=logo_content,
                        padding=ft.padding.only(bottom=10),
                        alignment=ft.alignment.center,
                    ),



                    ft.Container(height=24),

                    # API Key explanation
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text(
                                    "Your Google API Key is required for accessing Google services.",
                                    color=FG,
                                    size=14,
                                    text_align="left",
                                ),
                                ft.Row(
                                    [
                                        ft.Text(
                                            "Get your API key from ",
                                            color=FG,
                                            size=14,
                                        ),
                                        ft.TextButton(
                                            "Google AI Studio",
                                            on_click=lambda e: webbrowser.open(
                                                "https://aistudio.google.com/app/api-keys"
                                            ),
                                            style=ft.ButtonStyle(
                                                color=ORANGE,
                                                padding=0,
                                            ),
                                        ),
                                    ],
                                    spacing=0,
                                ),
                            ],
                            spacing=4,
                        ),
                        padding=ft.padding.only(bottom=16),
                    ),

                    # API Key field
                    self.api_key_field,

                    ft.Container(height=8),

                    # Privacy disclaimer
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Text(
                                    "🔒",
                                    size=14,
                                ),
                                ft.Text(
                                    "Your API key is stored securely on your local device only",
                                    color=FG,
                                    size=12,
                                    italic=True,
                                ),
                            ],
                            spacing=6,
                        ),
                        padding=ft.padding.only(bottom=8),
                    ),

                    # Status message
                    self.oauth_status,

                    ft.Container(height=24),

                    # Continue button
                    ft.Container(
                        content=self.continue_button,
                        alignment=ft.alignment.center_right,
                    ),

                ],
                spacing=0,
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=ft.padding.all(40),
            bgcolor=BG,
            border=ft.border.all(1, ORANGE),
            border_radius=8,
            width=int(tabs_script.sh(450, self.app_page)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
        )

        return ft.Container(
            expand=True,
            bgcolor=BG,  # Original background color
            alignment=ft.alignment.center,
            content=ft.Column(
                [
                    sign_in_card
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
        )