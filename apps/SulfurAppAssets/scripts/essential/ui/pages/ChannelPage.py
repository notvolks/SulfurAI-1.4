import SulfurAI
from apps.SulfurAppAssets.scripts.verification import sulfuroauth
import os
import json
import flet as ft
from flet.plotly_chart import PlotlyChart
from pathlib import Path
import shutil
import keyring
import threading
import datetime
import re
import traceback
import requests
from apps.SulfurAppAssets.scripts.render.authdata import fetch_channel_metrics, _get_access_token
from apps.SulfurAppAssets.scripts.ai import predictor_template
from apps.SulfurAppAssets.scripts.ai import assistant
from dateutil import parser

youtube_data = None
base = Path(__file__).resolve().parents[6]
CACHE_BASE = Path(base / "apps" / "SulfurAppAssets" / "cache" / "tabs")
CACHE_BASE.mkdir(parents=True, exist_ok=True)
from apps.SulfurAppAssets.globalvar import global_var
from apps.SulfurAppAssets.scripts.essential.ui import tabs_script

DARK_GREY = "#0D0D0D"

BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()

# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       UI TEMPLATES                                                                                     |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


# general ui templates
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.uitemplates import _size_s
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.uitemplates import create_scrollable_panel
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.uitemplates import create_text_block
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.uitemplates import make_dashboard_lines

# advanced
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.base_template import base_template
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.button_template import button_template
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.chatbot_template import chatbot_template
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.graph import graph
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.graph import graph_switch
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.tab_template import tab_template_simple
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.tab_template import tab_template_advanced
from apps.SulfurAppAssets.scripts.essential.ui.pages.ChannelPageTemplates.switch_template import switch_template

def _get_call_file_path():
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()

call = _get_call_file_path()
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |                                                                                                                                                                        |
# |                                                                       UI CODE                                                                                     |
# |                                                                                                                                                                        |
# |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

class PageBase:
    def __init__(self, title: str):
        self.title = title

    def content(self):
        raise NotImplementedError

        # -----------------------------------------------------------------


class ChannelPage(PageBase):
    """
    ChannelPage with per-tab inner selection caching and per-inner-tab text fields.
    """

    def __init__(self, title, token_data, user_info, app_page=None, tab_id=None, keyring_name=None,
                 platform_inner=None):
        super().__init__(title)
        self.token_data = token_data or {}
        self.user_info = user_info or {}
        self.app_page = app_page
        self.tab_id = tab_id
        self.keyring_name = keyring_name

        self.inner_selected = None
        self.inner_data = {}
        self._restored = False

        # references for sidebar and inner pages
        self.sidebar_buttons: dict[str, ft.Container] = {}
        self.sidebar_search = ""
        self.inner_pages = {}
        self.platform_inner = platform_inner

        # Selected video state for Videos page
        self.selected_video = None
        self.selected_video_stats = {}
        self.selected_video_display = None  # Container to update when video is selected
        self.ai_explanation_container = None
        self.latest_video_growth_score_text = None  # Reference to growth score Text widget
        self.latest_video_data = None

        self.main_container = ft.Container(expand=True, bgcolor=BG,
                                           padding=ft.padding.all(int(tabs_script.s(12, self.app_page))))

        # dashboard graph widget refs
        self.dashboard_graph_widget = None
        self.dashboard_graph_funcs = None
        self.chatbot_host_controls = {}
        self.chatbot_context = {}
        self.chat_attach_mode = False

        # AI Assistant for chatbot
        self.gemini_api_key = None
        self.ai_processing = False
        self._load_gemini_api_key()

        from apps.SulfurAppAssets.scripts.essential.ui.pages.SetupPage import SetupPage
        if SetupPage.check_and_refresh_token(self.tab_id):
            print("✓ Token refreshed successfully on ChannelPage init")
        else:
            print("Token Refresh Failed")

    # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    # |                                                                                                                                                                        |
    # |                                                                       UI                                                                                     |
    # |                                                                                                                                                                        |
    # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

    def _on_chat_attach_click(self, e=None):
        """
        Show the upload video popup when attach button is clicked in chatbot.
        Solution: Navigate to Videos page and trigger the button there.
        """
        print("DEBUG: _on_chat_attach_click called - switching to Videos page")

        # Set attach mode FIRST so when video is clicked, it knows to attach to chat
        self.chat_attach_mode = True

        # Simply navigate to the Videos page - this makes the button visible and clickable
        self._on_inner_click("Videos", e)

        # Now that we're on Videos page, trigger the upload button
        # The button is now visible and in the correct layout context
        try:
            button = getattr(self, 'upload_a_video_view', None)
            if button and hasattr(button, 'on_click') and callable(button.on_click):
                print("DEBUG: Triggering upload_a_video_view.on_click()")
                # Small delay to let page render, then trigger
                import threading
                def trigger_button():
                    try:
                        import time
                        time.sleep(0.1)  # Brief delay for page to render
                        button.on_click(e)
                        if self.app_page:
                            self.app_page.update()
                        print("DEBUG: Button triggered successfully")
                    except Exception as ex:
                        print(f"DEBUG: Error triggering button: {ex}")

                threading.Thread(target=trigger_button, daemon=True).start()
            else:
                print("DEBUG: upload_a_video_view button not found or not clickable")
        except Exception as ex:
            print(f"ERROR: Failed to trigger upload button: {ex}")
            import traceback
            traceback.print_exc()

    def _load_gemini_api_key(self):
        """Load Google API key from keyring using tab_id."""
        try:
            if not self.tab_id:
                print("ℹ No tab_id - cannot load API key")
                self.gemini_api_key = None
                return

            # The API key is stored as: keyring.set_password("sulfur-google", f"sulfur-google-api-key-{tab_id}", api_key)
            api_key_name = f"sulfur-google-api-key-{self.tab_id}"
            self.gemini_api_key = keyring.get_password("sulfur-google", api_key_name)

            if self.gemini_api_key:
                print(f"✓ Google API key loaded for chatbot (key: {api_key_name})")
            else:
                print(f"ℹ No Google API key found at: {api_key_name}")
        except Exception as e:
            print(f"Could not load Google API key: {e}")
            self.gemini_api_key = None

    def _save_gemini_api_key(self, api_key):
        """Save Gemini API key to keyring."""
        try:
            if api_key and api_key.strip():
                keyring.set_password("sulfurai_gemini", "api_key", api_key.strip())
                self.gemini_api_key = api_key.strip()
                print("✓ Gemini API key saved")
                return True
            else:
                try:
                    keyring.delete_password("sulfurai_gemini", "api_key")
                except:
                    pass
                self.gemini_api_key = None
                print("ℹ Gemini API key removed")
                return True
        except Exception as e:
            print(f"Error saving API key: {e}")
            return False

    def _build_chatbot_context(self):
        """Build context dictionary for AI from current state."""
        context = {}

        # Add chatbot context (from video attachment)
        if hasattr(self, 'chatbot_context') and self.chatbot_context:
            # This includes video, stats, stage2, stage3 from predictor
            context.update(self.chatbot_context)

        # Add selected video if not already in chatbot_context
        if hasattr(self, 'selected_video') and self.selected_video:
            if "video" not in context:
                context["video"] = self.selected_video

        # Add video stats if not already in chatbot_context
        if hasattr(self, 'selected_video_stats') and self.selected_video_stats:
            if "stats" not in context:
                context["stats"] = self.selected_video_stats

        return context if context else None

    def _on_generate_reply(self, user_message):
        """
        Generate AI reply for chatbot. This is called by chatbot_template.

        Args:
            user_message: The user's message (without "User: " prefix)

        Returns:
            str: The bot's reply text (without "Bot: " prefix)
        """
        if not self.gemini_api_key:
            print("No Gemini API key - using default response")
            return "Hi! I'm your YouTube growth assistant. To enable AI responses, add your Gemini API key in Settings."

        if self.ai_processing:
            return "Please wait, I'm still processing your previous message..."

        self.ai_processing = True
        try:
            # Build context
            context = self._build_chatbot_context()

            # Log what we're sending
            print(f"\n[AI REQUEST]")
            print(f"  Message: {user_message[:100]}...")
            if context:
                print(f"  Context keys: {list(context.keys())}")
            else:
                print(f"  Context: None")

            # Generate reply
            result = assistant.generate_reply(
                user_message=user_message,
                context=context,
                api_key=self.gemini_api_key,
                temperature=0.7,
                max_tokens=2048
            )

            if result["success"]:
                print(f"[AI RESPONSE] Generated {len(result['response'])} chars")
                return result['response']
            else:
                print(f"[AI ERROR] {result.get('error')}")
                return f"I encountered an error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            print(f"Error in AI generation: {e}")
            traceback.print_exc()
            return f"I encountered an error: {str(e)}"

        finally:
            self.ai_processing = False

    def graph_set_data(self, new_data):
        try:
            funcs = getattr(self, "dashboard_graph_funcs", None)
            if funcs and "set_data" in funcs:
                funcs["set_data"](new_data)
        except Exception:
            pass

    def graph_switch_type(self, new_type):
        try:
            funcs = getattr(self, "dashboard_graph_funcs", None)
            if funcs and "switch_graph" in funcs:
                funcs["switch_graph"](new_type)
        except Exception:
            pass

    def graph_zoom(self, x0, x1):
        try:
            funcs = getattr(self, "dashboard_graph_funcs", None)
            if funcs and "zoom" in funcs:
                funcs["zoom"](x0, x1)
        except Exception:
            pass

    def graph_add_point(self, x, y):
        try:
            funcs = getattr(self, "dashboard_graph_funcs", None)
            if funcs and "add_point" in funcs:
                funcs["add_point"](x, y)
        except Exception:
            pass

    def _build_alert_card(self, text: str):
        """Create a single alert UI card."""
        return base_template(
            height=int(tabs_script.sv(90, self.app_page)),
            width=int(tabs_script.sh(580, self.app_page)),
            page=self.app_page,
            children=[
                ft.Text(
                    text,
                    color=FG,
                    size=int(tabs_script.s(14, self.app_page)),
                )
            ],
        )

    def fetch_comprehensive_channel_data(self, days_back=90, max_videos=50, max_comments_per_video=5):
        """
        Fetch comprehensive YouTube channel data using keyring_name and authdata.py.

        This method fetches:
        - Subscriber growth data (graph-ready)
        - Revenue data (graph-ready)
        - Views data (graph-ready)
        - Watch time data (graph-ready)
        - Latest video with thumbnail, stats, and comments
        - All past videos and shorts
        - Aggregated channel statistics

        Args:
            days_back: Number of days of historical analytics (default 90)
            max_videos: Maximum number of videos to fetch (default 50)
            max_comments_per_video: Max comments per video (default 5)

        Returns:
            Dictionary containing:
            {
                'subscriber_growth': {'points': [...], 'label_points': [...], 'x_labels': [...]},
                'revenue': {'points': [...], 'label_points': [...], 'x_labels': [...]},
                'views': {'points': [...], 'label_points': [...], 'x_labels': [...]},
                'watch_time': {'points': [...], 'label_points': [...], 'x_labels': [...]},
                'latest_video': {'video_id': '...', 'title': '...', 'thumbnail': '...', ...},
                'all_videos': [{...}, {...}, ...],
                'stats': {'total_subscribers': ..., 'total_videos': ..., ...}
            }

        Example:
            data = self.fetch_comprehensive_channel_data(days_back=30)
            self.graph_set_data(data['subscriber_growth']['points'])
        """
        import datetime

        # Validate keyring_name
        if not self.keyring_name:
            raise ValueError("No keyring_name available. Cannot fetch channel data.")

        # Retrieve token data from keyring
        try:
            token_data = tabs_script.restore_tokens_from_keyring(self.keyring_name)
            if not token_data:
                raise ValueError(f"No token found for keyring_name: {self.keyring_name}")
        except Exception as e:
            raise ValueError(f"Failed to retrieve tokens: {e}")

        provider = token_data.get("provider_token")
        if isinstance(provider, dict):
            access_token = provider.get("access_token")
        elif isinstance(provider, str):
            access_token = provider
        else:
            access_token = None

        if not access_token:
            raise RuntimeError("No access_token found in provider_token; cannot call YouTube Data API")

        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

        # Calculate date range
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_back)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        result = {
            'fetched_at': datetime.datetime.now().isoformat() + 'Z',
            'date_range': {'start': start_date_str, 'end': end_date_str}
        }

        # Fetch analytics data using authdata.py
        analytics_metrics = {}
        revenue_available = False

        try:
            # First, try to fetch with revenue metric (requires monetization)
            try:
                analytics_metrics = fetch_channel_metrics(
                    token_data=access_token,
                    metrics=['subscribersGained', 'subscribersLost', 'estimatedRevenue',
                             'views', 'estimatedMinutesWatched'],
                    start_date=start_date_str,
                    end_date=end_date_str,
                    dimension='day'
                )
                revenue_available = True
                print("✓ Successfully fetched analytics including revenue data")
            except Exception as revenue_error:
                # If revenue fails (channel not monetized or insufficient permission), try without it
                error_str = str(revenue_error).lower()
                if "insufficient permission" in error_str or "401" in error_str or "403" in error_str:
                    print(f"⚠ Revenue data not available (channel may not be monetized), fetching other analytics...")
                    analytics_metrics = fetch_channel_metrics(
                        token_data=access_token,
                        metrics=['subscribersGained', 'subscribersLost',
                                 'views', 'estimatedMinutesWatched'],
                        start_date=start_date_str,
                        end_date=end_date_str,
                        dimension='day'
                    )
                    revenue_available = False
                    print("✓ Successfully fetched analytics without revenue data")
                else:
                    # Re-raise if it's a different error
                    raise revenue_error

            # Calculate net subscriber growth
            gained = analytics_metrics.get('subscribersGained', {})
            lost = analytics_metrics.get('subscribersLost', {})

            if gained.get('points') and lost.get('points'):
                net_points = []
                net_label_points = []

                for (idx, gained_val), (_, lost_val) in zip(gained['points'], lost['points']):
                    net_points.append((idx, gained_val - lost_val))

                for (label, gained_val), (_, lost_val) in zip(gained['label_points'], lost['label_points']):
                    net_label_points.append((label, gained_val - lost_val))

                result['subscriber_growth'] = {
                    'points': net_points,
                    'label_points': net_label_points,
                    'x_labels': gained.get('x_labels', []),
                    'raw_data': {'gained': gained, 'lost': lost},
                    'available': True
                }
            else:
                result['subscriber_growth'] = {
                    'points': [],
                    'label_points': [],
                    'x_labels': [],
                    'raw_data': {},
                    'available': False
                }

            # Revenue data
            if revenue_available and analytics_metrics.get('estimatedRevenue'):
                revenue_data = analytics_metrics.get('estimatedRevenue', {})
                # Check if revenue data is actually present (not just zeros)
                has_revenue = any(val > 0 for _, val in revenue_data.get('points', []))
                result['revenue'] = {
                    'points': revenue_data.get('points', []),
                    'label_points': revenue_data.get('label_points', []),
                    'x_labels': revenue_data.get('x_labels', []),
                    'raw_data': revenue_data.get('raw_rows', []),
                    'available': True,
                    'has_data': has_revenue
                }
            else:
                result['revenue'] = {
                    'points': [],
                    'label_points': [],
                    'x_labels': [],
                    'raw_data': [],
                    'available': False,
                    'has_data': False
                }

            # Views data
            views_data = analytics_metrics.get('views', {})
            result['views'] = {
                'points': views_data.get('points', []),
                'label_points': views_data.get('label_points', []),
                'x_labels': views_data.get('x_labels', []),
                'raw_data': views_data.get('raw_rows', []),
                'available': bool(views_data.get('points'))
            }

            watch_minutes = analytics_metrics.get('estimatedMinutesWatched', {})
            if watch_minutes.get('points'):
                result['watch_time'] = {
                    'points': [(idx, val / 60.0) for idx, val in watch_minutes['points']],
                    'label_points': [(label, val / 60.0) for label, val in watch_minutes['label_points']],
                    'x_labels': watch_minutes.get('x_labels', []),
                    'raw_data': watch_minutes.get('raw_rows', []),
                    'available': True
                }
            else:
                result['watch_time'] = {
                    'points': [],
                    'label_points': [],
                    'x_labels': [],
                    'raw_data': [],
                    'available': False
                }


        except Exception as e:

            print(f"Warning: Failed to fetch analytics: {e}")

            result['subscriber_growth'] = {

                'points': [],

                'label_points': [],

                'x_labels': [],

                'raw_data': {},

                'available': False

            }

            result['revenue'] = {

                'points': [],

                'label_points': [],

                'x_labels': [],

                'raw_data': [],

                'available': False,

                'has_data': False

            }

            result['views'] = {

                'points': [],

                'label_points': [],

                'x_labels': [],

                'raw_data': [],

                'available': False

            }

            result['watch_time'] = {

                'points': [],

                'label_points': [],

                'x_labels': [],

                'raw_data': [],

                'available': False

            }

        # Fetch channel information
        try:
            channel_resp = requests.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "statistics,contentDetails", "mine": "true"},
                headers=headers,
                timeout=30
            )
            channel_resp.raise_for_status()
            channel_data = channel_resp.json()

            if not channel_data.get('items'):
                raise RuntimeError("No channel found")

            channel_item = channel_data['items'][0]
            channel_stats = channel_item.get('statistics', {})
            uploads_playlist_id = channel_item.get('contentDetails', {}).get('relatedPlaylists', {}).get('uploads')

            result['stats'] = {
                'total_subscribers': int(channel_stats.get('subscriberCount', 0)),
                'total_videos': int(channel_stats.get('videoCount', 0)),
                'total_views': int(channel_stats.get('viewCount', 0)),
                'total_shorts': 0,
                'average_views_per_video': 0
            }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch channel info: {e}")

        # Fetch videos
        all_videos = []
        try:
            if uploads_playlist_id:
                next_page_token = None
                videos_fetched = 0

                while videos_fetched < max_videos:
                    playlist_params = {
                        "part": "snippet,contentDetails",
                        "playlistId": uploads_playlist_id,
                        "maxResults": min(50, max_videos - videos_fetched)
                    }
                    if next_page_token:
                        playlist_params["pageToken"] = next_page_token

                    playlist_resp = requests.get(
                        "https://www.googleapis.com/youtube/v3/playlistItems",
                        params=playlist_params,
                        headers=headers,
                        timeout=30
                    )
                    playlist_resp.raise_for_status()
                    playlist_data = playlist_resp.json()

                    video_ids = [item['contentDetails']['videoId'] for item in playlist_data.get('items', [])]
                    if not video_ids:
                        break

                    videos_resp = requests.get(
                        "https://www.googleapis.com/youtube/v3/videos",
                        params={"part": "snippet,statistics,contentDetails", "id": ",".join(video_ids)},
                        headers=headers,
                        timeout=30
                    )
                    videos_resp.raise_for_status()
                    videos_data = videos_resp.json()

                    for video in videos_data.get('items', []):
                        snippet = video.get('snippet', {})
                        statistics = video.get('statistics', {})
                        content_details = video.get('contentDetails', {})
                        duration = content_details.get('duration', 'PT0S')

                        # Check if Short (< 60 seconds)
                        def is_short(dur):
                            try:
                                match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
                                if not match:
                                    return False
                                h = int(match.group(1) or 0)
                                m = int(match.group(2) or 0)
                                s = int(match.group(3) or 0)
                                return (h * 3600 + m * 60 + s) < 60
                            except:
                                return False

                        all_videos.append({
                            'video_id': video['id'],
                            'title': snippet.get('title', ''),
                            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url',
                                                                                           snippet.get('thumbnails',
                                                                                                       {}).get(
                                                                                               'default', {}).get('url',
                                                                                                                  '')),
                            'view_count': int(statistics.get('viewCount', 0)),
                            'like_count': int(statistics.get('likeCount', 0)),
                            'comment_count': int(statistics.get('commentCount', 0)),
                            'published_at': snippet.get('publishedAt', ''),
                            'duration': duration,
                            'is_short': is_short(duration),
                            'description': snippet.get('description', '')[:200]
                        })
                        videos_fetched += 1

                    next_page_token = playlist_data.get('nextPageToken')
                    if not next_page_token or videos_fetched >= max_videos:
                        break

            result['all_videos'] = all_videos
            result['stats']['total_shorts'] = sum(1 for v in all_videos if v['is_short'])
            if all_videos:
                result['stats']['average_views_per_video'] = int(
                    sum(v['view_count'] for v in all_videos) / len(all_videos))
        except Exception as e:
            print(f"Warning: Failed to fetch videos: {e}")
            result['all_videos'] = []

        # Get latest video with comments
        result['latest_video'] = None
        if all_videos:
            sorted_videos = sorted(all_videos, key=lambda v: v['published_at'], reverse=True)
            latest = sorted_videos[0].copy()

            try:
                try:
                    token_info_resp = requests.get(
                        f"https://oauth2.googleapis.com/tokeninfo?access_token={access_token}"
                    )
                    print("Token scopes:", token_info_resp.json().get("scope"))
                except Exception as e:
                    print(f"Could not verify token: {e}")
                comments_resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    params={
                        "part": "snippet",
                        "videoId": latest['video_id'],
                        "maxResults": max_comments_per_video,
                        "order": "time"
                    },
                    headers=headers,
                    timeout=30
                )
                comments_resp.raise_for_status()
                comments_data = comments_resp.json()

                latest['comments'] = []
                for item in comments_data.get('items', []):
                    snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                    latest['comments'].append({
                        'author': snippet.get('authorDisplayName', 'Unknown'),
                        'text': snippet.get('textDisplay', ''),
                        'like_count': int(snippet.get('likeCount', 0)),
                        'published_at': snippet.get('publishedAt', '')
                    })

            except requests.HTTPError as e:
                if e.response.status_code == 403:
                    print(f"⚠ Comments require 'youtube.readonly' scope. Please re-authenticate with read permissions.")
                    latest['comments'] = []
                elif e.response.status_code == 404:
                    print(f"⚠ Comments disabled for this video")
                    latest['comments'] = []
                else:
                    print(f"⚠ Comments API error: {e}")
                    latest['comments'] = []

            except Exception as e:
                print(f"⚠ Failed to fetch comments: {e}")
                latest['comments'] = []

            result['latest_video'] = latest

        return result

    def _load_video_stats_for_ai(self, video_data, youtube_data=None):
        """
        Load comprehensive video statistics for AI prediction.

        Args:
            video_data: Dictionary containing basic video information
            youtube_data: Optional full channel data for additional context

        Returns:
            Dictionary containing all stats accessible via current scopes that an AI
            would need to predict video growth, including:
            - video_age_days: Age of the video in days
            - comment_count: Total number of comments
            - comments: List of comment texts and metadata
            - like_count: Number of likes
            - view_count: Current view count
            - views_growth_rate: Daily average views (views / days_old)
            - retention_proxy: Views per day vs channel average
            - subscribers_context: Channel subscriber count for context
            - video_duration: Duration in seconds
            - is_short: Boolean indicating if it's a short
            - publish_date: ISO 8601 publish timestamp
            - title_length: Length of video title
            - description_length: Length of video description
            - thumbnail_url: URL to video thumbnail
            - channel_average_views: Channel's average views per video
            - relative_performance: This video's views / channel average
            - days_since_publish: Exact days since publication
        """
        import datetime
        from dateutil import parser

        stats = {}

        # Basic video information
        stats['video_id'] = video_data.get('video_id', '')
        stats['title'] = video_data.get('title', '')
        stats['title_length'] = len(video_data.get('title', ''))
        stats['description'] = video_data.get('description', '')
        stats['description_length'] = len(video_data.get('description', ''))
        stats['thumbnail_url'] = video_data.get('thumbnail', '')

        # Publication date and age
        publish_date_str = video_data.get('published_at', '')
        stats['publish_date'] = publish_date_str

        if publish_date_str:
            try:
                publish_date = parser.parse(publish_date_str)
                now = datetime.datetime.now(datetime.timezone.utc)
                age_delta = now - publish_date
                stats['video_age_days'] = age_delta.days
                stats['video_age_hours'] = age_delta.total_seconds() / 3600
                stats['days_since_publish'] = age_delta.days + (age_delta.seconds / 86400)
            except Exception as e:
                print(f"Error parsing date: {e}")
                stats['video_age_days'] = 0
                stats['video_age_hours'] = 0
                stats['days_since_publish'] = 0
        else:
            stats['video_age_days'] = 0
            stats['video_age_hours'] = 0
            stats['days_since_publish'] = 0

        # Duration and format
        duration_str = video_data.get('duration', 'PT0S')
        stats['duration_iso'] = duration_str
        stats['is_short'] = video_data.get('is_short', False)

        # Parse duration to seconds
        try:
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
            if match:
                h = int(match.group(1) or 0)
                m = int(match.group(2) or 0)
                s = int(match.group(3) or 0)
                stats['video_duration_seconds'] = h * 3600 + m * 60 + s
                stats['video_duration_minutes'] = stats['video_duration_seconds'] / 60
            else:
                stats['video_duration_seconds'] = 0
                stats['video_duration_minutes'] = 0
        except Exception:
            stats['video_duration_seconds'] = 0
            stats['video_duration_minutes'] = 0

        # Engagement metrics
        stats['view_count'] = video_data.get('view_count', 0)
        stats['like_count'] = video_data.get('like_count', 0)
        stats['comment_count'] = video_data.get('comment_count', 0)

        # Calculate engagement rates
        if stats['view_count'] > 0:
            stats['like_rate'] = stats['like_count'] / stats['view_count']
            stats['comment_rate'] = stats['comment_count'] / stats['view_count']
        else:
            stats['like_rate'] = 0
            stats['comment_rate'] = 0

        # Growth metrics
        if stats['days_since_publish'] > 0:
            stats['views_per_day'] = stats['view_count'] / stats['days_since_publish']
            stats['views_growth_rate'] = stats['views_per_day']
        else:
            stats['views_per_day'] = 0
            stats['views_growth_rate'] = 0

        # Channel context from youtube_data
        if youtube_data:
            channel_stats = youtube_data.get('stats', {})
            stats['channel_total_subscribers'] = channel_stats.get('total_subscribers', 0)
            stats['channel_total_videos'] = channel_stats.get('total_videos', 0)
            stats['channel_total_views'] = channel_stats.get('total_views', 0)
            stats['channel_average_views'] = channel_stats.get('average_views_per_video', 0)

            # Performance relative to channel average
            if stats['channel_average_views'] > 0:
                stats['relative_performance'] = stats['view_count'] / stats['channel_average_views']
                stats['performance_percentile'] = min(100, stats['relative_performance'] * 50)
            else:
                stats['relative_performance'] = 0
                stats['performance_percentile'] = 0

            # Retention proxy (views per day vs channel average per day)
            if stats['channel_average_views'] > 0 and stats['days_since_publish'] > 0:
                expected_daily_views = stats['channel_average_views'] / 30  # Assume 30 day window
                stats['retention_proxy'] = stats[
                                               'views_per_day'] / expected_daily_views if expected_daily_views > 0 else 0
            else:
                stats['retention_proxy'] = 0
        else:
            stats['channel_total_subscribers'] = 0
            stats['channel_total_videos'] = 0
            stats['channel_total_views'] = 0
            stats['channel_average_views'] = 0
            stats['relative_performance'] = 0
            stats['performance_percentile'] = 0
            stats['retention_proxy'] = 0

        # Comments data
        stats['comments'] = []

        # Try to fetch comments if not already present
        if 'comments' in video_data and video_data['comments']:
            stats['comments'] = video_data['comments']
        else:
            # Try to fetch comments using the API
            try:
                if self.keyring_name:
                    token_data = tabs_script.restore_tokens_from_keyring(self.keyring_name)
                    provider = token_data.get("provider_token")
                    if isinstance(provider, dict):
                        access_token = provider.get("access_token")
                    elif isinstance(provider, str):
                        access_token = provider
                    else:
                        access_token = None

                    if access_token:
                        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
                        comments_resp = requests.get(
                            "https://www.googleapis.com/youtube/v3/commentThreads",
                            params={
                                "part": "snippet",
                                "videoId": stats['video_id'],
                                "maxResults": 20,
                                "order": "relevance"
                            },
                            headers=headers,
                            timeout=30
                        )

                        if comments_resp.status_code == 200:
                            comments_data = comments_resp.json()
                            for item in comments_data.get('items', []):
                                snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                                stats['comments'].append({
                                    'author': snippet.get('authorDisplayName', 'Unknown'),
                                    'text': snippet.get('textDisplay', ''),
                                    'like_count': int(snippet.get('likeCount', 0)),
                                    'published_at': snippet.get('publishedAt', '')
                                })
            except Exception as e:
                print(f"Could not fetch comments: {e}")

        # Comment sentiment proxy (likes on comments)
        if stats['comments']:
            stats['average_comment_likes'] = sum(c.get('like_count', 0) for c in stats['comments']) / len(
                stats['comments'])
            stats['comment_texts'] = [c.get('text', '')[:200] for c in stats['comments']]
        else:
            stats['average_comment_likes'] = 0
            stats['comment_texts'] = []

        # Subscribers gained proxy (not directly available, but we can estimate from growth)
        # This would require video-level analytics which needs different scopes
        stats['subscribers_gained_estimate'] = "Not available - requires YouTube Analytics API with video-level metrics"

        # Additional useful metrics for AI
        stats['video_maturity'] = 'new' if stats['days_since_publish'] < 7 else 'established' if stats[
                                                                                                     'days_since_publish'] < 30 else 'old'
        stats['engagement_score'] = (stats['like_rate'] * 100 + stats['comment_rate'] * 1000) if stats[
                                                                                                     'view_count'] > 0 else 0

        # Velocity metrics (useful for predicting growth)
        if stats['video_age_hours'] > 0:
            stats['views_per_hour'] = stats['view_count'] / stats['video_age_hours']
            stats['initial_velocity'] = stats['views_per_hour'] if stats['video_age_hours'] < 24 else 0
        else:
            stats['views_per_hour'] = 0
            stats['initial_velocity'] = 0

        return stats

    def _extract_api_key_from_keyring_data(self, keyring_name: str):
        """
        Try to restore tokens from keyring name and extract a usable API key string.
        Common shapes:
         - token_data is a plain string: use it as the key
         - token_data is dict with 'provider_token' or 'access_token' or 'api_key' fields
        Returns: api_key (str) or None if not found.
        """
        try:
            if not keyring_name:
                return None
            token_data = tabs_script.restore_tokens_from_keyring(keyring_name)
            if not token_data:
                return None

            # If token_data is a plain string (some keyrings store the key directly)
            if isinstance(token_data, str):
                return token_data

            # If token_data is a dict, check common locations
            if isinstance(token_data, dict):
                # common nested shapes used elsewhere in this codebase
                provider = token_data.get("provider_token") or token_data.get("token") or token_data.get("value")
                if isinstance(provider, str):
                    return provider
                if isinstance(provider, dict):
                    # sometimes inside provider_token: { 'access_token': '...' } or { 'api_key': '...' }
                    for candidate in ("access_token", "api_key", "apiKey", "key", "value"):
                        if provider.get(candidate):
                            return provider.get(candidate)
                # other direct keys
                for candidate in ("api_key", "access_token", "key", "value"):
                    if token_data.get(candidate):
                        return token_data.get(candidate)

            # nothing found
            return None
        except Exception:
            return None

    def _on_video_click(self, video_data, youtube_data=None):
        self.selected_video = video_data
        self.selected_video_stats = self._load_video_stats_for_ai(video_data, youtube_data)
        self._update_selected_video_display()

        # -------------------------
        # CHAT ATTACH MODE
        # -------------------------
        if self.chat_attach_mode:
            print("DEBUG: Video selected in chat attach mode")
            ctx = predictor_template.Context(self)

            ctx.stage_1(video_data.get("video_id"), None)
            stage2 = ctx.stage_2()
            stage3 = ctx.stage_3()

            self.chatbot_context = {
                "video": video_data,
                "stats": self.selected_video_stats,
                "stage2": stage2,
                "stage3": stage3,
            }

            # update chat thumbnail
            setter = self.chatbot_host_controls.get("set_thumbnail")
            if setter:
                setter(video_data.get("thumbnail"))

            # Close the popup if it exists
            if hasattr(self, 'upload_video_view') and hasattr(self.upload_video_view, 'data'):
                close_func = self.upload_video_view.data.get('close_popup') if isinstance(self.upload_video_view.data,
                                                                                          dict) else None
                if callable(close_func):
                    try:
                        print("DEBUG: Closing popup via upload_video_view.data['close_popup']")
                        close_func(None)
                    except Exception as ex:
                        print(f"DEBUG: Error closing popup: {ex}")

            self.chat_attach_mode = False

            # Navigate back to Chatbot page
            print("DEBUG: Navigating back to Chatbot page")
            self._on_inner_click("Chatbot", None)

            return

        # -------------------------
        # NORMAL FLOW (unchanged)
        # -------------------------
        predictor_thread = threading.Thread(
            target=self._run_predictor_for_selected_video,
            daemon=True
        )
        predictor_thread.start()

    def _update_selected_video_display(self):
        """Update the display area with the selected video's information"""
        if not self.selected_video_display:
            return

        try:
            stats = self.selected_video_stats

            # Create basic text displays
            children = [
                ft.Text("Selected Video", color=FG, size=14),
                ft.Text(f"Title: {stats.get('title', 'Unknown')[:50]}", color=FG, size=12),
                ft.Text(f"Views: {stats.get('view_count', 0):,}", color=FG, size=12),
                ft.Text(f"Age: {stats.get('video_age_days', 0)} days", color=FG, size=12),
                ft.Text(f"Growth: {stats.get('views_per_day', 0):.1f} views/day", color=FG, size=12),
            ]

            # Update the base_template children
            if hasattr(self.selected_video_display, 'content') and hasattr(self.selected_video_display.content,
                                                                           'controls'):
                self.selected_video_display.content.controls = children

            if self.app_page:
                self.app_page.update()

        except Exception as e:
            print(f"Error updating display: {e}")

    def _run_predictor_for_selected_video(self):
        """Run predictor_template.Context stages 1-4 for the currently selected video,
        update the dashboard graph (overwrite placeholder), and update AI explanation widget."""
        try:
            if not self.selected_video:
                return

            # Determine channel_id: try pulling from cached channel data if present
            channel_id = None
            try:
                # If we have channel data fetched earlier, use it
                ch_cache = getattr(self, "_channel_data_cache", None) or None
                if hasattr(self, "_channel_data_cache") and self._channel_data_cache:
                    # Not guaranteed to be present; predictor will fetch if needed.
                    channel_id = None
            except Exception:
                channel_id = None

            video_id = self.selected_video.get("video_id")

            # Instantiate predictor context
            predictor_ctx = predictor_template.Context(self)

            # Run stage 1: gather YouTube data and virality
            try:
                predictor_ctx.stage_1(video_id, channel_id)
            except Exception as e:
                print(f"[Predictor] stage_1 failed: {e}")
                # allow continuing if stage_1 partially populated context (but fail safely)

            # Run stage 2: calculate composite growth score
            try:
                stage2 = predictor_ctx.stage_2()
                print(f"[DEBUG] stage2 result: {stage2}")
            except Exception as e:
                print(f"[Predictor] stage_2 failed: {e}")
                stage2 = {}

            # Run stage 3: get numeric predictions
            try:
                stage3 = predictor_ctx.stage_3()
                print(f"[DEBUG] stage3 result: {stage3}")
            except Exception as e:
                print(f"[Predictor] stage_3 failed: {e}")
                stage3 = predictor_ctx.get_context()  # fallback to whatever we have
                print(f"[DEBUG] fallback stage3: {stage3}")

            # ===== Generate time series using predictor_template's calculations =====
            try:
                import datetime
                import math

                current_views = int(self.selected_video_stats.get("view_count", 0) or 0)

                # USE PREDICTOR_TEMPLATE'S GROWTH SCORE instead of recalculating
                growth_score = stage3.get("growth_score", 0.5)
                predicted_30d = stage3.get("predicted_30d", current_views)
                baseline = stage3.get("baseline", current_views)
                ceiling = stage3.get("ceiling", current_views * 2)

                print(f"[DEBUG] Using predictor_template scores:")
                print(f"  - growth_score: {growth_score:.3f}")
                print(f"  - current_views: {current_views:,}")
                print(f"  - predicted_30d: {predicted_30d:,}")
                print(f"  - baseline: {baseline:,}")
                print(f"  - ceiling: {ceiling:,}")

                # Get component scores for reference
                components = stage2.get("components", {})
                composite_score = stage2.get("composite_growth_score", growth_score)

                print(f"[DEBUG] Component scores: {components}")
                print(f"[DEBUG] Composite growth score: {composite_score:.3f}")

                # Generate smooth curve from current_views to predicted_30d
                total_days = 30
                num_points = 15  # Optimal for smooth curve without overload

                points = []

                # Calculate growth parameters using predictor_template's scores
                # Use growth_score to determine curve shape
                growth_rate = growth_score  # Already normalized 0-1

                # Logistic curve parameters based on growth_score
                # Higher growth_score = steeper curve with earlier inflection
                k = 0.2 + growth_score * 2.0  # Steepness
                t0 = max(5.0, 20.0 - growth_score * 15.0)  # Inflection point (earlier for higher growth)

                # Calculate carrying capacity from predicted_30d
                base = float(current_views)
                carrying_capacity = float(predicted_30d)

                print(
                    f"[DEBUG] Curve parameters: k={k:.3f}, t0={t0:.2f}, base={base:.0f}, capacity={carrying_capacity:.0f}")

                # Generate points using logistic growth curve
                for i in range(num_points):
                    day = i * total_days / (num_points - 1) if num_points > 1 else 0.0

                    # Logistic function: L(t) = base + (K - base) / (1 + exp(-k*(t - t0)))
                    L_t = base + (carrying_capacity - base) / (1.0 + math.exp(-k * (day - t0)))

                    # Constrain to reasonable bounds
                    L_t = max(base, min(ceiling, L_t))

                    points.append((int(round(day)), int(round(L_t))))

                print(f"[DEBUG] Generated {len(points)} points")
                print(f"[DEBUG] First point: {points[0]}, Last point: {points[-1]}")
                print(f"[Predictor] Growth score={growth_score:.3f}, views {points[0][1]:,} -> {points[-1][1]:,}")

                # Update the graph
                try:
                    if hasattr(self, 'predicted_growth_graph_funcs') and self.predicted_growth_graph_funcs:
                        print(f"[DEBUG] Calling set_data with {len(points)} points")
                        self.predicted_growth_graph_funcs['set_data'](points)
                        print(f"[DEBUG] Graph data updated!")

                        # Force graph widget update if available
                        if hasattr(self, 'predicted_growth_graph'):
                            if hasattr(self.predicted_growth_graph, 'update'):
                                self.predicted_growth_graph.update()
                                print(f"[DEBUG] Graph widget updated!")

                        # Force page update to render changes
                        if hasattr(self, 'app_page') and self.app_page:
                            self.app_page.update()
                            print(f"[DEBUG] Page updated!")
                    else:
                        print(f"[DEBUG] predicted_growth_graph_funcs not available")
                except Exception as e:
                    print(f"[Predictor] Graph update failed: {e}")
                    import traceback
                    traceback.print_exc()

            except Exception as e:
                print(f"[Predictor] building time-series failed: {e}")
                import traceback
                traceback.print_exc()

            # ===== Stage 4: generate AI explanation (Gemini) =====
            try:
                tab_state = tabs_script.read_tab_state(self.tab_id) or {}

                # Gemini API key uses its own keyring (NOT OAuth)
                google_keyring_name = tab_state.get("google_api_key_name")

                api_key = None

                if google_keyring_name:
                    try:
                        import keyring

                        # IMPORTANT:
                        # Saved via: keyring.set_password("sulfur-google", api_key_name, api_key)
                        # So we must restore with the same (service, username) pair
                        api_key = keyring.get_password(
                            "sulfur-google",  # service
                            google_keyring_name  # username
                        )

                        print("[DEBUG] Gemini api_key:", api_key)
                    except Exception as e:
                        print(f"[Gemini] Failed to restore API key: {e}")

                if api_key:
                    try:
                        predictor_ctx.stage_4(api_key)
                        ai_out = predictor_ctx.get_context().get("ai_explanation", "")
                    except Exception as e:
                        print(f"[Predictor] stage_4 (Gemini) failed: {e}")
                        ai_out = f"Error generating explanation: {e}"
                else:
                    ai_out = "No Google API key available in keyring (google_api_key_name)."

            except Exception as e:
                print(f"[Predictor] running stage_4 failed: {e}")
                ai_out = f"Error generating explanation: {e}"

            # ===== Insert AI explanation text into the ai_explanation_container =====
            try:
                if self.ai_explanation_container:
                    # Find the Column inside the base_template (children[1] in our construction)
                    inner_col = None
                    try:
                        # base_template likely contains .content with .controls or direct .controls
                        content = getattr(self.ai_explanation_container, "content", None)
                        if content and hasattr(content, "controls"):
                            # the Column should be the second child
                            for c in content.controls:
                                # find the Column control (first ft.Column)
                                if getattr(c, "__class__", type(c)).__name__ == "Column":
                                    inner_col = c
                                    break
                        # if not found, try scanning ai_explanation_container.controls
                        if inner_col is None and hasattr(self.ai_explanation_container, "controls"):
                            for c in self.ai_explanation_container.controls:
                                if getattr(c, "__class__", type(c)).__name__ == "Column":
                                    inner_col = c
                                    break
                    except Exception:
                        inner_col = None

                    # Update the Column controls to show the AI explanation, split into short lines
                    new_texts = []
                    if isinstance(ai_out, str):
                        # keep it compact: split into paragraphs if long
                        lines = [p.strip() for p in ai_out.split("\n") if p.strip()]
                        # if a single long line, wrap at ~200 chars
                        import textwrap
                        wrapped = []
                        for ln in lines:
                            if len(ln) > 200:
                                wrapped += textwrap.wrap(ln, width=200)
                            else:
                                wrapped.append(ln)

                        # Convert to Text controls
                        for line in wrapped:
                            new_texts.append(
                                ft.Text(
                                    line,
                                    color=FG,
                                    size=int(tabs_script.s(14, self.app_page)),
                                )
                            )

                    # Update the inner column if found
                    if inner_col:
                        inner_col.controls = new_texts
                        print(f"[DEBUG] Updated AI explanation with {len(new_texts)} text blocks")

                    # Trigger UI update
                    if hasattr(self, 'app_page') and self.app_page:
                        self.app_page.update()

            except Exception as e:
                print(f"[Predictor] Failed to update AI explanation: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"[Predictor] Overall failure: {e}")
            import traceback
            traceback.print_exc()

    def _run_predictor_for_latest_video(self):
        """Run predictor_template stages 1-2 for the latest video and update with composite growth score."""
        try:
            if not self.latest_video_data:
                print("[Dashboard] No latest video data available")
                return

            video_id = self.latest_video_data.get('video_id')
            if not video_id:
                print("[Dashboard] Latest video has no video_id")
                return

            print(f"[Dashboard] Running predictor for latest video: {video_id}")

            # Instantiate predictor context
            predictor_ctx = predictor_template.Context(self)

            # Run stages 1-2 to get composite growth score
            try:
                predictor_ctx.stage_1(video_id, None)
                stage2 = predictor_ctx.stage_2()

                # Extract composite growth score from stage 2
                composite_score = stage2.get("composite_growth_score", 0.5)
                components = stage2.get("components", {})

                # Format as X/10 score
                score_out_of_10 = round(composite_score * 10, 1)

                print(f"[Dashboard] Composite growth score: {composite_score:.3f} ({score_out_of_10}/10)")
                if components:
                    print(f"[Dashboard] Components: {components}")

                # Update the Text widget
                if self.latest_video_growth_score_text:
                    self.latest_video_growth_score_text.value = f"Overall Growth Score: {score_out_of_10}/10"

                    # Trigger UI update
                    if hasattr(self, 'app_page') and self.app_page:
                        try:
                            self.app_page.update()
                            print("[Dashboard] Growth score display updated")
                        except Exception as e:
                            print(f"[Dashboard] Failed to update UI: {e}")
                else:
                    print("[Dashboard] Growth score text widget not available")

            except Exception as e:
                print(f"[Dashboard] Predictor stages failed: {e}")
                import traceback
                traceback.print_exc()

                # Fallback: show error state
                if self.latest_video_growth_score_text:
                    self.latest_video_growth_score_text.value = "Overall Growth Score: Error"
                    if hasattr(self, 'app_page') and self.app_page:
                        try:
                            self.app_page.update()
                        except:
                            pass

        except Exception as e:
            print(f"[Dashboard] Failed to run predictor for latest video: {e}")
            import traceback
            traceback.print_exc()

    def add_alert(self, text: str, *, save=True, prepend=True):
        """
        Add an alert programmatically.

        Args:
            text: alert message
            save: persist immediately
            prepend: add to top instead of bottom
        """
        if not hasattr(self, "alert_list_container"):
            return

        card = self._build_alert_card(text)

        if prepend:
            self.alert_list_container.controls.insert(0, card)
        else:
            self.alert_list_container.controls.append(card)

        if save:
            self._save_alert_history()

        try:
            self.app_page.update()
        except Exception:
            pass

    # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    # |                                                                                                                                                                        |
    # |                                                                      IMPORTANT CODE (NOT UI)                                                                                     |
    # |                                                                                                                                                                        |
    # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

    def _traverse_controls_for_index(self, ctrls, path_prefix="", out=None):
        if out is None:
            out = []
        for c in ctrls:
            title_parts = []
            for attr in ("text", "value", "label", "hint_text"):
                try:
                    v = getattr(c, attr, None)
                    if v:
                        title_parts.append(str(v))
                except Exception:
                    pass
            title = " ".join(title_parts).strip()
            control_type = getattr(c, "__class__", type(c)).__name__
            path_label = title if title else control_type
            path = f"{path_prefix}/{path_label}".strip("/")

            out.append({
                # Serialize only safe fields
                "path": path,
                "title": title.lower() if title else control_type.lower()
            })

            # recurse
            try:
                if hasattr(c, "controls") and c.controls:
                    self._traverse_controls_for_index(c.controls, path_prefix=path, out=out)
            except Exception:
                pass
            try:
                if hasattr(c, "content") and hasattr(c.content, "controls") and c.content.controls:
                    self._traverse_controls_for_index(c.content.controls, path_prefix=path, out=out)
            except Exception:
                pass

        return out

    def _build_search_index(self):
        """
        Build a search index by scanning sidebar buttons and the current inner_pages' control trees.
        Returns an object with:
          - names: list of sidebar items (lowercased)
          - pages: dict mapping inner page name -> list of index entries (each entry holds control refs)
        The index entries have: {control, path, title}
        """
        index_data = {"names": [], "pages": {}}

        # add sidebar item names (so simple name matches still work)
        try:
            index_data["names"] = [name.lower() for name in getattr(self, "sidebar_items", [])]
        except Exception:
            index_data["names"] = []

        # For each inner page that exists, walk its control tree
        for name, page_container in (self.inner_pages or {}).items():
            try:
                entries = []
                # page_container may be a Container with .content that has controls
                content_root = None
                if page_container is None:
                    index_data["pages"][name] = []
                    continue
                # prefer page_container.content if present (this mirrors how you set main_container.content)
                if hasattr(page_container, "content") and page_container.content is not None:
                    # if it's a Container with a Column inside, dive one more level
                    inner = page_container.content
                    # if inner has a .controls, traverse them directly
                    if hasattr(inner, "controls") and inner.controls:
                        entries = self._traverse_controls_for_index(inner.controls, path_prefix=name, out=[])
                    # else if inner itself is a Column with content.controls, try that
                    elif hasattr(inner, "content") and hasattr(inner.content, "controls") and inner.content.controls:
                        entries = self._traverse_controls_for_index(inner.content.controls, path_prefix=name, out=[])
                    else:
                        # fallback: try traversing page_container.controls
                        if hasattr(page_container, "controls") and page_container.controls:
                            entries = self._traverse_controls_for_index(page_container.controls, path_prefix=name,
                                                                        out=[])
                else:
                    # fallback: if page_container is a Column/Container with .controls
                    if hasattr(page_container, "controls") and page_container.controls:
                        entries = self._traverse_controls_for_index(page_container.controls, path_prefix=name, out=[])
                index_data["pages"][name] = entries
            except Exception:
                index_data["pages"][name] = []

        # persist index to tab state if desired (keeps behavior similar to previous)
        # persist index into tab state safely
        if self.tab_id:
            try:
                safe_index = {"names": index_data["names"], "pages": index_data["pages"]}
                tabs_script.update_tab_state(self.tab_id, {"search_index": safe_index})
            except Exception as e:
                print(f"Failed to save search_index for tab {self.tab_id}: {e}")

        return index_data

    def _mark_search_matches(self, query: str, limit: int = 10):
        """
        Use the built index to find matching controls and insert markers under them.
        This is safer & faster than repeatedly scanning arbitrary attributes with ad-hoc recursion.
        """
        try:
            # clear previous markers
            self._clear_search_markers()
            if not query:
                return

            q = query.strip().lower()
            if not q:
                return

            found_controls = []

            # build or retrieve fresh index (scan live)
            index = self._build_search_index()

            # 1) match sidebar names quickly
            for name in index.get("names", []):
                if q in name and name not in found_controls:
                    found_controls.append({"control": None, "path": f"sidebar/{name}", "title": name})
                    if len(found_controls) >= limit:
                        break

            # 2) search each page's indexed entries (these include control refs)
            if len(found_controls) < limit:
                for page_name, entries in index.get("pages", {}).items():
                    if len(found_controls) >= limit:
                        break
                    for e in entries:
                        if len(found_controls) >= limit:
                            break
                        title = e.get("title", "")
                        if not title:
                            continue
                        if q in title:
                            ctrl = e.get("control")
                            # avoid duplicates by identity
                            already = any(item.get("control") is ctrl for item in found_controls if
                                          item.get("control") is not None)
                            if not already:
                                found_controls.append({"control": ctrl, "path": e.get("path"), "title": title})
                                if len(found_controls) >= limit:
                                    break

            # Insert markers for the found controls (skip items with no control obj like sidebar hits)
            for item in found_controls[:limit]:
                ctrl = item.get("control")
                if ctrl is None:
                    continue
                try:
                    self._mark_control_with_overlay(ctrl)
                except Exception:
                    pass

            # final UI update
            try:
                if self.app_page:
                    self.app_page.update()
            except Exception:
                pass

        except Exception:
            pass

    def _mark_control_with_overlay(self, control):
        """
        Insert a thin marker Container immediately after `control` inside its parent's controls list.
        Returns the marker container or None.
        """
        try:
            if control is None:
                return None

            # Find the parent container that actually holds a .controls list
            parent = getattr(control, "parent", None)

            # If parent is a Container with .content (Column), prefer parent.content
            controls_list = None
            effective_parent = parent
            if parent is None:
                # can't locate parent: bail out
                return None

            # If parent has direct .controls (e.g., Column), use it
            if hasattr(parent, "controls"):
                controls_list = parent.controls
            # If parent is a Container with .content that has controls, use parent.content.controls
            elif hasattr(parent, "content") and hasattr(parent.content, "controls"):
                controls_list = parent.content.controls
                effective_parent = parent.content
            else:
                # nothing we can insert into
                return None

            # find index of the control in that list
            try:
                idx = controls_list.index(control)
            except ValueError:
                # maybe the control is wrapped inside an entry's .content; try to find by identity
                idx = None
                for i, c in enumerate(controls_list):
                    if c is control or getattr(c, "content", None) is control:
                        idx = i
                        break
                if idx is None:
                    return None

            # Create marker (thin band; visually obvious)
            marker = ft.Container(
                border_radius=int(tabs_script.s(6, self.app_page)),
                border=ft.border.all(1, ORANGE),
                bgcolor="transparent",
                height=0,
                padding=ft.padding.symmetric(horizontal=4, vertical=0)
            )

            # Insert marker just after control
            insert_index = idx + 1
            controls_list.insert(insert_index, marker)

            # record so we can remove later
            if not hasattr(self, "_search_markers") or self._search_markers is None:
                self._search_markers = []
            self._search_markers.append({"marker": marker, "parent": effective_parent, "idx": insert_index})
            return marker
        except Exception:
            return None

    def _mark_search_matches(self, query: str, limit: int = 10):
        """
        Search visible UI (main content + sidebar buttons) for controls that match `query`
        and insert a visual marker under each match (up to `limit`).
        This does NOT change routing or scroll position (no jumping).
        """
        try:
            # clear existing markers first
            self._clear_search_markers()
            if not query:
                return

            q = query.strip().lower()
            if not q:
                return

            found = []

            def scan_controls(ctrls):
                for c in ctrls:
                    # gather textual candidates
                    text_candidates = []
                    for attr in ("text", "value", "label"):
                        if hasattr(c, attr):
                            v = getattr(c, attr, None)
                            if v:
                                text_candidates.append(str(v).lower())
                    combined = " ".join(text_candidates)
                    if q in combined and c not in found:
                        found.append(c)
                        # short-circuit if we've reached the limit
                        if len(found) >= limit:
                            return
                    # recursive descent: .controls
                    try:
                        if hasattr(c, "controls") and c.controls:
                            scan_controls(c.controls)
                            if len(found) >= limit:
                                return
                    except Exception:
                        pass
                    # recursive descent: .content.controls (Container with Column)
                    try:
                        if hasattr(c, "content") and hasattr(c.content, "controls") and c.content.controls:
                            scan_controls(c.content.controls)
                            if len(found) >= limit:
                                return
                    except Exception:
                        pass

            # scan the current main container content (if present)
            try:
                main_cont = getattr(self, "main_container", None)
                if main_cont is not None:
                    # main_container.content may be a Container/Column/Box; handle both patterns
                    mc = getattr(main_cont, "content", None)
                    if mc is not None:
                        # If mc is a Column with controls
                        if hasattr(mc, "controls"):
                            scan_controls(mc.controls)
                        # If mc is a Container with content.controls
                        elif hasattr(mc, "content") and hasattr(mc.content, "controls"):
                            scan_controls(mc.content.controls)
            except Exception:
                pass

            # scan the sidebar buttons (we keep references in self.sidebar_buttons)
            try:
                for btn in (self.sidebar_buttons or {}).values():
                    if len(found) >= limit:
                        break
                    scan_controls([btn])
            except Exception:
                pass

            # Now mark found controls (insert marker under each)
            for c in found[:limit]:
                try:
                    self._mark_control_with_overlay(c)
                except Exception:
                    pass

            # trigger UI update
            try:
                if self.app_page:
                    self.app_page.update()
            except Exception:
                pass

        except Exception:
            pass

    def _on_result_click(self, name, e):
        """
        Handle click on a search result: switch to the matching subpage and highlight the search term.
        """
        # 1. Switch to the selected page (updates inner_selected, rebuilds content)
        self._on_inner_click(name, e)
        # 2. Highlight the first match of the current query on this page
        query = self.sidebar_search
        if not query:
            return
        try:
            # Get the Column of the main content we just built
            main_col = getattr(self.main_container.content, "content", None)
            if main_col and hasattr(main_col, "controls"):
                for c in main_col.controls:
                    text_candidates = []
                    if hasattr(c, "text"):
                        text_candidates.append(str(c.text or ""))
                    if hasattr(c, "value"):
                        text_candidates.append(str(c.value or ""))
                    if hasattr(c, "label"):
                        text_candidates.append(str(c.label or ""))
                    combined = " ".join(t.lower() for t in text_candidates)
                    if query in combined:
                        # Mark this control (e.g., draw an orange border or dark background)
                        try:
                            c.border = ft.border.all(2, ORANGE)
                        except Exception:
                            c.bgcolor = "#222222"
                        try:
                            c.focus()
                        except Exception:
                            pass
                        try:
                            if self.app_page and hasattr(self.app_page, "scroll_to"):
                                self.app_page.scroll_to(c)
                        except Exception:
                            pass
                        break
        except Exception:
            pass
        # Finally, update the page UI
        try:
            if self.app_page:
                self.app_page.update()
        except:
            pass

    def _format_time(self, obtained_iso, expires_in):
        try:
            if obtained_iso:
                # remove trailing Z if present, then parse
                iso = obtained_iso.replace("Z", "")
                obtained = datetime.datetime.fromisoformat(iso)
                expires_at = obtained + datetime.timedelta(seconds=int(expires_in or 0))
                return expires_at.isoformat()
            else:
                return ""
        except Exception:
            try:
                # fallback: now + expires_in
                return (datetime.datetime.utcnow() + datetime.timedelta(seconds=int(expires_in or 0))).isoformat() + "Z"
            except Exception:
                return ""

    def _restore_cache(self):
        """Restore inner_selected and inner_data from saved tab file for this tab_id."""
        if not self.tab_id or self._restored:
            return
        try:
            state = tabs_script.read_tab_state(self.tab_id) or {}
            self.inner_selected = state.get("inner_tab", "Dashboard")
            self.inner_data = state.get("inner_data", {}) or {}
        except Exception:
            self.inner_selected = "Dashboard"
            self.inner_data = {}
        self._restored = True

    def _persist_inner_selection(self):
        """Persist the current inner_selected and inner_data into tab file (merge)."""
        if not self.tab_id:
            return
        try:
            tabs_script.update_tab_state(self.tab_id, {"inner_tab": self.inner_selected, "inner_data": self.inner_data})
        except Exception as e:

            from scripts.ai_renderer_sentences.error import SulfurError
            raise SulfurError(message=f"persist_inner_selection error: {e}")

    def _on_field_change(self, field_name, e):
        """Store live edits into inner_data for the current inner tab and persist."""
        try:
            val = getattr(e.control, "value", "")
            tab_key = self.inner_selected or "Dashboard"
            store = self.inner_data.get(tab_key, {})
            store[field_name] = val
            self.inner_data[tab_key] = store
            self._persist_inner_selection()
        except Exception:
            pass

    # ================= ALERTS / PLAN / CHAT CACHE HELPERS =================

    def _extract_text_from_control(self, ctrl):
        """Attempt to extract a user-visible text string from a control (defensive)."""
        try:
            # common direct text attributes
            for attr in ("text", "value", "label", "hint_text"):
                if hasattr(ctrl, attr):
                    v = getattr(ctrl, attr, None)
                    if v is not None and str(v).strip() != "":
                        return str(v)
            # nested container/content patterns
            if hasattr(ctrl, "content") and hasattr(ctrl.content, "controls"):
                for c in ctrl.content.controls:
                    t = self._extract_text_from_control(c)
                    if t:
                        return t
            if hasattr(ctrl, "controls"):
                for c in ctrl.controls:
                    t = self._extract_text_from_control(c)
                    if t:
                        return t
        except Exception:
            pass
        return ""

    # ------------------ Alerts ------------------

    def _restore_alert_history(self):
        if not getattr(self, "tab_id", None):
            return

        state = tabs_script.read_tab_state(self.tab_id) or {}
        alerts = state.get("alert_history", [])

        self.alert_list_container.controls.clear()

        for item in alerts:
            self.alert_list_container.controls.append(
                self._build_alert_card(item.get("text", ""))
            )

        try:
            self.app_page.update()
        except Exception:
            pass

    def _collect_alerts_for_save(self):
        """Collect serializable alert dicts from self.alert_list_container."""
        out = []
        if not hasattr(self, "alert_list_container") or self.alert_list_container is None:
            return out
        try:
            for ctrl in (self.alert_list_container.controls or []):
                # try to extract text
                txt = self._extract_text_from_control(ctrl) or ""
                out.append({"text": txt})
        except Exception:
            pass
        return out

    def _save_alert_history(self):
        if not getattr(self, "tab_id", None):
            return

        alerts = []
        for ctrl in self.alert_list_container.controls:
            try:
                text = ctrl.content.controls[0].value or ctrl.content.controls[0].text
            except Exception:
                text = ""
            alerts.append({"text": text})

        tabs_script.update_tab_state(self.tab_id, {"alert_history": alerts})

    # ------------------ Plan-a-Video (plan_tabs) ------------------

    def _restore_plan_tabs(self):
        """
        Restore all tabplan_N blocks from cache.
        """
        try:
            if not getattr(self, "tab_id", None):
                return

            state = tabs_script.read_tab_state(self.tab_id) or {}

            # Find all tabplan_N blocks
            tab_blocks = {}
            for key in state.keys():
                if key.startswith("tabplan_"):
                    try:
                        block_num = int(key.split("_")[1])
                        tab_blocks[block_num] = state[key]
                    except Exception:
                        pass

            # Check old format for backwards compatibility
            old_plan_tabs = state.get("plan_tabs", [])

            if not tab_blocks and not old_plan_tabs:
                print("No saved tabs found in cache")
                return

            widget = getattr(self, "plan_tabs_widget", None)
            if not widget:
                return

            # Clear existing tabs
            try:
                if hasattr(widget, "close_tab") and hasattr(widget, "get_tabs_state"):
                    current_tabs = [t.get("title") for t in (widget.get_tabs_state() or [])]
                    for tab_title in current_tabs:
                        try:
                            widget.close_tab(tab_title)
                        except Exception:
                            pass
            except Exception as ex:
                print(f"Could not clear tabs: {ex}")

            # Load from new format (tabplan_N blocks)
            if tab_blocks:
                for block_num in sorted(tab_blocks.keys()):
                    tab_data = tab_blocks[block_num]
                    title = tab_data.get("title", f"Video {block_num}")
                    content = tab_data.get("content", {})

                    if isinstance(content, dict):
                        ideas = content.get("ideas", "")
                        format_ideas = content.get("format_ideas", "")
                        sound_ideas = content.get("sound_ideas", "")
                    else:
                        ideas = str(content) if content else ""
                        format_ideas = ""
                        sound_ideas = ""

                    tab_content = self._make_plan_tab_content(
                        ideas=ideas,
                        format_ideas=format_ideas,
                        sound_ideas=sound_ideas
                    )

                    if hasattr(widget, "add_tab"):
                        widget.add_tab(title, tab_content)
                        print(f"Restored tab '{title}' from tabplan_{block_num}")

            # Load from old format (backwards compatibility)
            elif old_plan_tabs:
                for idx, tab_data in enumerate(old_plan_tabs):
                    title = tab_data.get("title", f"Video {idx + 1}")
                    ideas = tab_data.get("ideas", "")
                    format_ideas = tab_data.get("format_ideas", "")
                    sound_ideas = tab_data.get("sound_ideas", "")

                    tab_content = self._make_plan_tab_content(
                        ideas=ideas,
                        format_ideas=format_ideas,
                        sound_ideas=sound_ideas
                    )

                    if hasattr(widget, "add_tab"):
                        widget.add_tab(title, tab_content)
                        print(f"Restored tab '{title}' from old format")

            # Update page
            if self.app_page and hasattr(self.app_page, "update"):
                self.app_page.update()

        except Exception as e:
            print(f"Failed to restore plan tabs: {e}")
            import traceback
            traceback.print_exc()

    def _make_plan_tab_content(self, ideas="", format_ideas="", sound_ideas=""):
        def tf(val):
            return ft.TextField(
                value=val or "",
                multiline=True,
                min_lines=10,
                expand=True,
                on_change=lambda e: self._save_plan_tabs(),
                on_blur=lambda e: self._save_plan_tabs(),
            )

        return ft.Container(
            padding=10,
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=10,
                controls=[
                    ft.Row(
                        spacing=10,
                        controls=[
                            base_template(
                                height=int(tabs_script.sv(280, self.app_page)),
                                width=int(tabs_script.sh(275, self.app_page)),
                                page=self.app_page,
                                children=[
                                    ft.Container(
                                        expand=True,
                                        padding=8,
                                        content=ft.Column(
                                            expand=True,
                                            controls=[
                                                ft.Text("Ideas", weight="bold", color=FG),
                                                tf(ideas),
                                            ],
                                        ),
                                    )
                                ],
                            ),
                            base_template(
                                height=int(tabs_script.sv(280, self.app_page)),
                                width=int(tabs_script.sh(275, self.app_page)),
                                page=self.app_page,
                                children=[
                                    ft.Container(
                                        expand=True,
                                        padding=8,
                                        content=ft.Column(
                                            expand=True,
                                            controls=[
                                                ft.Text("Format Ideas", weight="bold", color=FG),
                                                tf(format_ideas),
                                            ],
                                        ),
                                    )
                                ],
                            ),
                            base_template(
                                height=int(tabs_script.sv(280, self.app_page)),
                                width=int(tabs_script.sh(275, self.app_page)),
                                page=self.app_page,
                                children=[
                                    ft.Container(
                                        expand=True,
                                        padding=8,
                                        content=ft.Column(
                                            expand=True,
                                            controls=[
                                                ft.Text("Sound Ideas", weight="bold", color=FG),
                                                tf(sound_ideas),
                                            ],
                                        ),
                                    )
                                ],
                            ),
                        ],
                    ),
                    base_template(
                        height=int(tabs_script.sv(50, self.app_page)),
                        width=int(tabs_script.sh(850, self.app_page)),
                        page=self.app_page,
                        children=[
                            ft.Text(
                                "Move video to Video Plans page.",
                                weight="bold",
                                color=FG,
                            )
                        ],
                    ),
                ],
            ),
        )

    def _save_plan_tabs(self):
        """
        Save each tab to its own tabplan_N block.
        """
        if not getattr(self, "tab_id", None):
            return

        try:
            widget = getattr(self, "plan_tabs_widget", None)
            if not widget:
                return

            # Get tabs state
            if hasattr(widget, "get_tabs_state"):
                tabs_state = widget.get_tabs_state() or []
            else:
                return

            # Initialize counter if not exists
            if not hasattr(self, "_plan_tab_counter"):
                state = tabs_script.read_tab_state(self.tab_id) or {}
                max_block = 0
                for key in state.keys():
                    if key.startswith("tabplan_"):
                        try:
                            block_num = int(key.split("_")[1])
                            max_block = max(max_block, block_num)
                        except Exception:
                            pass
                self._plan_tab_counter = max_block

            # Save each tab to its own block
            for idx, tab_data in enumerate(tabs_state):
                block_num = idx + 1

                if block_num > self._plan_tab_counter:
                    self._plan_tab_counter = block_num

                block_name = f"tabplan_{block_num}"
                tab_content = {
                    "title": tab_data.get("title", "Video"),
                    "content": tab_data.get("content", {})
                }

                tabs_script.update_tab_state(self.tab_id, {block_name: tab_content})
                print(f"Saved tab '{tab_content['title']}' to {block_name}")

        except Exception as e:
            print(f"Failed to save plan tabs: {e}")
            import traceback
            traceback.print_exc()

    def _collect_plan_tabs_for_save(self):
        """
        Collect plan tabs and return list of dict:
          [{"title": str, "ideas": str, "format_ideas": str, "sound_ideas": str}, ...]
        """
        out = []

        widget = getattr(self, "plan_tabs_widget", None) or getattr(self, "tabs", None) or getattr(self,
                                                                                                   "plan_tabs_host",
                                                                                                   None)
        if not widget:
            return out

        # 1) If the widget exposes a richer state getter, try that first
        try:
            if hasattr(widget, "get_tabs_state"):
                try:
                    state_list = widget.get_tabs_state() or []
                    if state_list:
                        for i in state_list:
                            title = i.get("title") or "Plan"
                            content = i.get("content") or {}
                            if isinstance(content, dict):
                                ideas = content.get("ideas", "") or ""
                                fmt = content.get("format_ideas", "") or ""
                                sound = content.get("sound_ideas", "") or ""
                            else:
                                # legacy: content may be a string -> put into ideas
                                ideas = content or ""
                                fmt = ""
                                sound = ""
                            out.append({
                                "title": title,
                                "ideas": ideas,
                                "format_ideas": fmt,
                                "sound_ideas": sound,
                            })
                        return out
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Fallback: inspect widget.get_tabs() or widget.tabs and extract up to 3 TextField values
        try:
            tabs_list = []
            if hasattr(widget, "get_tabs"):
                try:
                    tabs_list = widget.get_tabs() or []
                except Exception:
                    tabs_list = []
            if not tabs_list:
                try:
                    tabs_list = getattr(widget, "tabs", None) or []
                except Exception:
                    tabs_list = []

            def _extract_textfields_values(ctrl):
                vals = []
                try:
                    if isinstance(ctrl, ft.TextField):
                        vals.append(getattr(ctrl, "value", "") or "")
                        return vals
                    if hasattr(ctrl, "content") and ctrl.content is not None:
                        vals += _extract_textfields_values(ctrl.content)
                    if hasattr(ctrl, "controls") and ctrl.controls:
                        for ch in ctrl.controls:
                            vals += _extract_textfields_values(ch)
                except Exception:
                    pass
                return vals

            for tab in tabs_list:
                try:
                    title = ""
                    ctrl = None
                    if hasattr(tab, "title") or hasattr(tab, "label"):
                        title = getattr(tab, "title", None) or getattr(tab, "label", None) or ""
                        ctrl = getattr(tab, "content", None)
                    elif isinstance(tab, (list, tuple)) and len(tab) >= 2:
                        title = str(tab[0] or "")
                        ctrl = tab[1]
                    elif isinstance(tab, dict):
                        title = tab.get("title") or tab.get("label") or ""
                        if isinstance(tab.get("content"), dict):
                            c = tab.get("content")
                            ideas = c.get("ideas", "") or ""
                            fmt = c.get("format_ideas", "") or ""
                            sound = c.get("sound_ideas", "") or ""
                            out.append(
                                {"title": title or "Plan", "ideas": ideas, "format_ideas": fmt, "sound_ideas": sound})
                            continue
                        ctrl = tab.get("content")
                    else:
                        title = str(tab)

                    vals = _extract_textfields_values(ctrl) if ctrl is not None else []
                    ideas = vals[0] if len(vals) > 0 else ""
                    fmt = vals[1] if len(vals) > 1 else ""
                    sound = vals[2] if len(vals) > 2 else ""
                    out.append({"title": title or "Plan", "ideas": ideas, "format_ideas": fmt, "sound_ideas": sound})
                except Exception:
                    continue
        except Exception:
            pass

        return out

    # ------------------ Chatbot ------------------

    def _restore_chat_history(self):
        """
        Restore chat history from ALL chat blocks (chat_1, chat_2, etc.) in the per-tab cache.
        This will attempt to find a chat panel inside self.chatbot_container; if not found
        it will populate a fallback chat history column self.chat_history_list.
        """
        if not getattr(self, "tab_id", None):
            return
        try:
            state = tabs_script.read_tab_state(self.tab_id) or {}

            # Load from BOTH old format and new format for backwards compatibility
            history = []

            # First, check for old format (single chat_history key)
            old_history = state.get("chat_history", []) or []
            if old_history:
                history.extend(old_history)

            # Then, load all chat blocks (chat_1, chat_2, chat_3, etc.)
            chat_blocks = {}
            for key in state.keys():
                if key.startswith("chat_"):
                    try:
                        block_num = int(key.split("_")[1])
                        chat_blocks[block_num] = state[key]
                    except Exception:
                        pass

            # Add messages from all chat blocks in order
            if chat_blocks:
                for block_num in sorted(chat_blocks.keys()):
                    block_messages = chat_blocks[block_num]
                    if isinstance(block_messages, list):
                        history.extend(block_messages)

            # try to find the chat display area inside the chatbot container
            container = getattr(self, "chatbot_container", None)
            chat_panel = None
            if container is not None:
                # try common nested patterns
                try:
                    cand = getattr(container, "content", None)
                    if cand and hasattr(cand, "controls"):
                        # find a Column or Container named like chat area (heuristic)
                        for c in cand.controls:
                            # choose first Column or Container with many controls
                            if hasattr(c, "controls") or (hasattr(c, "content") and hasattr(c.content, "controls")):
                                chat_panel = c
                                break
                except Exception:
                    chat_panel = None

            # if not found, create fallback chat list
            if chat_panel is None:
                if not hasattr(self, "chat_history_list") or self.chat_history_list is None:
                    self.chat_history_list = ft.Column(controls=[], spacing=6, expand=True, scroll="auto")
                    # insert fallback under chatbot container if possible
                    try:
                        if container is not None and hasattr(container, "content") and hasattr(container.content,
                                                                                               "controls"):
                            container.content.controls.append(ft.Container(content=self.chat_history_list, padding=6))
                    except Exception:
                        pass
                chat_panel = self.chat_history_list

            # clear existing messages
            try:
                chat_panel.controls.clear()
            except Exception:
                pass

            # append messages from ALL loaded history
            for m in history:
                role = m.get("role", "assistant")
                content = m.get("content", "") or ""
                ts = m.get("timestamp")
                text = f"[{role}] {content}" + (f" ({ts})" if ts else "")
                try:
                    chat_panel.controls.append(base_template(
                        height=int(tabs_script.sv(50, self.app_page)),
                        width=int(tabs_script.sh(850, self.app_page)),
                        page=self.app_page,
                        children=[ft.Text(text, color=FG)],
                    ))
                except Exception:
                    chat_panel.controls.append(ft.Text(text, color=FG))

            try:
                if self.app_page:
                    self.app_page.update()
            except Exception:
                pass

        except Exception as e:
            print(f"Failed to restore chat history for tab {self.tab_id}: {e}")

    def _collect_chat_history_for_save(self):
        """
        Try to collect chat messages from the chat panel or fallback list; produce
        a list of dicts {"role","timestamp","content"}.
        """
        out = []
        container = getattr(self, "chatbot_container", None)
        chat_panel = None
        if container is not None:
            try:
                cand = getattr(container, "content", None)
                if cand and hasattr(cand, "controls"):
                    # choose the first Column-like child (heuristic)
                    for c in cand.controls:
                        if hasattr(c, "controls") or (hasattr(c, "content") and hasattr(c.content, "controls")):
                            chat_panel = c
                            break
            except Exception:
                chat_panel = None

        if chat_panel is None:
            chat_panel = getattr(self, "chat_history_list", None)

        if chat_panel is None:
            return out

        try:
            for ctrl in (chat_panel.controls or []):
                text = self._extract_text_from_control(ctrl) or ""
                role = "assistant"
                content = text
                # heuristics to parse role prefix "[role] message"
                if text.startswith("[") and "]" in text:
                    try:
                        role = text.split("]")[0].strip("[]")
                        content = text.split("] ", 1)[1] if "] " in text else text.split("]", 1)[1]
                    except Exception:
                        pass
                out.append({
                    "role": role,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "content": content
                })
        except Exception:
            pass
        return out

    def _save_chat_history(self, messages=None):
        """
        Persist chat history for this tab using separate cache blocks (chat_1, chat_2, etc.).
        All messages in the CURRENT SESSION go to the same chat block.

        If `messages` (list of strings) is provided by the caller, use it directly.
        Each string is expected to be like "User: <text>" or "Bot: <text>".
        Otherwise fall back to scanning the UI (existing heuristic).
        """
        if not getattr(self, "tab_id", None):
            return

        # Initialize counter if not exists
        if not hasattr(self, "_chat_block_counter"):
            # Load existing chat block count from cache
            state = tabs_script.read_tab_state(self.tab_id) or {}
            max_block = 0
            for key in state.keys():
                if key.startswith("chat_"):
                    try:
                        block_num = int(key.split("_")[1])
                        max_block = max(max_block, block_num)
                    except Exception:
                        pass
            self._chat_block_counter = max_block
            # Start a new session (new chat block)
            self._chat_block_counter += 1
            self._current_chat_block = f"chat_{self._chat_block_counter}"
            print(f"Started new chat session: {self._current_chat_block}")

        try:
            # If caller provided the authoritative message list, use that.
            if isinstance(messages, (list, tuple)):
                # Parse all messages
                parsed_messages = []
                for m in messages:
                    if not isinstance(m, str):
                        continue

                    role = "assistant"
                    content = m

                    if m.startswith("User:"):
                        role = "user"
                        content = m[len("User:"):].strip()
                    elif m.startswith("Bot:"):
                        role = "assistant"
                        content = m[len("Bot:"):].strip()
                    else:
                        # fallback: treat as assistant content
                        content = m

                    parsed_messages.append({
                        "role": role,
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "content": content
                    })

                # Save ALL messages to the CURRENT chat block
                if parsed_messages:
                    tabs_script.update_tab_state(self.tab_id, {self._current_chat_block: parsed_messages})
                    print(f"Saved {len(parsed_messages)} messages to {self._current_chat_block}")

            else:
                # fallback: original heuristic (attempt to collect from UI)
                history = self._collect_chat_history_for_save()
                if history:
                    tabs_script.update_tab_state(self.tab_id, {self._current_chat_block: history})

        except Exception as e:
            print(f"Failed to save chat history for tab {self.tab_id}: {e}")

    def _start_new_chat_session(self):
        """
        Start a new chat session by incrementing the chat block counter.
        Call this when "New Chat" is clicked.
        """
        if not getattr(self, "tab_id", None):
            return

        # Increment to next chat block
        if hasattr(self, "_chat_block_counter"):
            self._chat_block_counter += 1
        else:
            self._chat_block_counter = 1

        self._current_chat_block = f"chat_{self._chat_block_counter}"
        print(f"Started new chat session: {self._current_chat_block}")

    # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    # |                                                                                                                                                                        |
    # |                                                                       UI                                                                                     |
    # |                                                                                                                                                                        |
    # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

    def _create_inner_page(self, name):
        """
        Create and return a Container that represents the page for `name`.

        Developer Notes:
        - This version intentionally contains *no fields*.
        - Only the title/header is included so each inner page loads properly.
        - Additional content can be appended later directly via:
              self.inner_pages["TabName"].content.controls.append(...)
        """

        try:
            # Page title only
            title = ft.Text(
                f"{name}",
                size=int(tabs_script.s(22, self.app_page)),
                weight="w700",
                color=FG
            )

            SUBTITLES = {
                "Dashboard": "Overview: Predict Subscriber Growth & Revenue",
                "Videos": "Predict the Growth of your Videos",
                "Chatbot": "Ask questions about how to grow your channel",
                "Settings": "Configure application behavior",
                "Info": "System and model information",
            }

            subtitle_text = SUBTITLES.get(name, "")

            subtitle = ft.Text(
                subtitle_text,
                size=int(tabs_script.s(13, self.app_page)),
                color="#9A9A9A",
            )

            # A column that contains only the title (spaced nicely)
            content_column = ft.Column(
                controls=[
                    title,
                    subtitle,
                    ft.Container(height=int(tabs_script.sv(8, self.app_page))),  # spacing under title

                ],
                expand=True,
                spacing=int(tabs_script.sv(10, self.app_page)),
                horizontal_alignment=ft.CrossAxisAlignment.START
            )

            # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            # |                                                                                                                                                                        |
            # |                                                                       UI  SPEC                                                                                    |
            # |                                                                                                                                                                        |
            # |------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            if name == "Dashboard":
                youtube_data = None

                # To make new widgets:
                # - Define them below
                # - Shove them in a row
                # - If you need a stack, add.

                # To add new buttons to the widgets:
                # - Define all widgets, add them to the layout.
                # - Define the button using button_template()
                # - Append it to the row
                # - Then append the layout
                # eg, see line "if name == "Alerts":"

                try:
                    youtube_data = self.fetch_comprehensive_channel_data(days_back=30, max_videos=10,
                                                                         max_comments_per_video=5)
                    subscriber_growth_data = youtube_data['subscriber_growth']['points']
                    if not subscriber_growth_data:
                        subscriber_growth_data = [(0, 0)]
                except Exception as e:
                    print(f"Error fetching subscriber data: {e}")
                    youtube_data = None
                    subscriber_growth_data = [(0, 2), (1, 4), (2, 3), (3, 6)]
                subscriber_growth = graph_switch(
                    data=subscriber_growth_data,
                    title="Subscriber Growth",
                    x_axis_title="Time",
                    y_axis_title="Value",
                    height=int(tabs_script.sv(250, self.app_page)),
                    width=int(tabs_script.sh(620, self.app_page)),
                    initial_type="line",
                )

                try:
                    latest_video = youtube_data.get('latest_video')
                    if latest_video:
                        # Store latest video data for predictor
                        self.latest_video_data = latest_video
                        thumbnail_url_data = latest_video['thumbnail']
                        title_data = latest_video['title']
                        views_data = latest_video['view_count']
                        # Placeholder - will be updated by predictor
                        growth_score_data = "Calculating..."
                        comments_data = [comment['text'][:100] for comment in latest_video.get('comments', [])]
                        if not comments_data:
                            comments_data = ["No comments yet"]
                    else:
                        self.latest_video_data = None
                        thumbnail_url_data = "https://i.ytimg.com/vi/KEpNe2M0qJE/maxresdefault.jpg"
                        title_data = "No videos found"
                        views_data = 0
                        growth_score_data = "N/A"
                        comments_data = ["No videos to display"]
                except Exception as e:
                    print(f"Error: {e}")
                    self.latest_video_data = None
                    thumbnail_url_data = "https://i.ytimg.com/vi/KEpNe2M0qJE/maxresdefault.jpg"
                    title_data = "Error loading data"
                    views_data = 0
                    growth_score_data = "N/A"
                    comments_data = ["Error loading"]

                self.latest_video_growth_score_text = ft.Text(
                    f"Overall Growth Score: {growth_score_data}",
                    color=FG,
                    size=int(tabs_script.s(14, self.app_page)),
                )

                latest_video_stats = base_template(
                    width=int(tabs_script.sh(270, self.app_page)),
                    height=int(tabs_script.sv(450, self.app_page)),
                    page=self.app_page,
                    children=[
                        ft.Text(
                            "Latest video stats",
                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),

                        ft.Container(
                            ft.Image(
                                src=thumbnail_url_data,
                                width=320,
                                height=180,
                                fit=ft.ImageFit.COVER,
                            ),
                            width=float("inf"),
                            height=180,
                            bgcolor=DARK_GREY,
                            border_radius=12,
                            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                        ),

                        ft.Text(
                            f"Title: {title_data}",
                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),

                        ft.Text(
                            f"Views: {views_data}",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),

                        self.latest_video_growth_score_text,

                        ft.Column(
                            controls=[
                                ft.Container(
                                    content=ft.Text(item, color=FG),
                                    bgcolor=BG,
                                    padding=12,
                                    border_radius=12,
                                    border=ft.border.all(1, ORANGE),
                                ) for item in comments_data
                            ],
                            spacing=8,
                            scroll=ft.ScrollMode.AUTO,
                            expand=True,
                        )
                    ],
                )

                if self.latest_video_data:
                    try:
                        import threading
                        thread = threading.Thread(
                            target=self._run_predictor_for_latest_video,
                            daemon=True
                        )
                        thread.start()
                    except Exception as e:
                        print(f"[Dashboard] Failed to start predictor thread: {e}")
                try:
                    revenue_data = youtube_data.get('revenue', {})

                    # Check if revenue is available
                    if revenue_data.get('available') == False:
                        # Show "Not Available" message instead of graph
                        revenue = base_template(
                            width=int(tabs_script.sh(620, self.app_page)),
                            height=int(tabs_script.sv(190, self.app_page)),
                            page=self.app_page,
                            children=[
                                ft.Text(
                                    "Revenue Data",
                                    size=int(tabs_script.s(16, self.app_page)),
                                    weight="bold",
                                    color=FG
                                ),
                                ft.Container(height=int(tabs_script.sv(20, self.app_page))),
                                ft.Text(
                                    "Not Available",
                                    size=int(tabs_script.s(14, self.app_page)),
                                    color="#888888",
                                    text_align=ft.TextAlign.CENTER
                                ),
                                ft.Container(height=int(tabs_script.sv(5, self.app_page))),
                                ft.Text(
                                    "Channel may not be monetized",
                                    size=int(tabs_script.s(11, self.app_page)),
                                    color="#666666",
                                    text_align=ft.TextAlign.CENTER
                                ),
                            ],
                        )
                    elif not revenue_data.get('label_points') or not revenue_data.get('has_data', True):
                        # Show "No Data" message
                        revenue = base_template(
                            width=int(tabs_script.sh(620, self.app_page)),
                            height=int(tabs_script.sv(190, self.app_page)),
                            page=self.app_page,
                            children=[
                                ft.Text(
                                    "Revenue Data",
                                    size=int(tabs_script.s(16, self.app_page)),
                                    weight="bold",
                                    color=FG
                                ),
                                ft.Container(height=int(tabs_script.sv(20, self.app_page))),
                                ft.Text(
                                    "No Revenue Data",
                                    size=int(tabs_script.s(14, self.app_page)),
                                    color="#888888",
                                    text_align=ft.TextAlign.CENTER
                                ),
                                ft.Container(height=int(tabs_script.sv(5, self.app_page))),
                                ft.Text(
                                    "No earnings in selected period",
                                    size=int(tabs_script.s(11, self.app_page)),
                                    color="#666666",
                                    text_align=ft.TextAlign.CENTER
                                ),
                            ],
                        )
                    else:
                        # Show actual revenue graph
                        revenue_points = revenue_data['label_points']
                        revenue = graph_switch(
                            data=revenue_points,
                            title="Revenue Data",
                            x_axis_title="Time",
                            y_axis_title="Value",
                            width=int(tabs_script.sh(620, self.app_page)),
                            height=int(tabs_script.sv(190, self.app_page)),
                            initial_type="pie",
                        )
                except Exception as e:
                    print(f"Error creating revenue widget: {e}")
                    # Fallback to "Not Available"
                    revenue = base_template(
                        width=int(tabs_script.sh(620, self.app_page)),
                        height=int(tabs_script.sv(190, self.app_page)),
                        page=self.app_page,
                        children=[
                            ft.Text("Revenue Data", size=int(tabs_script.s(16, self.app_page)), weight="bold",
                                    color=FG),
                            ft.Container(height=int(tabs_script.sv(20, self.app_page))),
                            ft.Text("Not Available", size=int(tabs_script.s(14, self.app_page)), color="#888888"),
                        ],
                    )

                chat_history_children = [
                    ft.Text(
                        "Recent Chat History",
                        weight="bold",
                        color=FG,
                        size=int(tabs_script.s(14, self.app_page)),
                    )
                ]

                # Load chat history from cache
                try:
                    state = tabs_script.read_tab_state(self.tab_id) or {}
                    all_messages = []

                    # Load from old format for backwards compatibility
                    old_history = state.get("chat_history", []) or []
                    if old_history:
                        all_messages.extend(old_history)

                    # Load all chat blocks (chat_1, chat_2, chat_3, etc.)
                    chat_blocks = {}
                    for key in state.keys():
                        if key.startswith("chat_"):
                            try:
                                block_num = int(key.split("_")[1])
                                chat_blocks[block_num] = state[key]
                            except Exception:
                                pass

                    # Add messages from all chat blocks in order
                    if chat_blocks:
                        for block_num in sorted(chat_blocks.keys()):
                            block_messages = chat_blocks[block_num]
                            if isinstance(block_messages, list):
                                all_messages.extend(block_messages)

                    # Display recent messages (limit to last 5 for the dashboard)
                    recent_messages = all_messages[-5:] if len(all_messages) > 5 else all_messages

                    if recent_messages:
                        for msg in recent_messages:
                            role = msg.get("role", "assistant")
                            content = msg.get("content", "")
                            # Truncate long messages for dashboard display
                            display_content = content[:40] + "..." if len(content) > 40 else content
                            chat_history_children.append(
                                ft.Text(f"• [{role}] {display_content}", color=FG,
                                        size=int(tabs_script.s(12, self.app_page)))
                            )
                    else:
                        chat_history_children.append(ft.Text("• No chat history yet", color=FG))

                except Exception as e:
                    print(f"Failed to load chat history for dashboard: {e}")
                    chat_history_children.append(ft.Text("• Error loading history", color=FG))

                # Create the chat history widget with dynamic content
                chat_history = base_template(
                    width=int(tabs_script.sh(292, self.app_page)),
                    height=int(tabs_script.sv(190, self.app_page)),
                    page=self.app_page,
                    children=chat_history_children,
                )

                # Preserve graph references
                self.dashboard_graph_widget = subscriber_growth
                self.dashboard_graph_funcs = getattr(subscriber_growth, "_graph_funcs", None)

                # -----------Layout

                top_row = ft.Row(
                    controls=[subscriber_growth],
                    spacing=int(tabs_script.s(12, self.app_page)),
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )

                bottom_row = ft.Row(
                    controls=[revenue],
                    spacing=int(tabs_script.s(12, self.app_page)),
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )

                # layout

                dashboard_layout = ft.Stack(
                    controls=[
                        ft.Column(
                            controls=[
                                top_row,
                                ft.Container(height=int(tabs_script.sv(12, self.app_page))),
                                bottom_row,

                            ],
                            spacing=0,
                        ),

                        ft.Container(
                            content=latest_video_stats,
                            top=0,
                            right=0,
                        ),
                    ],
                    expand=True,
                )

                content_column.controls.append(dashboard_layout)

            elif name == "Videos":

                youtube_data = None
                try:
                    youtube_data = self.fetch_comprehensive_channel_data(days_back=30, max_videos=50)

                except Exception as e:

                    print(f"Error fetching YouTube data for Videos page: {e}")
                    youtube_data = None

                # Create simple display area
                if self.selected_video is None:
                    # Simple placeholder
                    display_children = [
                        ft.Text("No Video Selected", color=FG, size=14),
                        ft.Text("Click a video to view stats", color="#888888", size=12),
                    ]
                else:
                    # Simple stats display
                    stats = self.selected_video_stats
                    display_children = [
                        ft.Text("Selected Video", color=FG, size=14),
                        ft.Text(f"Title: {stats.get('title', 'Unknown')[:40]}", color=FG, size=12),
                        ft.Text(f"Views: {stats.get('view_count', 0):,}", color=FG, size=12),
                        ft.Text(f"Age: {stats.get('video_age_days', 0)} days", color=FG, size=12),
                        ft.Text(f"Growth: {stats.get('views_per_day', 0):.1f} views/day", color=FG, size=12),
                    ]

                ai_summary = base_template(
                    height=int(tabs_script.sv(460, self.app_page)),
                    width=int(tabs_script.sh(270, self.app_page)),
                    page=self.app_page,
                    children=display_children,
                )

                # Store reference for updates
                self.selected_video_display = ai_summary

                right_widgets_container = ft.Row(
                    controls=[
                        ai_summary,
                    ],
                    spacing=int(tabs_script.s(12, self.app_page)),
                )

                # alerts
                growth_chart_advanced = ft.Column(
                    controls=[
                        base_template(
                            height=int(tabs_script.sv(400, self.app_page)),
                            width=int(tabs_script.sh(580, self.app_page)),
                            page=self.app_page,
                            children=[
                                ft.Text(
                                    "Growth Chart - needs same graph type as before and ability to plot and zoom in place",
                                    color=FG,
                                    size=int(tabs_script.s(14, self.app_page)),
                                )
                            ],
                        ),

                    ],
                    spacing=int(tabs_script.sv(12, self.app_page)),
                    expand=True,
                )

                # layout

                advanced_view = base_template(
                    height=int(tabs_script.sv(460, self.app_page)),
                    width=int(tabs_script.sh(910, self.app_page)),
                    page=self.app_page,
                    children=[
                        ft.Text(
                            "Advanced View",
                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),

                        ft.Row(
                            controls=[
                                growth_chart_advanced,
                                right_widgets_container,
                            ],
                            spacing=int(tabs_script.s(12, self.app_page)),
                            expand=True,
                        ),
                    ],
                )

                try:
                    predicted_growth_data = 0  # fix
                    if not predicted_growth_data:
                        predicted_growth_data = [(0, 0)]
                    latest_video = youtube_data.get('latest_video')  # fix @ 8pm
                    if latest_video:
                        thumbnail_url_data = latest_video['thumbnail']
                        avg_views = youtube_data['stats']['average_views_per_video']
                        video_views = latest_video['view_count']
                    else:
                        thumbnail_url_data = "https://i.ytimg.com/vi/KEpNe2M0qJE/maxresdefault.jpg"
                        ai_summary_data = ["No videos to analyze"]
                except Exception as e:
                    predicted_growth_data = [(0, 2), (1, 4), (2, 3), (3, 6)]
                    thumbnail_url_data = "https://i.ytimg.com/vi/KEpNe2M0qJE/maxresdefault.jpg"
                    ai_summary_data = ["Error loading AI insights"]
                ai_summary_data = ["Select a video to see AI analysis."]

                # Create thumbnail image control and store reference for later updates
                self.upload_thumbnail_image = ft.Image(
                    src=thumbnail_url_data,
                    width=270,
                    height=150,
                    fit=ft.ImageFit.COVER,
                )

                self.ai_explanation_container = base_template(
                    width=int(tabs_script.sh(580, self.app_page)),
                    height=int(tabs_script.sv(150, self.app_page)),
                    page=self.app_page,
                    children=[
                        ft.Text(
                            "AI Explanation",
                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    item,
                                    color=FG,
                                    size=int(tabs_script.s(14, self.app_page)),
                                )
                                for item in ai_summary_data
                            ],
                            spacing=int(tabs_script.sv(8, self.app_page)),
                            scroll=ft.ScrollMode.AUTO,
                            expand=True,
                        ),
                    ],
                )

                self.predicted_growth_graph = graph_switch(
                    data=predicted_growth_data,
                    title="Predicted Growth",
                    x_axis_title="Time",
                    y_axis_title="Value",
                    width=int(tabs_script.sh(270, self.app_page)),
                    height=int(tabs_script.sv(150, self.app_page)),
                    initial_type="line",
                )
                # ADD THESE TWO LINES:
                self.predicted_growth_graph_funcs = getattr(self.predicted_growth_graph, "_graph_funcs", None)
                print(f"[DEBUG] Stored predicted_growth_graph_funcs: {self.predicted_growth_graph_funcs}")
                ai_summary_upload = ft.Column(
                    controls=[
                        base_template(
                            height=int(tabs_script.sv(350, self.app_page)),
                            width=int(tabs_script.sh(580, self.app_page)),
                            page=self.app_page,
                            children=[

                                ft.Row(
                                    controls=[
                                        self.predicted_growth_graph,
                                        ft.Container(
                                            content=self.upload_thumbnail_image,
                                            width=270,
                                            height=150,
                                            bgcolor=DARK_GREY,
                                            border_radius=12,
                                            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                                        ),

                                    ],
                                    alignment=ft.MainAxisAlignment.START,
                                    spacing=10,
                                ),

                                self.ai_explanation_container,

                            ],
                        ),

                    ],
                    spacing=int(tabs_script.sv(12, self.app_page)),
                    expand=True,
                )

                # -------------------- Plan Video View Layout ---------------------------------------

                video_ui = self._make_plan_tab_content()

                self.tabs = tab_template_advanced(
                    initial_tabs=[
                        ("Video", video_ui),
                    ],
                    tab_content_template=video_ui,
                    page=self.app_page,
                    width=900,
                    height=500,
                    on_tabs_changed=self._save_plan_tabs,
                )
                self.plan_tabs_widget = self.tabs

                # Initialize plan tab counter from cache
                if not hasattr(self, "_plan_tab_counter"):
                    state = tabs_script.read_tab_state(self.tab_id) or {}
                    max_block = 0
                    for key in state.keys():
                        if key.startswith("tabplan_"):
                            try:
                                block_num = int(key.split("_")[1])
                                max_block = max(max_block, block_num)
                            except Exception:
                                pass
                    self._plan_tab_counter = max_block
                    print(f"Initialized plan tab counter: {self._plan_tab_counter}")

                try:
                    widget = self.plan_tabs_widget

                    # 1) If widget exposes add_tab, wrap it so we save after creation.
                    add_fn = getattr(widget, "add_tab", None)
                    if callable(add_fn):
                        orig_add = add_fn

                        def _wrapped_add(*args, **kwargs):
                            # forward to original add_tab and then persist tabs
                            res = orig_add(*args, **kwargs)
                            try:
                                self._save_plan_tabs()
                            except Exception:
                                pass
                            return res

                        try:
                            widget.add_tab = _wrapped_add
                        except Exception:
                            # some proxy objects may not allow attribute assignment; ignore
                            pass

                    # 2) Hook tab-change events when available to persist on selection/edit
                    if hasattr(widget, "on_change"):
                        orig_on_change = getattr(widget, "on_change", None)

                        def _on_change(e):
                            try:
                                if callable(orig_on_change):
                                    orig_on_change(e)
                            except Exception:
                                pass
                            try:
                                self._save_plan_tabs()
                            except Exception:
                                pass

                        try:
                            widget.on_change = _on_change
                        except Exception:
                            pass
                    elif hasattr(widget, "on_tab_change"):
                        orig_tab_change = getattr(widget, "on_tab_change", None)

                        def _on_tab_change(e):
                            try:
                                if callable(orig_tab_change):
                                    orig_tab_change(e)
                            except Exception:
                                pass
                            try:
                                self._save_plan_tabs()
                            except Exception:
                                pass

                        try:
                            widget.on_tab_change = _on_tab_change
                        except Exception:
                            pass

                    # 3) If widget exposes a .tabs list but you cannot intercept append,
                    #    ensure any TextField you create for tab content calls _save_plan_tabs()
                    #    on on_change/on_blur (this is handled in _restore_plan_tabs already).
                except Exception:
                    pass

                # restore previously saved plan tabs (reads 'plan_tabs' from tab state)
                self._restore_plan_tabs()

                # ensure tab selection change triggers saving of plan tabs
                try:
                    try:
                        if hasattr(self.tabs, "on_change"):
                            orig_on_change = getattr(self.tabs, "on_change", None)

                            def _tabs_on_change(e):
                                try:
                                    if callable(orig_on_change):
                                        orig_on_change(e)
                                except Exception:
                                    pass
                                try:
                                    self._save_plan_tabs()
                                except Exception:
                                    pass

                            self.tabs.on_change = _tabs_on_change
                        elif hasattr(self.tabs, "on_tab_change"):
                            orig_tab_change = getattr(self.tabs, "on_tab_change", None)

                            def _tabs_tab_change(e):
                                try:
                                    if callable(orig_tab_change):
                                        orig_tab_change(e)
                                except Exception:
                                    pass
                                try:
                                    self._save_plan_tabs()
                                except Exception:
                                    pass

                            self.tabs.on_tab_change = _tabs_tab_change
                        else:
                            # fallback: attach click/save to tab headers if possible
                            for _tb in getattr(self.tabs, "tabs", []) or []:
                                hdr = getattr(_tb, "header", None) or getattr(_tb, "label", None)
                                if hdr and hasattr(hdr, "on_click"):
                                    orig = getattr(hdr, "on_click", None)

                                    def _wrap(e, o=orig):
                                        try:
                                            if callable(o):
                                                o(e)
                                        except Exception:
                                            pass
                                        try:
                                            self._save_plan_tabs()
                                        except Exception:
                                            pass

                                    hdr.on_click = _wrap
                    except Exception:
                        pass
                except Exception:
                    pass



                # ---------------------------------------- Upload Video View Layout ---------------------------------------

                try:
                    if youtube_data is None:
                        youtube_data = self.fetch_comprehensive_channel_data(days_back=30, max_videos=50)

                    all_videos_list = youtube_data.get('all_videos', [])

                    # Separate videos and shorts
                    videos_only = [v for v in all_videos_list if not v.get('is_short', False)]
                    shorts_only = [v for v in all_videos_list if v.get('is_short', False)]

                except Exception as e:
                    print(f"Error loading videos: {e}")
                    all_videos_list = []
                    videos_only = []
                    shorts_only = []

                # Build Videos tab content
                if videos_only:
                    videos_controls = []
                    for video in videos_only:
                        # Simple clickable container
                        video_item = ft.Container(
                            content=ft.Row(
                                controls=[
                                    base_template(
                                        height=int(tabs_script.sv(100, self.app_page)),
                                        width=int(tabs_script.sh(600, self.app_page)),
                                        page=self.app_page,
                                        children=[
                                            ft.Column(
                                                controls=[
                                                    ft.Text(
                                                        video.get('title', 'Untitled'),
                                                        weight="bold",
                                                        color=FG,
                                                        size=int(tabs_script.s(14, self.app_page)),
                                                    ),
                                                    ft.Text(
                                                        f"Views: {video.get('view_count', 0):,} | Likes: {video.get('like_count', 0):,}",
                                                        color=FG,
                                                        size=int(tabs_script.s(12, self.app_page)),
                                                    ),
                                                ],
                                                spacing=4,
                                            )
                                        ],
                                    ),
                                    base_template(
                                        height=int(tabs_script.sv(100, self.app_page)),
                                        width=int(tabs_script.sh(200, self.app_page)),
                                        page=self.app_page,
                                        children=[
                                            ft.Container(
                                                content=ft.Image(
                                                    src=video.get('thumbnail', ''),
                                                    width=180,
                                                    height=90,
                                                    fit=ft.ImageFit.COVER,
                                                ),
                                                width=180,
                                                height=90,
                                                bgcolor=DARK_GREY,
                                                border_radius=8,
                                                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                                            )
                                        ],
                                    )
                                ],
                                spacing=10,
                            ),

                            on_click=lambda e, v=video: (
                                # store clicked video as a dict
                                setattr(
                                    self,
                                    "last_clicked_video",
                                    dict(v) if isinstance(v, dict) else {"video": v}
                                ),

                                # Close popup using the close function stored in upload_video_view.data
                                (hasattr(self, 'upload_video_view') and
                                 hasattr(self.upload_video_view, 'data') and
                                 isinstance(self.upload_video_view.data, dict) and
                                 self.upload_video_view.data.get('close_popup') and
                                 self.upload_video_view.data['close_popup'](e)),

                                # Update upload thumbnail image
                                (hasattr(self, 'upload_thumbnail_image') and
                                 setattr(self.upload_thumbnail_image, 'src', v.get('thumbnail', '')) and
                                 self.upload_thumbnail_image.update()),

                                # preserve existing behavior
                                print(f"Clicked video: {v}"),
                                self._on_video_click(v, youtube_data)
                            )

                        )
                        videos_controls.append(video_item)

                    videos_page_content = ft.Column(
                        controls=videos_controls,
                        spacing=10,
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,
                    )
                else:
                    videos_page_content = base_template(
                        height=int(tabs_script.sv(200, self.app_page)),
                        width=int(tabs_script.sh(800, self.app_page)),
                        page=self.app_page,
                        children=[
                            ft.Text(
                                "No videos found",
                                color=FG,
                                size=int(tabs_script.s(16, self.app_page)),
                                weight="bold",
                            ),
                            ft.Text(
                                "Upload your first video to get started",
                                color="#888888",
                                size=int(tabs_script.s(14, self.app_page)),
                            ),
                        ],
                    )

                # Build Shorts tab content
                if shorts_only:
                    shorts_controls = []
                    for short in shorts_only:
                        # Simple clickable container
                        short_item = ft.Container(
                            content=ft.Row(
                                controls=[
                                    base_template(
                                        height=int(tabs_script.sv(100, self.app_page)),
                                        width=int(tabs_script.sh(600, self.app_page)),
                                        page=self.app_page,
                                        children=[
                                            ft.Column(
                                                controls=[
                                                    ft.Text(
                                                        short.get('title', 'Untitled'),
                                                        weight="bold",
                                                        color=FG,
                                                        size=int(tabs_script.s(14, self.app_page)),
                                                    ),
                                                    ft.Text(
                                                        f"Views: {short.get('view_count', 0):,} | Likes: {short.get('like_count', 0):,}",
                                                        color=FG,
                                                        size=int(tabs_script.s(12, self.app_page)),
                                                    ),
                                                ],
                                                spacing=4,
                                            )
                                        ],
                                    ),
                                    base_template(
                                        height=int(tabs_script.sv(100, self.app_page)),
                                        width=int(tabs_script.sh(200, self.app_page)),
                                        page=self.app_page,
                                        children=[
                                            ft.Container(
                                                content=ft.Image(
                                                    src=short.get('thumbnail', ''),
                                                    width=180,
                                                    height=90,
                                                    fit=ft.ImageFit.COVER,
                                                ),
                                                width=180,
                                                height=90,
                                                bgcolor=DARK_GREY,
                                                border_radius=8,
                                                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                                            )
                                        ],
                                    )
                                ],
                                spacing=10,
                            ),
                            on_click=lambda e, s=short: (
                                # store clicked short as a dict
                                setattr(
                                    self,
                                    "last_clicked_video",
                                    dict(s) if isinstance(s, dict) else {"video": s}
                                ),

                                # Close popup using the close function stored in upload_video_view.data
                                (hasattr(self, 'upload_video_view') and
                                 hasattr(self.upload_video_view, 'data') and
                                 isinstance(self.upload_video_view.data, dict) and
                                 self.upload_video_view.data.get('close_popup') and
                                 self.upload_video_view.data['close_popup'](e)),

                                # Update upload thumbnail image
                                (hasattr(self, 'upload_thumbnail_image') and
                                 setattr(self.upload_thumbnail_image, 'src', s.get('thumbnail', '')) and
                                 self.upload_thumbnail_image.update()),

                                # preserve existing behavior
                                print(f"Clicked video: {s}"),
                                self._on_video_click(s, youtube_data)
                            )

                        )
                        shorts_controls.append(short_item)

                    shorts_page_content = ft.Column(
                        controls=shorts_controls,
                        spacing=10,
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,
                    )
                else:
                    shorts_page_content = base_template(
                        height=int(tabs_script.sv(200, self.app_page)),
                        width=int(tabs_script.sh(800, self.app_page)),
                        page=self.app_page,
                        children=[
                            ft.Text(
                                "No shorts found",
                                color=FG,
                                size=int(tabs_script.s(16, self.app_page)),
                                weight="bold",
                            ),
                            ft.Text(
                                "Upload your first short to get started",
                                color="#888888",
                                size=int(tabs_script.s(14, self.app_page)),
                            ),
                        ],
                    )

                local_content_debug1 = base_template(
                    height=int(tabs_script.sv(300, self.app_page)),
                    width=int(tabs_script.sh(900, self.app_page)),
                    page=self.app_page,
                    children=[
                        ft.Text(
                            "This version currently does not support frame-by-frame analysis, therefore this function cannot work. :(",

                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),

                    ],
                )



                videos_tab = ft.Container(
                    content=videos_page_content,
                    padding=12
                )

                shorts_tab = ft.Container(
                    content=shorts_page_content,
                    padding=12
                )

                local_tab = ft.Container(
                    content=local_content_debug1,
                    padding=12
                )



                tabs_container = tab_template_simple(
                    [
                        ("Videos", videos_tab),
                        ("Shorts", shorts_tab),
                        ("Locally stored", local_tab),

                    ],
                    page=self.app_page,
                    width=900,
                    height=500
                )

                upload_video_view = base_template(
                    height=int(tabs_script.sv(440, self.app_page)),
                    width=int(tabs_script.sh(900, self.app_page)),
                    page=self.app_page,
                    children=[
                        tabs_container
                    ],
                )

                # Store reference so video click handlers can access close_popup
                self.upload_video_view = upload_video_view

                upload_view = base_template(
                    height=int(tabs_script.sv(440, self.app_page)),
                    width=int(tabs_script.sh(900, self.app_page)),
                    page=self.app_page,
                    children=[

                        ft.Row(
                            controls=[
                                ai_summary_upload,
                            ],
                            spacing=int(tabs_script.s(12, self.app_page)),
                            expand=True,
                        ),
                    ],
                )

                # ----------------------------------------  ---------------------------------------

                # create empty bottom_row placeholder so we can create the host Stack next
                bottom_row = ft.Row(
                    controls=[],
                    spacing=int(tabs_script.s(12, self.app_page)),
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )

                top_row_1 = ft.Column(
                    controls=[
                        upload_view,

                    ],
                )

                # create the host Stack that spans the full area (this will host the popup container)
                layout = ft.Stack(
                    controls=[
                        ft.Column(
                            controls=[

                                top_row_1,
                            ],
                            spacing=0,
                        ),

                    ],
                    expand=True,
                )

                # Now create the button and pass the host=layout so the popup is appended into the host Stack
                click_for_advanced_view = button_template(
                    title="Click for advanced view",
                    base=advanced_view,
                    width=int(tabs_script.sh(890, self.app_page)),
                    height=int(tabs_script.sv(42, self.app_page)),
                    page=self.app_page,
                    host=layout,  # <-- critical: host must be the Stack that spans the content area
                )



                upload_a_video_view = button_template(
                    "Choose a video, upload one or create a plan.",
                    base=upload_video_view,
                    width=250,
                    height=350,
                    page=self.app_page,
                    host=layout,
                    x_offset=0,
                    y_offset=0,
                )
                self.upload_a_video_view = upload_a_video_view

                # append the button into bottom_row now that host/layout exists
                # append the button into bottom_row now that host/layout exists
                bottom_row.controls.append(click_for_advanced_view)

                upload_with_button = ft.Stack(
                    controls=[

                        upload_view,


                        ft.Container(
                            content=upload_a_video_view,
                            top=15,
                            right=30,
                        ),
                    ],
                    expand=False,
                    clip_behavior=ft.ClipBehavior.NONE,
                )

                top_row_1.controls[0] = upload_with_button

                content_column.controls.append(layout)


            elif name == "Chatbot":

                initial_messages = None

                try:

                    state = tabs_script.read_tab_state(self.tab_id) or {}

                    # Find all chat blocks (chat_1, chat_2, chat_3, etc.)

                    chat_blocks = {}
                    max_block_num = 0
                    for key in state.keys():
                        if key.startswith("chat_"):

                            try:

                                block_num = int(key.split("_")[1])
                                chat_blocks[block_num] = state[key]
                                max_block_num = max(max_block_num, block_num)

                            except Exception:
                                pass

                    # Load all chat blocks in order

                    converted = []

                    if chat_blocks:
                        for block_num in sorted(chat_blocks.keys()):
                            block_messages = chat_blocks[block_num]
                            if isinstance(block_messages, list):
                                for item in block_messages:

                                    try:

                                        role = (item.get("role") or "").lower()
                                        content = item.get("content") or ""

                                        if role == "user":
                                            converted.append(f"User: {content}")
                                        else:
                                            converted.append(f"Bot: {content}")

                                    except Exception:

                                        # fallback: stringify

                                        try:
                                            converted.append(f"Bot: {json.dumps(item)}")
                                        except Exception:
                                            converted.append(str(item))

                    if converted:

                        initial_messages = converted
                        # Initialize counters to track what's already loaded
                        self._last_saved_msg_count = len(converted)
                        self._chat_block_counter = max_block_num

                    else:

                        self._last_saved_msg_count = 0
                        self._chat_block_counter = 0


                except Exception as e:

                    print(f"Failed to load chat history: {e}")
                    initial_messages = None
                    self._last_saved_msg_count = 0
                    self._chat_block_counter = 0

                self.chatbot_host_controls = {}

                self.chatbot_container = ft.Container(
                    content=chatbot_template(
                        width=900,
                        height=430,
                        page=self.app_page,
                        on_message=self._save_chat_history,
                        initial_messages=initial_messages,
                        tab_id=self.tab_id,
                        on_new_chat_callback=self._start_new_chat_session,
                        on_attach=self._on_chat_attach_click,
                        on_generate_reply=self._on_generate_reply,
                        host_controls=self.chatbot_host_controls,
                    ),
                    width=900,
                    height=430,
                    padding=ft.padding.all(int(tabs_script.s(12, self.app_page))),
                )

                content_column.controls.append(self.chatbot_container)

                self._restore_chat_history()



            elif name == "Info":

                general_info_list = SulfurAI.retrieve_app_info()

                general_info = base_template(
                    width=int(tabs_script.sh(905, self.app_page)),
                    height=int(tabs_script.sv(200, self.app_page)),
                    page=self.app_page,
                    children=[
                        ft.Text(
                            "General Info",
                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),
                        ft.Container(
                            content=ft.ListView(
                                spacing=10,
                                padding=10,
                                auto_scroll=False,
                                controls=[
                                    ft.Text(str(item), color=FG, size=int(tabs_script.s(12, self.app_page)))
                                    for item in general_info_list
                                ],
                            ),
                            expand=True,
                        ),
                    ],
                )

                model_list = SulfurAI.retrieve_current_models()
                all_models = base_template(
                    width=int(tabs_script.sh(905, self.app_page)),
                    height=int(tabs_script.sv(200, self.app_page)),
                    page=self.app_page,
                    children=[
                        ft.Text(
                            "All models Used",
                            weight="bold",
                            color=FG,
                            size=int(tabs_script.s(14, self.app_page)),
                        ),
                        ft.Container(
                            content=ft.Row(
                                scroll=ft.ScrollMode.AUTO,
                                spacing=10,
                                controls=[
                                    ft.Text(str(item), color=FG, size=int(tabs_script.s(12, self.app_page)))
                                    for item in model_list
                                ],
                            ),
                            expand=True,
                        ),
                    ],
                )

                gibl = ft.Row(
                    controls=[general_info],
                    spacing=int(tabs_script.s(12, self.app_page)),
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )

                tabs_content_debug1 = ft.Row(
                    controls=[
                        base_template(  # REVAMP
                            width=int(tabs_script.sh(590, self.app_page)),
                            height=int(tabs_script.sv(200, self.app_page)),
                            page=self.app_page,
                            children=[
                                ft.Text(
                                    "csv list of data",
                                    weight="bold",
                                    color=FG,
                                    size=int(tabs_script.s(14, self.app_page)),
                                ),
                                ft.Text(
                                    "ability to filter out, modify or remove entries",
                                    color=FG
                                ),
                            ],
                        ),

                    ],
                    spacing=int(tabs_script.s(12, self.app_page)),

                )

                sentence_tab = ft.Container(
                    content=
                    tabs_content_debug1,
                    padding=12
                )

                dataprofiles_tab = ft.Container(
                    content=tabs_content_debug1,
                    padding=12
                )

                general_tab = ft.Container(
                    content=tabs_content_debug1,
                    padding=12
                )

                tabs_container = tab_template_simple(
                    [
                        ("Sentences", sentence_tab),
                        ("Dataprofiles", dataprofiles_tab),
                        ("General", general_tab),

                    ],
                    page=self.app_page,
                    width=900,
                    height=500
                )


                top_row_1 = ft.Row(
                    controls=[gibl],
                    spacing=int(tabs_script.s(12, self.app_page)),
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )

                bottom_row_1 = ft.Row(
                    controls=[all_models],
                    spacing=int(tabs_script.s(12, self.app_page)),
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )

                # create the host Stack that spans the full area (this will host the popup container)
                layout = ft.Stack(
                    controls=[
                        ft.Column(
                            controls=[
                                top_row_1,
                                ft.Container(height=int(tabs_script.sv(12, self.app_page))),
                                bottom_row_1,
                            ],
                            spacing=0,
                        ),

                    ],
                    expand=True,
                )

                content_column.controls.append(layout)

            # ---------------------------------------------------

            # The styled boxed container (same as before)
            page_container = ft.Container(
                content=ft.Column(
                    controls=[content_column],
                    scroll="auto",  # scrolling inside the boxed area
                    expand=True,  # <-- ESSENTIAL: give this outer Column height so inner expand works
                ),
                padding=ft.padding.all(int(tabs_script.s(14, self.app_page))),
                border=ft.border.all(1, ORANGE),
                border_radius=int(tabs_script.s(12, self.app_page)),
                bgcolor="#0d0d0d",
                expand=True,  # boxed fills available height
            )

            return page_container

        except Exception:
            # safe fallback: empty container
            try:
                return ft.Container()
            except Exception:
                return None

    def _build_main_content(self):
        """
        Attach the persistent inner page container for the currently selected
        inner tab to self.main_container.content.

        Behavior:
        - If a container for self.inner_selected exists in self.inner_pages,
          use it (preserves any custom edits you make later).
        - Otherwise create it via _create_inner_page(name) and store it.
        This replacement adds diagnostics to help find why Dashboard content
        exists but does not render.
        """
        try:
            name = self.inner_selected or "Dashboard"

            # If page for this inner tab already exists, reuse it.
            page_container = self.inner_pages.get(name)
            if page_container is None:
                # build it and cache
                page_container = self._create_inner_page(name)
                try:
                    self.inner_pages[name] = page_container
                except Exception:
                    # caching is optional, continue even on failure
                    pass

            # Attach the page container to the persistent main container
            try:
                self.main_container.content = page_container

            except Exception as e:
                print("DEBUG: Failed to set self.main_container.content:", e, traceback.format_exc())

            # Force an update on the Page (or fallback to updating the main_container)
            try:
                if getattr(self, "app_page", None) and hasattr(self.app_page, "update"):
                    try:
                        self.app_page.update()

                    except Exception as e:
                        print("DEBUG: app_page.update() raised:", e, traceback.format_exc())
                else:
                    try:
                        # update the nearest container we mutated
                        if hasattr(self.main_container, "update"):
                            self.main_container.update()

                    except Exception as e:
                        print("DEBUG: main_container.update() raised:", e, traceback.format_exc())
            except Exception:
                # last-resort - do nothing
                pass

        except Exception:
            # On failure, fall back to a minimal container (prevents restore crash)
            try:
                self.main_container.content = ft.Container()
                print("DEBUG: Exception occurred; set main_container.content to empty container")
            except Exception:
                pass

    def _on_inner_click(self, name, e=None):
        """Handle sidebar button clicks: set selection, persist, rebuild, update page and refresh sidebar styling."""
        try:
            self.inner_selected = name
            self._persist_inner_selection()
            # rebuild the main content for the new selection
            self._build_main_content()

            # update sidebar button visuals (we keep references in self.sidebar_buttons)
            try:
                for n, btn in (self.sidebar_buttons or {}).items():
                    try:
                        selected = (n == self.inner_selected)
                        btn.bgcolor = "#0f0f0f" if selected else BG
                        # update the inner Text weight if present
                        if hasattr(btn, "content") and isinstance(btn.content, ft.Text):
                            btn.content.weight = "w700" if selected else "w500"
                    except Exception:
                        pass
            except Exception:
                pass

            # update the page so both sidebar and main area refresh
            if e and hasattr(e, "page"):
                e.page.update()
            elif self.app_page:
                self.app_page.update()
        except Exception:
            pass

    def content(self):
        # restore cached inner selection & data only once
        self._restore_cache()

        td = self.token_data
        ui = self.user_info or {}

        # Sidebar sizing / items (keep the same variable names)
        text_size = int(tabs_script.s(14, self.app_page))
        left_pad = int(tabs_script.s(14, self.app_page))
        btn_height = int(tabs_script.sv(44, self.app_page))

        self.sidebar_items = [
            "Dashboard",
            "Videos",
            "Chatbot",
            "Info"
        ]

        # Title row (no icon version)
        title_row = ft.Row(
            [
                ft.Text("Channel", size=int(tabs_script.s(25, self.app_page)), weight="w700", color=FG),
                ft.Container(expand=True),
                ft.TextButton(
                    "≡",  # simple menu symbol
                    style=ft.ButtonStyle(color=FG),
                    on_click=lambda e: None
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0
        )

        # Search: searches the CURRENT visible page content (not the sidebar items)
        if not hasattr(self, "_search_dropdown_visible"):
            self._search_dropdown_visible = False
        if not hasattr(self, "_search_collapse_timer"):
            self._search_collapse_timer = None

        # column that will contain clickable result rows and support scroll
        search_results_column = ft.Column(controls=[], scroll="auto")

        # container that wraps the column; height toggles between tiny slither and expanded
        ITEM_HEIGHT = int(tabs_script.sv(44, self.app_page))  # approximate per-item height
        EXPANDED_HEIGHT = ITEM_HEIGHT * 2 + int(tabs_script.sv(8, self.app_page))  # max height for 2 items
        SLITHER_HEIGHT = int(tabs_script.sv(6, self.app_page))  # tiny visible slither when collapsed

        search_results_container = ft.Container(
            content=search_results_column,
            padding=ft.padding.only(top=int(tabs_script.sv(6, self.app_page)),
                                    bottom=int(tabs_script.sv(6, self.app_page))),
            border_radius=int(tabs_script.s(8, self.app_page)),
            border=ft.border.all(1, ORANGE),  # <<< ADD THIS LINE
            bgcolor=BG,
            height=SLITHER_HEIGHT,
            width=0,
        )

        # helpers to show/collapse the dropdown. We delay collapse slightly so clicks register.
        def _cancel_pending_collapse():
            try:
                t = getattr(self, "_search_collapse_timer", None)
                if t is not None:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                    self._search_collapse_timer = None
            except Exception:
                pass

        def _show_search_dropdown():
            _cancel_pending_collapse()
            self._search_dropdown_visible = True
            try:
                search_results_container.height = EXPANDED_HEIGHT
            except Exception:
                pass
            try:
                if self.app_page:
                    self.app_page.update()
            except Exception:
                pass

        def _collapse_search_dropdown(delay: float = 0.12):
            # schedule a collapse after a small delay to allow click handlers to run
            _cancel_pending_collapse()
            try:
                def _do_collapse():
                    try:
                        self._search_dropdown_visible = False
                        # keep the column contents (optional) but collapse height to tiny slither
                        search_results_container.height = SLITHER_HEIGHT
                        # optionally clear items (comment out if you want them preserved)
                        # search_results_column.controls = []
                        try:
                            if self.app_page:
                                self.app_page.update()
                        except Exception:
                            pass
                    except Exception:
                        pass

                t = threading.Timer(delay, _do_collapse)
                t.daemon = True
                t.start()
                self._search_collapse_timer = t
            except Exception:
                pass

        # Focus/Blur handlers for the text field
        def _on_search_focus(e):
            _show_search_dropdown()

        def _on_search_blur(e):
            # collapse shortly after losing focus (so clicks on results still fire)
            _collapse_search_dropdown(0.12)

        # Search/change handler (builds results and markers). It will also expand dropdown if needed.
        def _sidebar_search_changed(e):
            query = (e.control.value or "").strip().lower()
            self.sidebar_search = query

            state = tabs_script.read_tab_state(self.tab_id) or {}
            index = state.get("search_index")
            if index is None:
                # fallback: build a fresh index from the live controls
                index = self._build_search_index()

            # If query empty: clear results/markers and collapse dropdown
            if not query:
                self.search_results = []
                try:
                    self._mark_search_matches(query)
                except Exception:
                    pass
                try:
                    search_results_column.controls = []
                except Exception:
                    pass
                # collapse right away (or keep open if you prefer)
                _collapse_search_dropdown(0.0)
                try:
                    if self.app_page:
                        self.app_page.update()
                except Exception:
                    pass
                return

            # Find matching sidebar page names first
            results = []
            # Match page (tab) names from the cached index
            for name_lower in index.get("names", []):
                if query in name_lower:
                    # Map lower-case name back to original (e.g. "dashboard" -> "Dashboard")
                    for orig in self.sidebar_items:
                        if orig.lower() == name_lower:
                            results.append(orig)
            # Match content in each page’s index entries
            for name, page_entries in index.get("pages", {}).items():
                if name in results:
                    continue
                page_text = " ".join([str(e.get("title", "")) for e in (page_entries or [])])
                if query in page_text.lower():
                    results.append(name)

            # Find matching content in each indexed inner page
            # index["pages"][name] may be a list of entry-dicts (preferred) or a string (backcompat).
            for name, page_entries in index.get("pages", {}).items():
                if name in results:
                    continue
                page_text = ""
                try:
                    if isinstance(page_entries, list):
                        # each entry is a dict -> join titles
                        page_text = " ".join([str(e.get("title", "") or "") for e in page_entries])
                    elif isinstance(page_entries, str):
                        page_text = page_entries
                except Exception:
                    page_text = ""

                if page_text and query in page_text.lower():
                    results.append(name)

            # cap results and save
            MAX_RESULTS = 20
            results = results[:MAX_RESULTS]
            self.search_results = results

            # Build clickable rows (keep simple so height estimate works)
            res_controls = []
            for name in results:
                try:
                    item = ft.Container(
                        content=ft.Text(name, color=FG, size=int(tabs_script.s(14, self.app_page))),
                        padding=ft.padding.symmetric(horizontal=int(tabs_script.s(10, self.app_page)),
                                                     vertical=int(tabs_script.sv(8, self.app_page))),
                        on_click=(lambda n: (lambda ev: self._on_result_click(n, ev)))(name),
                        border_radius=int(tabs_script.s(8, self.app_page)),
                        expand=False,
                    )
                    res_controls.append(item)
                except Exception:
                    pass

            # Put them into the column (which scrolls), then ensure container expanded
            try:
                search_results_column.controls = res_controls
            except Exception:
                pass

            # show the dropdown (expanded to max 2 items)
            _show_search_dropdown()

            # place visual markers in main content (existing behavior)
            try:
                self._mark_search_matches(query)
            except Exception:
                pass

            # update UI
            try:
                if self.app_page:
                    self.app_page.update()
            except Exception:
                pass

        # Create TextField with focus/blur handlers
        search_bar = ft.TextField(
            hint_text="Search page...",
            value=getattr(self, "sidebar_search", "") or "",
            on_focus=_on_search_focus,
            on_blur=_on_search_blur,
            on_change=_sidebar_search_changed,
            border_radius=int(tabs_script.s(12, self.app_page)),
            content_padding=ft.padding.all(int(tabs_script.s(10, self.app_page))),
            height=int(tabs_script.sv(44, self.app_page)),
            color=FG,
            cursor_color=FG
        )

        # small visual break line between search and tabs
        divider = ft.Divider(thickness=1, color=FG)

        # Build / reuse sidebar buttons and store references so we can update styles
        buttons = []
        for name in self.sidebar_items:
            selected = (self.inner_selected == name)
            bg = "#0f0f0f" if selected else BG
            weight = "w700" if selected else "w500"

            # Try to reuse an existing stored button (preserves identity)
            existing = self.sidebar_buttons.get(name)
            if existing is not None:
                try:
                    existing.bgcolor = bg
                    if hasattr(existing, "content") and isinstance(existing.content, ft.Text):
                        existing.content.weight = weight
                except Exception:
                    pass
                btn = existing
            else:
                # New button: label stored as ft.Text inside the Container
                label = ft.Text(name, size=text_size, weight=weight, color=FG)
                btn = ft.Container(
                    content=label,
                    height=btn_height,
                    padding=ft.padding.only(left=left_pad, right=int(tabs_script.s(10, self.app_page))),
                    alignment=ft.alignment.center_left,
                    on_click=(lambda n: (lambda e: self._on_inner_click(n, e)))(name),
                    border_radius=int(tabs_script.s(10, self.app_page)),
                    bgcolor=bg,
                    expand=True
                )
                # store reference for future mutation
                try:
                    self.sidebar_buttons[name] = btn
                except Exception:
                    pass

            buttons.append(btn)

        sidebar_col = ft.Column(
            controls=[
                title_row,
                ft.Container(height=int(tabs_script.sv(8, self.app_page))),
                search_bar,
                search_results_container,
                ft.Container(height=int(tabs_script.sv(8, self.app_page))),
                ft.Container(height=int(tabs_script.sv(6, self.app_page))),
                *buttons
            ],
            spacing=int(tabs_script.sv(6, self.app_page)),
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )

        outer = ft.Container(
            width=250,
            bgcolor=BG,
            padding=ft.padding.all(int(tabs_script.s(28, self.app_page))),
            content=ft.Container(
                content=sidebar_col,
                border=ft.border.all(1, ORANGE),
                border_radius=int(tabs_script.s(16, self.app_page)),
                bgcolor=BG,
                expand=True,
                padding=ft.padding.all(int(tabs_script.s(14, self.app_page))),
            )
        )

        # ensure initial main content built
        if not getattr(self, "_tabs_cached", False):
            try:
                original = self.inner_selected or "Dashboard"
                # ensure we have a sensible starting selection
                self.inner_selected = original
                # ensure main content is present for original
                self._build_main_content()

                for name in self.sidebar_items:
                    try:
                        # switch the logical selection and build content — but do not call app.update()
                        self.inner_selected = name
                        self._build_main_content()
                    except Exception:
                        pass

                # restore original selection and build content
                self.inner_selected = original
                self._build_main_content()

                # build and persist the search index (this writes into tab_state if self.tab_id present)
                try:
                    self._build_search_index()
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                self._tabs_cached = True

        return ft.Row(
            expand=True,
            controls=[
                outer,
                ft.Container(width=int(tabs_script.s(14, self.app_page))),  # spacer
                ft.Container(expand=True, content=self.main_container)
            ],
        )