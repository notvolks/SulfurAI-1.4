import flet as ft
from flet.plotly_chart import PlotlyChart
from pathlib import Path
import shutil
import keyring
import traceback
import datetime

base = Path(__file__).resolve().parents[7]
CACHE_BASE = Path(base / "apps" / "SulfurAppAssets" / "cache" / "tabs")
CACHE_BASE.mkdir(parents=True, exist_ok=True)
from apps.SulfurAppAssets.globalvar import global_var
from apps.SulfurAppAssets.scripts.essential.ui import tabs_script

BG, FG, ORANGE, UI_BASE_WIDTH, UI_BASE_HEIGHT, UI_LOCK_MIN_WIDTH, UI_LOCK_MIN_HEIGHT, UI_MIN_SCALE, UI_MAX_SCALE, UI_WIDTH, UI_HEIGHT = global_var()


def chatbot_template(*, width=None, height=None, page=None, on_message=None,
                     initial_messages=None, tab_id=None, on_new_chat_callback=None,
                     on_attach=None, on_generate_reply=None, host_controls: dict = None):
    # -------------------------
    # Internal state
    # -------------------------
    chat_sessions = []  # list of {"id": int, "messages": [str, ...]}
    current_session_id = 1
    current_session_messages = []  # live list of message strings (e.g. "User: ..." or "Bot: ...")
    show_history = True

    # -------------------------
    # Helpers / UI builders
    # -------------------------
    def header_btn(label, on_click):
        return ft.Container(
            content=ft.Text(label, color=FG, size=int(tabs_script.s(12, page))),
            padding=ft.padding.symmetric(horizontal=12, vertical=6),
            border=ft.border.all(1, ORANGE),
            border_radius=int(tabs_script.s(8, page)),
            bgcolor=BG,
            on_click=on_click,
        )

    # user message container: NOT expanded (locks to text width)
    def user_bubble(text):
        return ft.Container(
            content=ft.Text(text, color=FG, size=int(tabs_script.s(12, page))),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border=ft.border.all(1, ORANGE),
            border_radius=int(tabs_script.s(8, page)),
            bgcolor=BG,
            alignment=ft.alignment.center_right,  # push to the right
            # no width/expand -> sizes to the text content only
        )

    # bot messages are plain Text (no Container)
    def bot_text(text):
        return ft.Text(text, color=FG, size=int(tabs_script.s(12, page)))

    # load a saved session into the current session (switch current chat)
    def load_session(session_id, ev=None):
        nonlocal current_session_id, current_session_messages, show_history
        # find session
        target = None
        for s in chat_sessions:
            if s.get("id") == session_id:
                target = s
                break
        if target is None:
            return
        # replace live session state
        current_session_id = target.get("id", current_session_id)
        current_session_messages = list(target.get("messages", []))

        # rebuild messages_column UI from messages (strings)
        messages_column.controls.clear()
        for m in current_session_messages:
            try:
                if isinstance(m, str) and m.startswith("User:"):
                    # display only the text after "User: "
                    body = m[len("User:"):].strip()
                    messages_column.controls.append(user_bubble(body))
                elif isinstance(m, str) and m.startswith("Bot:"):
                    body = m[len("Bot:"):].strip()
                    messages_column.controls.append(bot_text(body))
                else:
                    # fallback plain
                    messages_column.controls.append(bot_text(str(m)))
            except Exception:
                try:
                    messages_column.controls.append(bot_text(str(m)))
                except Exception:
                    pass

        # hide history pane after switching
        show_history = False
        history_list_column.controls.clear()

        # update UI and scroll to bottom if possible
        try:
            if page and getattr(page, "update", None):
                page.update()
        except Exception:
            pass

    # -------------------------
    # Header actions
    # -------------------------
    def on_new_chat(e):
        nonlocal current_session_id, current_session_messages, show_history

        # SAVE CURRENT SESSION TO CACHE FIRST (if non-empty)
        if current_session_messages:
            # Save to internal sessions (for fallback)
            chat_sessions.append({
                "id": current_session_id,
                "messages": list(current_session_messages)
            })

            # SAVE TO CACHE via on_message callback
            if callable(on_message):
                try:
                    on_message(list(current_session_messages))
                    print(f"Saved current chat to cache before starting new chat")
                except Exception as ex:
                    print(f"Failed to save to cache on new chat: {ex}")

            # Increment session and clear
            current_session_id += 1
            current_session_messages = []

        # NOTIFY HOST TO START NEW CHAT SESSION
        if callable(on_new_chat_callback):
            try:
                on_new_chat_callback()
                print(f"Started new chat session")
            except Exception as ex:
                print(f"Failed to start new chat session: {ex}")

        # clear message UI
        messages_column.controls.clear()
        history_list_column.controls.clear()
        show_history = False
        try:
            e.page.update()
        except Exception:
            pass

    def on_history(e):
        nonlocal show_history
        show_history = not show_history
        history_list_column.controls.clear()
        if show_history:
            # LOAD FROM CACHE instead of internal chat_sessions
            try:
                # Validate tab_id is actually usable (not None and not empty)
                has_valid_tab_id = tab_id is not None and tab_id != "" and str(tab_id).strip() != ""

                if not has_valid_tab_id:
                    # Fallback: use internal sessions
                    if chat_sessions:
                        for session in chat_sessions:
                            sid = session.get("id")
                            count = len(session.get("messages", []))
                            item = ft.Container(
                                content=ft.Text(f"Session {sid}: {count} messages", color=FG),
                                padding=ft.padding.symmetric(horizontal=10, vertical=8),
                                border=ft.border.all(1, ORANGE),
                                border_radius=int(tabs_script.s(8, page)),
                                bgcolor=BG,
                                on_click=lambda ev, sid=sid: load_session(sid, ev),
                            )
                            history_list_column.controls.append(item)
                    else:
                        history_list_column.controls.append(
                            ft.Text("No chat history available (tab_id not provided)", color=FG, size=10)
                        )
                else:
                    # Load from tab_state cache
                    state = tabs_script.read_tab_state(tab_id) or {}

                    # Find all chat_N blocks in cache
                    chat_blocks = {}
                    for key in state.keys():
                        if key.startswith("chat_"):
                            try:
                                block_num = int(key.split("_")[1])
                                chat_blocks[block_num] = state[key]
                            except Exception:
                                pass

                    # Also check for old format
                    old_history = state.get("chat_history", [])
                    if old_history and not chat_blocks:
                        # Show old format as one block
                        count = len(old_history)
                        preview_items = [ft.Text(f"Chat History: {count} messages", color=FG, weight="bold")]
                        for msg in old_history[:3]:
                            role = msg.get('role', 'bot')
                            content = msg.get('content', '')[:50] + "..."
                            preview_items.append(ft.Text(f"• [{role}]: {content}", color=FG, size=10))

                        # Create click handler for old format
                        def load_old_history(ev):
                            nonlocal current_session_messages, show_history
                            current_session_messages = []
                            messages_column.controls.clear()

                            for msg in old_history:
                                role = msg.get("role", "assistant")
                                content = msg.get("content", "")

                                if role == "user":
                                    current_session_messages.append(f"User: {content}")
                                    messages_column.controls.append(user_bubble(content))
                                else:
                                    current_session_messages.append(f"Bot: {content}")
                                    messages_column.controls.append(bot_text(content))

                            show_history = False
                            history_list_column.controls.clear()

                            try:
                                if page and getattr(page, "update", None):
                                    page.update()
                            except Exception:
                                pass

                        item = ft.Container(
                            content=ft.Column(preview_items),
                            padding=ft.padding.symmetric(horizontal=10, vertical=8),
                            border=ft.border.all(1, ORANGE),
                            border_radius=int(tabs_script.s(8, page)),
                            bgcolor=BG,
                            on_click=load_old_history,
                        )
                        history_list_column.controls.append(item)

                    # Display each chat block from cache
                    if chat_blocks:
                        for block_num in sorted(chat_blocks.keys()):
                            block_messages = chat_blocks[block_num]
                            if isinstance(block_messages, list) and block_messages:
                                count = len(block_messages)
                                # Create preview of messages
                                preview_items = [ft.Text(f"Chat {block_num}: {count} messages", color=FG, weight="bold",
                                                         size=int(tabs_script.s(12, page)))]
                                for msg in block_messages[:3]:  # Show first 3 messages
                                    role = msg.get("role", "bot")
                                    content = msg.get("content", "")
                                    preview_text = content[:40] + "..." if len(content) > 40 else content
                                    preview_items.append(ft.Text(f"• [{role}]: {preview_text}", color=FG, size=10))

                                # Create click handler to load this chat block
                                def make_load_chat_handler(block_num, block_messages):
                                    def load_chat_block(ev):
                                        nonlocal current_session_messages, show_history
                                        # Load the messages from this block
                                        current_session_messages = []
                                        messages_column.controls.clear()

                                        for msg in block_messages:
                                            role = msg.get("role", "assistant")
                                            content = msg.get("content", "")

                                            # Add to current session as string format
                                            if role == "user":
                                                current_session_messages.append(f"User: {content}")
                                                messages_column.controls.append(user_bubble(content))
                                            else:
                                                current_session_messages.append(f"Bot: {content}")
                                                messages_column.controls.append(bot_text(content))

                                        # Hide history after loading
                                        show_history = False
                                        history_list_column.controls.clear()

                                        # Update UI
                                        try:
                                            if page and getattr(page, "update", None):
                                                page.update()
                                        except Exception:
                                            pass

                                    return load_chat_block

                                item = ft.Container(
                                    content=ft.Column(preview_items),
                                    padding=ft.padding.symmetric(horizontal=10, vertical=8),
                                    border=ft.border.all(1, ORANGE),
                                    border_radius=int(tabs_script.s(8, page)),
                                    bgcolor=BG,
                                    on_click=make_load_chat_handler(block_num, block_messages),
                                )
                                history_list_column.controls.append(item)

                    if not chat_blocks and not old_history:
                        history_list_column.controls.append(ft.Text("No chat history found in cache", color=FG))
            except Exception as ex:
                print(f"Error loading history from cache: {ex}")
                import traceback
                traceback.print_exc()
                # Show error message to user
                history_list_column.controls.append(
                    ft.Text(f"Error loading history: {str(ex)[:100]}", color=FG, size=10)
                )
                # Fallback to internal sessions
                if chat_sessions:
                    for session in chat_sessions:
                        sid = session.get("id")
                        count = len(session.get("messages", []))
                        item = ft.Container(
                            content=ft.Text(f"Session {sid}: {count} messages", color=FG),
                            padding=ft.padding.symmetric(horizontal=10, vertical=8),
                            border=ft.border.all(1, ORANGE),
                            border_radius=int(tabs_script.s(8, page)),
                            bgcolor=BG,
                            on_click=lambda ev, sid=sid: load_session(sid, ev),
                        )
                        history_list_column.controls.append(item)
        try:
            e.page.update()
        except Exception:
            pass

    # -------------------------
    # Header UI
    # -------------------------
    new_chat_btn = header_btn("New Chat", on_new_chat)
    history_btn = header_btn("History", on_history)
    header = ft.Row(
        controls=[new_chat_btn, history_btn],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )

    # -------------------------
    # History list (collapsible)
    # -------------------------
    history_list_column = ft.Column(
        controls=[],
        spacing=int(tabs_script.sv(6, page)),
        scroll="auto",
    )

    # -------------------------
    # Messages area
    # -------------------------
    messages_column = ft.Column(
        controls=[],
        spacing=int(tabs_script.sv(6, page)),
        scroll="always",
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    messages_container = ft.Container(
        content=messages_column,
        expand=True,
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(8, page)),
        padding=6,
        bgcolor=BG,
    )

    if initial_messages:
        try:
            # initialize internal live message list to match the template's state
            current_session_messages = list(initial_messages)

            # rebuild messages UI from initial messages
            messages_column.controls.clear()
            for m in current_session_messages:
                try:
                    if isinstance(m, str) and m.startswith("User:"):
                        body = m[len("User:"):].strip()
                        messages_column.controls.append(user_bubble(body))
                    elif isinstance(m, str) and m.startswith("Bot:"):
                        body = m[len("Bot:"):].strip()
                        messages_column.controls.append(bot_text(body))
                    else:
                        messages_column.controls.append(bot_text(str(m)))
                except Exception:
                    try:
                        messages_column.controls.append(bot_text(str(m)))
                    except Exception:
                        pass
            # ensure UI is updated
            try:
                if page and getattr(page, "update", None):
                    page.update()
            except Exception:
                pass
        except Exception:
            pass

    # -------------------------
    # Input field and send
    # -------------------------
    new_message = ft.TextField(
        expand=True,
        border=None,
        bgcolor="transparent",
        color=FG,
        cursor_color=ORANGE,
        on_submit=lambda e: on_send(e),
    )

    new_message_box = ft.Container(
        content=new_message,
        padding=ft.padding.symmetric(horizontal=10, vertical=6),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(8, page)),
        bgcolor=BG,
        expand=True,
    )

    def on_send(e):
        nonlocal current_session_messages
        text = (new_message.value or "").strip()
        if not text:
            return

        # append user message (string stored)
        user_str = f"User: {text}"
        current_session_messages.append(user_str)
        messages_column.controls.append(user_bubble(text))

        # Generate bot reply using callback or default
        if callable(on_generate_reply):
            try:
                # Call the AI reply generator from ChannelPage
                bot_response = on_generate_reply(text)  # Pass just the text, not "User: text"
            except Exception as ex:
                print(f"Error generating AI reply: {ex}")
                bot_response = "Sorry, I encountered an error generating a response."
        else:
            # Fallback to default message if no generator provided
            bot_response = "Hi! I'm your YouTube growth assistant. Configure the Gemini API key in Settings to enable AI responses."

        # Append bot reply
        bot_reply = f"Bot: {bot_response}"
        current_session_messages.append(bot_reply)
        messages_column.controls.append(bot_text(bot_response))

        new_message.value = ""
        # update UI
        try:
            if e and getattr(e, "page", None):
                e.page.update()
        except Exception:
            pass

        if callable(on_message):
            try:
                on_message(list(current_session_messages))
            except TypeError:
                # backward-compatible: if host expects no args, call anyway
                try:
                    on_message()
                except Exception:
                    pass

    send_btn = ft.Container(
        content=ft.Text("Send", color=FG, size=int(tabs_script.s(12, page))),
        padding=ft.padding.symmetric(horizontal=16, vertical=20),
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(8, page)),
        bgcolor=BG,
        on_click=on_send,
    )

    thumbnail_container = ft.Container(
        width=80,
        height=56,
        padding=4,
        border=ft.border.all(1, ORANGE),
        border_radius=6,
        bgcolor=BG,
        content=None,
    )

    attach_button = ft.Container(
        width=80,
        height=56,
        border=ft.border.all(1, ORANGE),
        border_radius=6,
        bgcolor=BG,
        alignment=ft.alignment.center,
        content=ft.Text(
            "Attach\nVideo",
            size=11,
            text_align=ft.TextAlign.CENTER,
            color=FG,
            weight=ft.FontWeight.W_500,
        ),
        on_click=(
            lambda e: (print("DEBUG: Attach Video button clicked in chatbot_template"), on_attach(e))[1]) if callable(
            on_attach) else None,
    )

    attach_row = ft.Row(
        controls=[attach_button, thumbnail_container],
        spacing=int(tabs_script.s(8, page)),
    )

    if isinstance(host_controls, dict):
        def set_chat_thumbnail(src: str | None):
            if src:
                thumbnail_container.content = ft.Image(
                    src=src,
                    width=72,
                    height=48,
                    fit=ft.ImageFit.COVER,
                )
            else:
                thumbnail_container.content = None
            if page:
                page.update()

        host_controls["set_thumbnail"] = set_chat_thumbnail
        host_controls["thumbnail_container"] = thumbnail_container

    input_row = ft.Row(
        controls=[new_message_box, send_btn],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )

    # -------------------------
    # Main layout assembly
    # -------------------------
    main_col = ft.Column(
        controls=[
            header,
            history_list_column if show_history else ft.Container(),
            messages_container,
            attach_row,
            input_row,
        ],
        expand=True,
    )

    # -------------------------
    # Outer container
    # -------------------------
    return ft.Container(
        content=ft.Column(
            controls=[main_col],
            scroll="auto",  # <-- Makes entire chatbot scrollable
            expand=True,
        ),
        width=int(tabs_script.sh(width, page)) if width else None,
        height=int(tabs_script.sv(height, page)) if height else None,
        padding=ft.padding.all(int(tabs_script.s(12, page))),
        bgcolor=BG,
        border=ft.border.all(1, ORANGE),
        border_radius=int(tabs_script.s(12, page)),
    )