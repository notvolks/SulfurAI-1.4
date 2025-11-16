import os

################ Welcome to SulfurAI dashboard!
### Functions here should be modified with proper intent.
### This python script was written in the Holly format. To find out how it works go into VersionDATA/HollyFormat/ReadMe.txt
### This python script is designed to host all SulfurAI API functions for python and run via the __main__ tag.

### LAYOUT:
# ---------------GOING DOWN!
#####-TOS reminder
#####-Imports call_file_path (important dependency)
#####-Ensures required packages are installed
#####-Importing all files
#####-Running all functions to import the dash (Sulfax UI engine).
#####-Running all creative files + dependancies
#####-Running the server/ webapp


### FOR DEVS (CREATING NEW UI):
# -add the sections in the sections dictionary
# -change the custom css and js in SulfurDashboardAssets/styling
# -add section dividers in the all_sections_html dictionary
# -add a custom title in the run_dashboard() function under the st.markdown() function (adjust css)
# - change the default graph for a section by re-arranging the graphs options or see line 319

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from extra_models.Sulfur.GeneralScripts.LocalUserEvents import events_hoster
events_hoster.write_event("event_OpenedDashboard")
from scripts.ai_renderer_sentences import error
import time

print_verti_list = error.print_verti_list

# Print Terms of Service on direct run
if __name__ == "__main__":
    TOS = [
        "--------------------------------------------------------------------------------------------------",
        "⚠️ This application is external to SulfurAI and is maintained by different sources. Therefore project works may be different.",
        "--------------------------------------------------------------------------------------------------",
        "By using this application you agree to the Terms of Service listed in the project files.",
        "If you do not consent, stop using our services.",
        "If you cannot find it, install a new version OR look in the root folder for 'Terms of Service.txt'.",
        "--------------------------------------------------------------------------------------------------",
        "",
        "                                       🔃 Loading... 🔃                                                    ",
        "                           App powered by the SulfurAI Sulfax UI engine.                                                             ",
        "",
        "--------------------------------------------------------------------------------------------------",
        "⚠️ App freezing? Restart the app and wait. Cacheing may take a while.",
        "--------------------------------------------------------------------------------------------------",

    ]
    print_verti_list(TOS)

# DELETING THE TOS NOTICE SCRIPT RESULTS IN INSTANT TERMINATION OF SULFUR WARRANTY AND CANCELS YOUR CONTRACT. IT IS *IN VIOLATION* OF THE TOS.
# YOU MAY BE INDEFINITELY BANNED FROM SULFUR SERVICES IF YOU REMOVE THIS TOS NOTICE SCRIPT WITHOUT PRIOR WRITTEN CONSENT BY VOLKSHUB GROUP.

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import subprocess
import importlib


#####################------------------------------------------------INBUILT FUNCTIONS------------------------------------------------


def _get_call_file_path():
    """
    Returns the call object for accessing internal SulfurAI file paths.

    This is a wrapper to load the Call() interface for retrieving dynamic or predefined file paths
    required by the dashboard and related modules.

    Returns:
        object: Instance with path-fetching methods.
    """
    from extra_models.Sulfur.TrainingScript.Build import call_file_path
    return call_file_path.Call()


# Call file paths
call = _get_call_file_path()

# Ensure required packages
modules = ["streamlit", "dash", "pandas", "plotly", "pywebview"]
for mod in modules:
    try:
        __import__(mod)
    except ImportError:
        file_path_cache_localHost_pip_debug = call.cache_LocalpipCacheDebug()
        with open(file_path_cache_localHost_pip_debug, "r", encoding="utf-8", errors="ignore") as file:
            cache_stored_pip_debug = file.readlines()
        if mod not in [line.strip() for line in cache_stored_pip_debug]: print(
            f"The dependancies for SulfurAI Dashboard are not installed. Please install them using the installer in INSTALLER/INSTALL SULFURAI-DASHBOARD/Run Installer.bat")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# imports app modules
from apps.SulfurDashboardAssets.renderer import sidebar

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from streamlit.runtime.scriptrunner import get_script_run_ctx
from pathlib import Path
import webbrowser
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

all_sections_html = ""


@st.cache_data(show_spinner=True)  # Show spinner while loading data
def load_data():
    """
    Load and return the necessary data for the dashboard.
    This function is cached to improve performance and avoid repeated loading.
    It will remove problematic lines if errors occur during loading.
    """
    intent_df = None
    devices_df = None
    location_df = None

    # Load intent data
    try:
        intent_df = pd.read_csv("scripts/ai_renderer_2/training_data_sentences/data.csv")
    except Exception as e:
        st.error(f"Error loading intent data: {e}")

    # Load user devices data with error handling
    try:
        current_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__)
        ))
        file_path = os.path.join(current_dir, 'scripts', 'ai_renderer', 'training_data', 'data_train_sk', 'data.txt')
        devices_df = get_devices_df_from_datafile(file_path)

    except Exception as e:
        st.error(f"Error loading devices data: {e}")

    # Load location data
    try:
        location_df = load_training_data_txt(
            "scripts/ai_renderer_sentences/sentence_location_build/training_data_sentences/data.txt")
    except Exception as e:
        st.error(f"Error loading location data: {e}")

    return intent_df, devices_df, location_df


def load_and_clean_devices_data(filepath):
    """
    Load devices data and remove problematic lines.

    Args:
        filepath (str): Path to the devices data file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with valid data.
    """
    cleaned_data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(",")  # Adjust delimiter if necessary
            if len(parts) == 3:  # Expecting 3 fields
                cleaned_data.append(parts)
            else:
                pass

    # Convert cleaned data back to DataFrame
    return pd.DataFrame(cleaned_data, columns=["field1", "field2", "field3"])


def load_training_data_txt(filepath):
    """
    Loads training data from a .txt file for location classification or display.

    The file must have 3 tab-separated values per line: [sentence, location, accuracy].

    Args:
        filepath (str): Path to the training .txt file.

    Returns:
        pd.DataFrame: DataFrame with 'sentence', 'location', 'accuracy' columns or None if loading fails.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    acc_val = float(parts[-1].rstrip('%'))
                except ValueError:
                    acc_val = None
                data.append({
                    "sentence": parts[0].strip("[]'\""),
                    "location": parts[1],
                    "accuracy": acc_val
                })
    return pd.DataFrame(data)


# ---------------------------paths---------------------------


css_path = os.path.join("apps\SulfurDashboardAssets\styling", "style.css")
with open(css_path, "r") as f:
    css = f.read()

js_path = os.path.join("apps\SulfurDashboardAssets\styling", "script.js")
with open(js_path, "r") as f:
    js = f.read()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def render_glowing_section(title, dataframes, graph_names, section_id="section-container", custom_style=""):
    """
     Renders a glowing dashboard section with visualizations (pie, bar, line, or accuracy metrics).

     Args:
         title (str): Section title.
         dataframes (list): List of pandas DataFrames to visualize.
         graph_names (list): Metadata dicts describing charts and types (e.g., 'device', 'intent').
         section_id (str): HTML section ID for linking or styling.
         custom_style (str): Custom inline CSS for positioning and layout.

     Returns:
         str: Complete HTML string for the section including chart switching JS and layout.
     """
    import pandas as pd
    import plotly.express as px

    inner_charts_html = []
    global chart_ids
    chart_ids = []

    for idx, df in enumerate(dataframes):
        if df is None or df.empty:
            continue

        # Special handling for accuracy metrics
        if "Accuracy" in df.columns:
            # Read and parse the values from Output.txt
            with open("DATA/Output.txt", "r", encoding='utf-8') as f:
                content = f.read()

            # Extract each value using the exact format from Output.txt
            device_acc = float(content.split("Predicted Device Accuracy : ")[1].split("%")[0])
            mean_device_acc = float(content.split("Average/Mean Accuracy: ")[1].split("%")[0])

            # Extract mood accuracies from the specific line
            mood_line = content.split("Predicted Mood Accuracy : ")[1].split("\n")[0]
            user_mood_acc = float(mood_line.split("(user)")[0].strip().rstrip("%"))
            global_mood_acc = float(mood_line.split("(global)")[0].split(",")[1].strip().rstrip("%"))

            # Extract user:mean and overall accuracy
            user_mean_acc = float(content.split("Average <User : Mean> Accuracy : ")[1].split("%")[0])
            overall_acc = float(content.split("Overall Accuracy : ")[1].split("%")[0])

            # Create DataFrame with actual values
            df = pd.DataFrame({
                'Metric': ['Device', 'Mean Device', 'Mood (User)', 'Mood (Global)', 'User:Mean', 'Overall'],
                'Accuracy': [device_acc, mean_device_acc, user_mood_acc, global_mood_acc, user_mean_acc, overall_acc]
            })

            import plotly.graph_objects as go

            # Create the bar chart using graph_objects
            bar_fig = go.Figure()

            # Add the bars
            bar_fig.add_trace(go.Bar(
                x=df['Metric'],
                y=df['Accuracy'],
                text=df['Accuracy'].round(1).astype(str) + '%',
                textposition='outside',
                marker_color='rgb(64, 224, 208)',
                hoverinfo='none'  # Disable hover tooltip
            ))

            # Update layout with fixed positioning
            bar_fig.update_layout(
                title='System Accuracy Metrics',
                template="plotly_dark",
                bargap=0.15,
                width=800,
                height=500,
                showlegend=False,
                yaxis_title='Accuracy (%)',
                margin=dict(l=50, r=50, t=50, b=50, pad=4),  # Fixed margins
                hovermode=False,  # Disable hover interactions
                xaxis=dict(
                    fixedrange=True,  # Disable zoom/pan
                    showgrid=False  # Optional: hide grid
                ),
                yaxis=dict(
                    fixedrange=True,  # Disable zoom/pan
                    showgrid=False  # Optional: hide grid
                ),
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)'  # Transparent container
            )

            # Configure text appearance
            bar_fig.update_traces(
                textfont=dict(
                    size=14,
                    color='white',
                    family='Arial Black'
                ),
                cliponaxis=False
            )

            chart_id = f"accuracy-metrics-{idx}"
            chart_ids.append(chart_id)

            chart_html = f"""
                <div class="glowing-chart"
                     id="chart-{chart_id}"
                     style="position:absolute;
                            left:10%;
                            top:200px;
                            background-color:#111;
                            pointer-events:none;">
                    <div class="chart-ring"></div>
                    <div style="pointer-events:auto;">
                        {bar_fig.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})}
                    </div>
                </div>
            """
            inner_charts_html.append(chart_html)
            continue

        try:
            pos = graph_names[idx]["positions"][idx]
            left = pos.get("left", "0px")
            top = pos.get("top", "0px")
        except:
            left, top = "0px", "0px"

        is_device_data = graph_names[idx].get("type") == "device"
        is_data = graph_names[idx].get("type")

        if is_device_data:
            if "device_type" in df.columns:
                df = df[df["device_type"].notna()]
                counts = df["device_type"].value_counts()
                percentages = (counts / counts.sum() * 100).round(2)

                intent_data = pd.DataFrame({
                    'category': percentages.index,
                    'percentage': percentages.values,
                    'label': [f"{cat} ({val:.1f}%)" for cat, val in zip(percentages.index, percentages.values)]
                })
                col = 'device_type'  # Set column name for device data
        else:
            # Determine which column to use before the value counts
            if 'intent' in df.columns:
                col = 'intent'
            elif 'location' in df.columns:
                col = 'location'
            else:
                col = df.columns[0]

            # Now process the data with the determined column
            if "count" in df.columns:
                counts = df.groupby(col)["count"].sum()
                intent_counts = (counts / counts.sum()) * 100
            else:
                intent_counts = df[col].value_counts(normalize=True) * 100

            intent_data = pd.DataFrame({
                'category': intent_counts.index,
                'percentage': intent_counts.values
            })

        def create_charts(df, chart_type):
            """Creates custom Plotly charts using direct percentage values"""
            import plotly.graph_objects as go
            
            if chart_type == "device":
                # Ensure correct column names and data for device charts
                categories = df['category'].tolist()
                percentages = df['percentage'].tolist()
            else:
                # Handle other chart types normally
                categories = df['category'].tolist()
                percentages = df['percentage'].tolist()
            
            chart_id = f"{chart_type}-{len(categories)}"
            chart_ids.append(chart_id)
            
            # Create pie chart
            pie_fig = go.Figure(data=[go.Pie(
                labels=categories,
                values=percentages,
                textinfo='label+percent',
                hoverinfo='label+percent',
                marker=dict(colors=['#FF4B4B', '#39FF14']),  # Only two colors for Desktop/Mobile
                textfont=dict(size=14)
            )])
            pie_fig.update_layout(
                title=dict(text="", font=dict(size=18)),
                width=400,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark"
            )

            # Create bar chart
            bar_fig = go.Figure(data=[go.Bar(
                x=categories,
                y=percentages,
                text=[f'{p:.1f}%' for p in percentages],
                textposition='outside',
                marker_color=['#FF4B4B', '#39FF14'],  # Only two colors for Desktop/Mobile
                textfont=dict(size=14)
            )])
            bar_fig.update_layout(
                title=dict(text="", font=dict(size=18)),
                width=400,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                showlegend=False,
                yaxis=dict(range=[0, 100]),
                xaxis=dict(tickfont=dict(size=14))
            )

            # Create line chart
            line_fig = go.Figure(data=[go.Scatter(
                x=categories,
                y=percentages,
                mode='lines+markers+text',
                text=[f'{p:.1f}%' for p in percentages],
                textposition='top center',
                line=dict(color='#39FF14', width=3),
                marker=dict(size=12),
                textfont=dict(size=14)
            )])
            line_fig.update_layout(
                title=dict(text="", font=dict(size=18)),
                width=400,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                showlegend=False,
                yaxis=dict(range=[0, 100]),
                xaxis=dict(tickfont=dict(size=14))
            )

            return (
                pie_fig.to_html(include_plotlyjs=False, full_html=False),
                bar_fig.to_html(include_plotlyjs=False, full_html=False),
                line_fig.to_html(include_plotlyjs=False, full_html=False),
                chart_id
            )

        if is_device_data:
            pie_html, bar_html, line_html, chart_id = create_charts(intent_data, "device")
        elif graph_names[idx].get("type") == "insight":
            pie_html, bar_html, line_html, chart_id = create_charts(intent_data, "insight") 
        elif graph_names[idx].get("type") == "location":
            pie_html, bar_html, line_html, chart_id = create_charts(intent_data, "location")

        if is_data == "device":
            slider_default_value = 1
        elif is_data == "location":
            slider_default_value = 2
        else:
            slider_default_value = 0

        try:
            pos = graph_names[idx]["positions"][idx]
            left = pos.get("left", "0px")
            top = pos.get("top", "0px")
        except (KeyError, IndexError):
            left, top = "0px", "0px"

        chart_html = f"""
            <div class="glowing-chart"
                id="chart-{chart_id}"
                style="position:absolute;
                        left:{left};
                        top:{top};
                        background-color:#111;
                        padding: 20px;  
                        min-width: 450px;  
                        min-height: 450px;">  
            <div class="chart-ring"></div>
            <div class="slider-wrapper">
                <div class="slider-ring"></div>
                <input id="slider-{chart_id}"
                        type="range"
                        min="0" max="2"
                        value="{slider_default_value}"
                        style="width:100%; margin-bottom:15px; height: 10px;">  
            </div>
            <div id="pie-{chart_id}"
                style="display:{'block' if slider_default_value == 0 else 'none'}">
                {pie_html}
            </div>
            <div id="bar-{chart_id}"
                style="display:{'block' if slider_default_value == 1 else 'none'}">
                {bar_html}
            </div>
            <div id="line-{chart_id}"
                style="display:{'block' if slider_default_value == 2 else 'none'}">
                {line_html}
            </div>
            </div>
        """
        inner_charts_html.append(chart_html)

    chart_scripts = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
        """

    for cid in chart_ids:  # Now chart_ids will have all IDs from this section
        chart_scripts += f"""
            document.getElementById('slider-{cid}').addEventListener('input', function(e) {{
                const value = e.target.value;
                document.getElementById('pie-{cid}').style.display = value === '0' ? 'block' : 'none';
                document.getElementById('bar-{cid}').style.display = value === '1' ? 'block' : 'none';
                document.getElementById('line-{cid}').style.display = value === '2' ? 'block' : 'none';
            }});
            """

    chart_scripts += """
    });
    </script>
    """

    section_html = f"""
                 <div class="dashboard-section glowing-section" id="{section_id}" style="{custom_style}">
                    <div class="section-ring" id="ring-{section_id}"></div>
                    <div class="glowing-section-title">{title}</div>
                    <div class="charts-container">
                        {"".join(inner_charts_html)}
                    </div>
                    {chart_scripts}
                </div>
                """

    return section_html


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def render_summary_section(metrics, section_id="summary-section", custom_style=""):
    """
    Renders a summary section with key insights from the dashboard data.

    Args:
        metrics (dict): Dictionary containing summary metrics like top_intent, top_device, top_location, and overall_acc.
        section_id (str): HTML section ID for linking or styling.
        custom_style (str): Custom inline CSS for positioning and layout.

    Returns:
        str: Complete HTML string for the summary section.
    """
    summary_html = f"""
    <div class="dashboard-section glowing-section" id="{section_id}" style="{custom_style}">
        <div class="section-ring" id="ring-{section_id}"></div>
        <div class="glowing-section-title">Summary</div>
        <div class="summary-content">
            <p><strong>Most Common Intent:</strong> {metrics.get('top_intent', 'N/A')}</p>
            <p><strong>Primary Device:</strong> {metrics.get('top_device', 'N/A')}</p>
            <p><strong>Top User Location:</strong> {metrics.get('top_location', 'N/A')}</p>
            <p><strong>Overall Accuracy:</strong> {metrics.get('overall_acc', 0):.2f}%</p>
        </div>
    </div>
    """
    return summary_html


def get_devices_df_from_datafile(filepath):
    """
    Read the devices training data file and return a DataFrame of individual device rows.

    - Tries to detect device info in each line (codes "1"/"2", words "desktop"/"mobile", or
      tokens like "d"/"m").
    - Produces a DataFrame with columns:
        - 'raw' : the original token/string found (for backwards compatibility)
        - 'device_type' : normalized "Desktop" or "Mobile" (used by the charting code)
    - Prints debug counts/percentages (remove or convert to st.write() once verified).
    """
    import re
    import pandas as pd
    devices_rows = []
    counts = {'Desktop': 0.0, 'Mobile': 0.0}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # split and strip tokens
                parts = [p.strip() for p in re.split(r"[,\t]", line) if p.strip() != ""]

                # Candidate token is last meaningful token
                token = parts[-1].lower() if parts else ""

                # Look for explicit device code or keyword
                device = None
                if token in ("1", "d", "desktop", "desktop_device"):
                    device = "Desktop"
                elif token in ("2", "m", "mobile", "mobile_device"):
                    device = "Mobile"
                else:
                    # fallback: inspect entire line for keywords
                    low = line.lower()
                    if "desktop" in low:
                        device = "Desktop"
                    elif "mobile" in low:
                        device = "Mobile"
                    else:
                        # also accept tokens that look numeric where weight may be parts[1]
                        if parts and parts[-1].isdigit():
                            # if numeric last token it's probably a code; try earlier tokens
                            for p in parts[:-1][::-1]:
                                if p.lower() in ("1", "2", "d", "m", "desktop", "mobile"):
                                    token2 = p.lower()
                                    if token2 in ("1", "d", "desktop"):
                                        device = "Desktop"
                                    elif token2 in ("2", "m", "mobile"):
                                        device = "Mobile"
                                    break

                if device:
                    devices_rows.append({"raw": token, "device_type": device})
                    counts[device] += 1.0
                else:
                    # if we couldn't determine a device, skip that line (safe)
                    continue

        # Build DataFrame of rows so rendering uses the same .value_counts() path as other charts
        result_df = pd.DataFrame(devices_rows)

        # debug prints (ok while debugging)
        total = sum(counts.values()) if counts else 0.0
        percentages = {k: (v / total * 100.0) if total > 0 else 0.0 for k, v in counts.items()}

        return result_df

    except FileNotFoundError:
        print(f"[get_devices_df_from_datafile] file not found: {filepath}")
        return pd.DataFrame(columns=["raw", "device_type"])
    except Exception as e:
        print(f"[get_devices_df_from_datafile] error: {e}")
        return pd.DataFrame(columns=["raw", "device_type"])



def get_accuracy_metrics():
    """
    Parse accuracy metrics from DATA/Output.txt using regex,
    works with the current plain text format.
    """
    import re
    try:
        with open("DATA/Output.txt", "r", encoding='utf-8') as f:
            content = f.read()

        device_acc = float(re.search(r"Predicted Device Accuracy\s*:\s*([\d.]+)%", content).group(1))
        mean_device_acc = float(re.search(r"Average/Mean Accuracy\s*:\s*([\d.]+)%", content).group(1))

        mood_match = re.search(r"Predicted Mood Accuracy\s*:\s*([\d.]+)%\s*\(user\),\s*([\d.]+)%\s*\(global\)", content)
        user_mood_acc = float(mood_match.group(1))
        global_mood_acc = float(mood_match.group(2))

        user_mean_acc = float(re.search(r"Average <User : Mean> Accuracy\s*:\s*([\d.]+)%", content).group(1))
        overall_acc = float(re.search(r"Overall Accuracy\s*:\s*([\d.]+)%", content).group(1))

        return {
            'Metric': ['Device', 'Mean Device', 'Mood (User)', 'Mood (Global)', 'User:Mean', 'Overall'],
            'Accuracy': [
                device_acc,
                mean_device_acc,
                user_mood_acc,
                global_mood_acc,
                user_mean_acc,
                overall_acc
            ]
        }
    except Exception as e:
        print(f"Error reading accuracy metrics: {e}")
        return None


def run_dashboard():
    """
    Initializes and renders the SulfurAI dashboard inside Streamlit.

    - Loads CSV and TXT datasets.
    - Constructs each dashboard section.
    - Combines all components into a glowing HTML template.
    - Launches a local server for iframe-based visual rendering.
    """

    try:
        st.set_page_config(page_title="SulfurAI Dashboard", layout="wide")

        try:
            file_path_logo = call.EXTERNALAPP_dashboard_sulfurLogo64()
            with open(file_path_logo, "r") as f:
                b64logo = f.read()
        except Exception as e:
            print(f"Error loading logo: {e}")
            b64logo = ""

        default_pattern = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAEElEQVR42mP8z8DwD:wAAIAAAAKODoJAAAACkJREFUeNpj/Pz/AwMDAAAAaAP//wC8AAAAPgABiwAAAI0AAAAASUVORK5CYII="

        st.markdown(f"""
        <style>
        /* Reset default styles */
        body, html {{
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden; /* Prevent scrollbars that might interfere with fixed backgrounds */
        }}

        .stApp {{
            background: transparent !important; /* Set background to black */
        }}

        [data-testid="stAppViewContainer"] {{
            background: transparent !important; /* Set background to black */
        }}

        /* Moving background */
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #000000; /* Set background to black */
            z-index: -2;
        }}

        /* Pattern overlay */
        body::after {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('data:image/png;base64,{b64logo if b64logo else default_pattern}');
            background-repeat: repeat;
            opacity: 0.3; /* Increased opacity */
            z-index: -1;
            animation: patternMove 20s linear infinite;
        }}

        /* Animations */
        @keyframes gradientMove {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        @keyframes patternMove {{
            from {{ background-position: 0 0; }}
            to {{ background-position: 100px 100px; }}
        }}

        /* Title and subtitle styles */
        .custom-title {{
            font-size: 64px;
            font-weight: 900;
            background: linear-gradient(90deg, #FFA500, #FFD700, #FFA500);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 5px rgba(255, 215, 0, 0.7);
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 1rem;
            position: relative;
            z-index: 1;
            animation: titleGlow 5s ease-in-out infinite;
        }}

        .custom-subtitle {{
            font-size: 24px;
            color: #FFD700;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
            position: relative;
            z-index: 1;
        }}

        @keyframes titleGlow {{
            0%, 100% {{ text-shadow: 0 0 5px rgba(255, 215, 0, 0.7); }}
            50% {{ text-shadow: 0 0 20px rgba(255, 215, 0, 0.9); }}
        }}
        </style>

        <div class="custom-title">SulfurAI Dashboard - Graphs</div>
        <div class="custom-subtitle">Real-time AI Analytics & Insights</div>
        """, unsafe_allow_html=True)

        # Load first dataset: intents from CSV
        intent_csv_path = "scripts/ai_renderer_2/training_data_sentences/data.csv"
        user_devices = r"scripts\ai_renderer\training_data\data_train_sk\data.txt"
        try:
            intent_df, devices_df, location_df = load_data()
        except Exception as e:
            st.error(f"Failed to load intent CSV data: {e}")
            print(f"Failed to load intent CSV data: {e}")
            intent_df, devices_df, location_df = None, None, None

        from io import StringIO

        try:
            devices = get_devices_df_from_datafile(user_devices)
        except Exception as e:
            st.error(f"Failed to load user CSV data: {e}")
            print(f"Failed to load user CSV data: {e}")
            devices = None

        intent_csv_path = os.path.join("scripts", "ai_renderer_2", "training_data_sentences", "data.csv")
        user_devices_path = os.path.join("scripts", "ai_renderer", "training_data", "data_train_sk", "data.txt")

        # Accuracy metrics DataFrame (as in original code)
        acc_metrics = get_accuracy_metrics()
        if acc_metrics:
            accuracy_df = pd.DataFrame(acc_metrics)
        else:
            accuracy_df = pd.DataFrame({
                'Metric': ['Device', 'Mean Device', 'Mood (User)', 'Mood (Global)', 'User:Mean', 'Overall'],
                'Accuracy': [0, 0, 0, 0, 0, 0]
            })

        # Compute text summaries
        # Most common intent
        if not intent_df.empty and 'intent' in intent_df.columns:
            most_intent = intent_df['intent'].mode()[0]
        else:
            most_intent = "N/A"
        # Dominant device type
        if not devices.empty:
            # Map codes to types (as in rendering code)
            def map_device_type(val):
                val = str(val).strip()
                return "mobile" if val == "2" else ("desktop" if val == "1" else None)

            devices['device_type'] = devices.iloc[:, -1].apply(map_device_type)
            devices = devices[devices['device_type'].notna()]
            if not devices.empty:
                dominant_device = devices['device_type'].mode()[0]
            else:
                dominant_device = "N/A"
        else:
            dominant_device = "N/A"
        # Top locations
        if not location_df.empty and 'location' in location_df.columns:
            top_locations = location_df['location'].value_counts().index.tolist()[:3]  # top 3
            top_locations_str = ", ".join(top_locations)
        else:
            top_locations_str = "N/A"
        # Accuracy breakdown
        acc_values = accuracy_df.set_index('Metric')['Accuracy'].round(1).astype(str) + '%'

        # Build HTML content for Content Summary
        content_paragraphs = f"""
               <p><strong>Most common intent:</strong> {most_intent}.</p>
               <p><strong>Dominant device:</strong> {dominant_device.capitalize()}.</p>
               <p><strong>Top locations:</strong> {top_locations_str}.</p>
               <p><strong>Accuracy metrics:</strong></p>
               <ul>
                   <li>Device: {acc_values['Device']}</li>
                   <li>Mean Device: {acc_values['Mean Device']}</li>
                   <li>Mood (User): {acc_values['Mood (User)']}</li>
                   <li>Mood (Global): {acc_values['Mood (Global)']}</li>
                   <li>User:Mean: {acc_values['User:Mean']}</li>
                   <li>Overall: {acc_values['Overall']}</li>
               </ul>
           """


        sections_data = {
            "User Insight": {
                "dataframes": [intent_df],
                "graph_names": [{
                    "pie": "User Intents Pie",
                    "bar": "User Intents Bar",
                    "line": "User Intents Line",
                    "type": "insight",
                    "positions": [{"left": "20%", "top": "150px"}]
                }],
                "position": {"left": "15%", "top": "20px", "width": "30%"}
            },
            "User Average Devices": {
                "dataframes": [get_devices_df_from_datafile(user_devices_path)],
                "graph_names": [{
                    "pie": "User Devices Pie",
                    "bar": "User Devices Bar",
                    "line": "User Devices Line",
                    "type": "device",
                    "positions": [{"left": "20%", "top": "150px"}]
                }],
                "position": {"left": "55%", "top": "20px", "width": "30%"}
            },
            "User Location": {
                "dataframes": [location_df],
                "graph_names": [{
                    "pie": "User Country Distribution Pie",
                    "bar": "User Country Distribution Bar",
                    "line": "User Country Distribution Line",
                    "type": "location",
                    "positions": [{"left": "20%", "top": "150px"}]
                }],
                "position": {"left": "15%", "top": "750px", "width": "30%"}
            },
            "Accuracy Metrics": {
                "dataframes": [accuracy_df],
                "graph_names": [{
                    "name": "accuracy_metrics",
                    "type": "bar",
                    "positions": [{"left": "-150px", "top": "100px"}]
                }],
                "position": {"left": "55%", "top": "750px", "width": "40%"}
            }
        }

        # --- Compute summary metrics from sections_data
        summary_metrics = {
            "top_intent": "N/A",
            "top_device": "N/A",
            "top_location": "N/A",
            "overall_acc": 0
        }

        # Extract top intent from User Insight section
        intent_df = sections_data["User Insight"]["dataframes"][0]
        if intent_df is not None and not intent_df.empty and "intent" in intent_df.columns:
            summary_metrics["top_intent"] = intent_df["intent"].value_counts().idxmax()

        # Extract top device from User Average Devices section
        devices_df = sections_data["User Average Devices"]["dataframes"][0]
        if devices_df is not None and not devices_df.empty:
            # Ensure device_type column exists
            if "device_type" not in devices_df.columns:
                def map_device_type(row):
                    val = str(row.iloc[-1]).strip()
                    if val == "2":
                        return "mobile"
                    elif val == "1":
                        return "desktop"
                    else:
                        return None

                devices_df["device_type"] = devices_df.apply(map_device_type, axis=1)
                devices_df = devices_df[devices_df["device_type"].notna()]
            if not devices_df.empty:
                summary_metrics["top_device"] = devices_df["device_type"].value_counts().idxmax()

        # Extract top location from User Location section
        location_df = sections_data["User Location"]["dataframes"][0]
        if location_df is not None and not location_df.empty and "location" in location_df.columns:
            summary_metrics["top_location"] = location_df["location"].value_counts().idxmax()

        # Extract overall accuracy from Accuracy Metrics section
        accuracy_df = sections_data["Accuracy Metrics"]["dataframes"][0]
        if accuracy_df is not None and not accuracy_df.empty and "Accuracy" in accuracy_df.columns:
            overall_acc_row = accuracy_df[accuracy_df["Metric"] == "Overall"]
            if not overall_acc_row.empty:
                summary_metrics["overall_acc"] = overall_acc_row["Accuracy"].values[0]

        all_sections_html = (
                render_glowing_section(
                    title="User Insight",
                    dataframes=sections_data["User Insight"]["dataframes"],
                    graph_names=sections_data["User Insight"]["graph_names"],
                    section_id="user-insight-section",
                    custom_style=f"""
                    position: absolute;
                    top: {sections_data["User Insight"]["position"]["top"]};
                    left: {sections_data["User Insight"]["position"]["left"]};
                    width: {sections_data["User Insight"]["position"]["width"]};
                """
                ) +
                '<div class="gradient-divider" id="user-insight-section-divider" style="top: -20px;"></div>' +

                render_glowing_section(
                    title="User Average Devices",
                    dataframes=sections_data["User Average Devices"]["dataframes"],
                    graph_names=sections_data["User Average Devices"]["graph_names"],
                    section_id="user-devices-section",
                    custom_style=f"""
                    position: absolute;
                    top: {sections_data["User Average Devices"]["position"]["top"]};
                    left: {sections_data["User Average Devices"]["position"]["left"]};
                    width: {sections_data["User Average Devices"]["position"]["width"]};
                """
                ) +
                '<div class="gradient-divider" id="user-devices-section-divider" style="top: 700px;"></div>' +

                render_glowing_section(
                    title="User Location",
                    dataframes=sections_data["User Location"]["dataframes"],
                    graph_names=sections_data["User Location"]["graph_names"],
                    section_id="user-expression-section",
                    custom_style=f"""
                    position: absolute;
                    top: {sections_data["User Location"]["position"]["top"]};
                    left: {sections_data["User Location"]["position"]["left"]};
                    width: {sections_data["User Location"]["position"]["width"]};
                """
                ) +

                render_glowing_section(
                    title="Accuracy Metrics",
                    dataframes=sections_data["Accuracy Metrics"]["dataframes"],
                    graph_names=sections_data["Accuracy Metrics"]["graph_names"],
                    section_id="accuracy-metrics-section",
                    custom_style=f"""
                    position: absolute;
                    top: {sections_data["Accuracy Metrics"]["position"]["top"]};
                    left: {sections_data["Accuracy Metrics"]["position"]["left"]};
                    width: {sections_data["Accuracy Metrics"]["position"]["width"]};
                """
                ) +

                # Add the Summary section at the bottom center
                render_summary_section(
                    metrics=summary_metrics,
                    section_id="summary-section",
                    custom_style=f"""
                    position: absolute;
                    top: 0px;
                    left: 35%;
                    width: 30%;
                    text-align: center;
                    padding: 20px;
                """
                )
        )

        # graphs shii
        sections_nav = {
            "graphs": {
                "name": "Graphs",
                "target": [
                    "user-insight-section",
                    "user-devices-section",
                    "user-expression-section",
                    "accuracy-metrics-section"
                ]
            },
            "summary": {
                "name": "Summary",
                "target": "summary-section"
            }
        }

        sidebar_css, sidebar_nav_html, sidebar_js = sidebar.get_sidebar_includes(sections_nav)
        sidebar_position_script = sidebar.get_sidebar_position_script()

        # Combine all CSS and JS blocks properly
        final_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8" />
            <title>SulfurAI Dashboard - Graphs</title>

            <!-- Global dashboard styles -->
            <style>
                {css}  <!-- Main dashboard visualizations CSS -->
            </style>

            <!-- Sidebar styles -->
            {sidebar_css}

            <!-- External libraries -->
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="dashboard-layout">
                <!-- Sidebar container -->
                <aside class="sidebar glowing-section" id="sidebar">
                    {sidebar_nav_html}
                </aside>

                <!-- Main content -->
                <main id="section-container" class="glowing-dashboard-container">
                    {all_sections_html}
                </main>
            </div>

            <!-- Main dashboard script (for graphs, charts, etc.) -->
            <script>
                {js}
            </script>

            <!-- Sidebar positioning logic -->
            {sidebar_position_script}

            <!-- Sidebar navigation logic -->
            {sidebar_js}
        </body>
        </html>
        """

        import threading
        from http.server import HTTPServer, SimpleHTTPRequestHandler

        file_path = call.EXTERNALAPP_dashboard_renderer()  # This returns path to your HTML file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_html)

        # Step 2: Start a simple HTTP server in the directory of the HTML file
        directory = os.path.dirname(file_path)
        os.chdir(directory)

        # Serve on localhost:8000 or any free port
        PORT = 8000

        def start_server():
            httpd = HTTPServer(("localhost", PORT), SimpleHTTPRequestHandler)
            httpd.serve_forever()

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        # Step 3: Build URL to the dashboard HTML file
        dashboard_url = f"http://localhost:{PORT}/{os.path.basename(file_path)}"

        # Step 4: Embed with iframe pointing to HTTP URL (works better than local file path)
        components.iframe(dashboard_url, height=1000, scrolling=True)

    except Exception as e:
        st.error(f"Failed to render section: {e}")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def launch_self():
    """
       Relaunches the Streamlit dashboard script on a defined port (8502) in headless mode.

       - Ensures no other process is blocking the port.
       - Uses PyWebView if installed, otherwise opens in default browser.
       """

    port = 8502
    script = Path(__file__).resolve()

    # Kill any existing process on that port
    try:
        import psutil
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                psutil.Process(conn.pid).terminate()
    except Exception:
        pass

    # Set up and launch Streamlit
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', str(script), f'--server.port={port}'
    ])

    # Wait for Streamlit to start
    for _ in range(30):
        try:
            import socket
            socket.create_connection(('localhost', port), 1).close()
            break
        except:
            time.sleep(0.2)

    # Try to open in PyWebView, fallback to browser
    try:
        import webview
        webview.create_window("SulfurAI Dashboard", f"http://localhost:{port}")
        webview.start()
    except:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --- Main entrypoint ---
if __name__ == "__main__":
    try:
        ctx = get_script_run_ctx()
        if ctx:
            # ✅ Inside Streamlit — only run dashboard once
            if "dashboard_ran" not in st.session_state:
                st.session_state.dashboard_ran = True
                run_dashboard()
        else:
            # 🚀 Not inside Streamlit — relaunch self
            launch_self()
    except:
        # 🛡️ Fallback if anything goes wrong
        launch_self()


def create_expression_distribution_chart(df):
    """
     Creates a donut chart for distribution of expression types using plotly.

     Args:
         df (pd.DataFrame): DataFrame with a 'sentence_type' column.

     Returns:
         plotly.graph_objects.Figure: Configured pie chart showing type percentage.
     """
    # Calculate percentages for each expression type
    total_entries = len(df)
    type_counts = df['sentence_type'].value_counts()
    type_percentages = (type_counts / total_entries * 100).round(2)

    # Create a DataFrame with the percentages
    type_df = pd.DataFrame({
        'Expression Type': type_percentages.index,
        'Percentage': type_percentages.values
    })

    # Create donut chart with custom colors and styling
    fig = px.pie(type_df,
                 values='Percentage',
                 names='Expression Type',
                 title='Distribution of Expression Types (%)',
                 hole=0.3,
                 color_discrete_sequence=['#FF4B4B', '#39FF14', '#1E90FF', '#FFD700'])  # Custom colors

    # Update layout for better readability and styling
    fig.update_traces(textposition='inside',
                      textinfo='label+percent',
                      textfont=dict(color='white'))

    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=24, color='white'),
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot
        font=dict(color='white'),  # White text
        showlegend=False  # Hide legend for cleaner look
    )

    return fig


# In the main dashboard section where you display visualizations:
def main():
    """
    Dashboard tab controller — loads training data and builds visualization tabs.

    - Tab 1: Placeholder for training stats.
    - Tab 2: Shows expression type chart and metrics summary.

    Run this only in a Streamlit session context.
    """
    # Load the training data
    try:
        df = pd.read_csv(r'scripts/ai_renderer_2/training_data_sentences/data.csv')

        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Training Data Stats", "Expression Distribution"])

        with tab1:
            # Your existing visualizations
            pass  # placeholder for existing code

        with tab2:
            st.plotly_chart(create_expression_distribution_chart(df), use_container_width=True)

            # Add summary statistics
            st.markdown("### Expression Type Summary")
            total_entries = len(df)
            type_counts = df['sentence_type'].value_counts()
            type_percentages = (type_counts / total_entries * 100).round(2)

            cols = st.columns(len(type_counts))
            for i, (exp_type, percentage) in enumerate(type_percentages.items()):
                with cols[i]:
                    st.metric(
                        label=exp_type.capitalize(),
                        value=f"{percentage}%",
                        delta=f"{type_counts[exp_type]} entries"
                    )

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")