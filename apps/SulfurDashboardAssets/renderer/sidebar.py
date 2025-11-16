# sidebar.py
"""
Sidebar module for SulfurAI Dashboard.
Applies existing glowing-section styles to the sidebar container and
sets up navigation links to switch between dashboard sections.

Requires that `style.css` and `script.js` (with glowing-section and section-ring logic)
are included globally in the dashboard.
Each sidebar item is styled as its own boxed element.

Usage:
    from sidebar import get_sidebar_includes
    css, html, script = get_sidebar_includes(sections)

    # Then embed into your dashboard HTML as raw includes.
"""


def _get_sidebar_css():
    """Return the sidebar CSS with glow-on-hover, glow-on-touch, and glow-on-focus"""
    return '''
    /* Sidebar placement */
    .dashboard-layout {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }

    /* Sidebar styling */
    .sidebar {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 240px;
        height: 50vh;
        padding: 20px 10px;
        z-index: 1000;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        align-items: stretch;
        transition: background-color 0.3s, box-shadow 0.3s;
    }

    /* Sidebar glow when active/focused/touched */
    .sidebar:active,
    .sidebar:focus-within,
    .sidebar.glow-on-touch {
        background-color: rgba(255,255,255,0.15) !important;
        box-shadow: 0 0 16px rgba(255,215,0,0.8) !important;
        outline: none;
    }

    /* List reset */
    .sidebar ul {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    /* Sidebar items */
    .sidebar ul li {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 6px;
        width: 200px;
        cursor: pointer;
        outline: none;
        transition: background-color 0.3s, box-shadow 0.3s;
    }

    /* Hover glow */
    .sidebar ul li:hover {
        background-color: rgba(255,255,255,0.1);
        box-shadow: 0 0 8px rgba(255,215,0,0.5);
    }

    /* Active/touch glow */
    .sidebar ul li:active,
    .sidebar ul li:focus-within,
    .sidebar ul li.glow-on-touch {
        background-color: rgba(255,255,255,0.15) !important;
        box-shadow: 0 0 12px rgba(255,215,0,0.7) !important;
    }

    /* Link styling */
    .sidebar ul li a {
        display: block;
        padding: 15px 16px;
        color: #FFD700;
        text-decoration: none;
        font-family: 'Roboto', sans-serif;
        font-size: 1rem;
        font-weight: 700;
    }

    .sidebar ul li a.active {
        color: #FFF;
        background-color: rgba(255,215,0,0.1);
    }

    /* Shift main container to the right */
    .glowing-dashboard-container,
    #section-container {
        margin-left: 260px !important;
        transition: margin-left 0.3s;
    }
    '''


def _get_sidebar_js():
    """Return the sidebar JavaScript with glow and section switching"""
    return '''
    document.addEventListener("DOMContentLoaded", () => {
        const sidebarLinks = document.querySelectorAll(".sidebar a");
        if (!sidebarLinks.length) return;

        const sidebar = document.querySelector('.sidebar');

        // Glow effect for sidebar container
        if (sidebar) {
            sidebar.addEventListener('touchstart',  () => sidebar.classList.add('glow-on-touch'));
            sidebar.addEventListener('touchend',    () => sidebar.classList.remove('glow-on-touch'));
            sidebar.addEventListener('touchcancel', () => sidebar.classList.remove('glow-on-touch'));
            sidebar.addEventListener('mouseenter',  () => sidebar.classList.add('glow-on-touch'));
            sidebar.addEventListener('mouseleave',  () => sidebar.classList.remove('glow-on-touch'));
        }

        function hideAllSections() {
            document.querySelectorAll(".dashboard-section")
                    .forEach(s => s.style.display = "none");
        }

        function updateDividers(targetIds) {
            const want = new Set(targetIds.map(id => id + "-divider"));
            document.querySelectorAll(".gradient-divider")
                    .forEach(div => div.style.display = want.has(div.id) ? "block" : "none");
        }

        function activate(link) {
            const ids = link.getAttribute("data-target").split(" ");
            hideAllSections();
            ids.forEach(id => {
                const s = document.getElementById(id);
                if (s) s.style.display = "block";
            });
            updateDividers(ids);

            sidebarLinks.forEach(a => a.classList.remove("active"));
            link.classList.add("active");
        }

        // Initialize first link
        activate(sidebarLinks[0]);

        // Glow effect per item
        sidebarLinks.forEach(link => {
            link.addEventListener("click", e => {
                e.preventDefault();
                activate(link);
                if (sidebar) sidebar.classList.remove('glow-on-touch');
            });

            link.addEventListener("touchstart",  () => link.parentElement.classList.add('glow-on-touch'));
            link.addEventListener("touchend",    () => link.parentElement.classList.remove('glow-on-touch'));
            link.addEventListener("touchcancel", () => link.parentElement.classList.remove('glow-on-touch'));
            link.addEventListener("mousedown",   () => link.parentElement.classList.add('glow-on-touch'));
            link.addEventListener("mouseup",     () => link.parentElement.classList.remove('glow-on-touch'));
            link.addEventListener("mouseleave",  () => link.parentElement.classList.remove('glow-on-touch'));
        });
    });
    '''


def _generate_sidebar_html(sections_nav):
    """Build sidebar HTML from sections_nav mapping."""
    html_items = []
    for sec_id, sec_data in sections_nav.items():
        targets = sec_data["target"] if isinstance(sec_data["target"], str) else " ".join(sec_data["target"])
        item = f'<li><a href="#" data-target="{targets}">{sec_data["name"]}</a></li>'
        html_items.append(item)

    html = [
        '<div class="dashboard-layout">',
        '    <div class="sidebar glowing-section" id="sidebar">',
        '        <ul>',
        '            ' + '\n            '.join(html_items),
        '        </ul>',
        '    </div>',
        '</div>'
    ]
    return '\n'.join(html)


def get_sidebar_includes(sections_nav):
    """Return CSS, HTML, and JS for sidebar navigation with glow effects."""
    css = f'<style>\n{_get_sidebar_css()}\n</style>'
    html = _generate_sidebar_html(sections_nav)
    js = f'<script>\n{_get_sidebar_js()}\n</script>'
    return css, html, js


def get_sidebar_position_script():
    return '''
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        const sidebar = document.getElementById('sidebar');
        const container = document.getElementById('section-container');
        const dashContainer = document.querySelector('.glowing-dashboard-container');

        if (sidebar) sidebar.style.left = '0px';
        if (container) container.style.marginLeft = '260px';
        if (dashContainer) dashContainer.style.marginLeft = '260px';
    });
    </script>
    '''




