"""Build the GitHub Pages copy of the dashboard. The same Streamlit app runs in the visitor's
browser (stlite on Pyodide), so there is no server memory limit.

Usage:
    python -m src.pages        # writes _site/: index.html + bundle.zip

The browser can't download from GitHub releases or nflverse (no CORS headers), so everything the
dashboard reads goes into bundle.zip: its code, the git-tracked reports and the published data
bundle. The Pages workflow (.github/workflows/pages.yml) downloads that bundle and runs this.
"""
import json
import shutil
import subprocess
import zipfile

try:
    import tomllib  # Python 3.11+ (the Pages workflow uses 3.12)
except ModuleNotFoundError:  # older local Pythons: the toml package that Streamlit installs
    import toml as tomllib

from src.config import ROOT
from src.publish import BUNDLE

SITE = ROOT / "_site"
STLITE = "https://cdn.jsdelivr.net/npm/@stlite/browser@1.12.1/build"
TRACKED = ["app", "src", "reports", "config", "models", ".streamlit"]

# The theme comes from .streamlit/config.toml, so the hosted and Pages copies look the same.
THEME = {f"theme.{k}": v for k, v in tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"].items()}
ICONS = ["app/assets/favicon.png", "app/assets/apple-touch-icon.png"]

INDEX = f"""<!doctype html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <title>Fourth and Why</title>
  <link rel="icon" href="favicon.png" type="image/png" />
  <link rel="apple-touch-icon" href="apple-touch-icon.png" />
  <link rel="stylesheet" href="{STLITE}/stlite.css" />
  <style>html, body {{ background: {THEME["theme.backgroundColor"]}; color: {THEME["theme.textColor"]}; }}</style>
</head>
<body>
  <div id="root"></div>
  <script type="module">
    import {{ mount }} from "{STLITE}/stlite.js";
    mount({{
      entrypoint: "app/dashboard.py",
      archives: [{{ url: "./bundle.zip", format: "zip", options: {{}} }}],
      requirements: ["scikit-learn", "scipy", "joblib", "requests", "tzdata"],
      streamlitConfig: {json.dumps(THEME | {"client.toolbarMode": "viewer"})},
    }}, document.getElementById("root"));
  </script>
</body>
</html>
"""


def build():
    tracked = subprocess.run(["git", "ls-files", *TRACKED], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    files = [ROOT / f for f in tracked if f.endswith((".py", ".csv", ".json", ".toml", ".svg", ".png"))] + list(BUNDLE.values())
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing files (run the pipeline or src.publish.download first): {missing}")
    SITE.mkdir(exist_ok=True)
    with zipfile.ZipFile(SITE / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
    (SITE / "index.html").write_text(INDEX)
    for icon in ICONS:  # the browser tab and home-screen icons
        shutil.copy(ROOT / icon, SITE)
    size = (SITE / "bundle.zip").stat().st_size / 1e6
    print(f"Built {SITE}: {len(files)} files, bundle.zip {size:.1f} MB")


if __name__ == "__main__":
    build()
