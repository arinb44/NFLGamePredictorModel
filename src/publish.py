"""Publish the dashboard's data bundle to a GitHub Release so the hosted dashboard
(Streamlit Community Cloud) can download it. Each run replaces the release's files,
so the repo's git history doesn't grow.

Usage:
    python -m src.publish          # build the bundle and upload it
"""
import subprocess
import sys

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, ROOT

RELEASE_TAG = "dashboard-data"
REPO = "arinb44/NFLGamePredictorModel"
RELEASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

# asset name -> local path. Everything the dashboard reads that isn't in git.
BUNDLE = {
    "games_features.parquet": PROCESSED_DIR / "games_features.parquet",
    "oos_predictions.parquet": PROCESSED_DIR / "oos_predictions.parquet",
    "missing_players.parquet": PROCESSED_DIR / "missing_players.parquet",
    "team_games.parquet": PROCESSED_DIR / "team_games.parquet",
    "qb_status.parquet": PROCESSED_DIR / "qb_status.parquet",
    "coverage_defense.parquet": PROCESSED_DIR / "coverage_defense.parquet",
    "coverage_types.parquet": PROCESSED_DIR / "coverage_types.parquet",
    "coverage_offense.parquet": PROCESSED_DIR / "coverage_offense.parquet",
    "blitz_team_games.parquet": PROCESSED_DIR / "blitz_team_games.parquet",
    "schedules.parquet": RAW_DIR / "schedules.parquet",
    "teams.csv": RAW_DIR / "teams.csv",
    "logistic.joblib": MODELS_DIR / "logistic.joblib",
    "logistic_blitz.joblib": MODELS_DIR / "logistic_blitz.joblib",
    "spread.joblib": MODELS_DIR / "spread.joblib",
}


def build():
    from src.matchups import blitz_team_games
    blitz_team_games().to_parquet(PROCESSED_DIR / "blitz_team_games.parquet", index=False)
    missing = [str(p) for p in BUNDLE.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing files (run the pipeline first): {missing}")


def upload():
    gh = ["gh", "release"]
    exists = subprocess.run(gh + ["view", RELEASE_TAG, "-R", REPO], capture_output=True).returncode == 0
    if not exists:
        subprocess.run(gh + ["create", RELEASE_TAG, "-R", REPO, "--title", "Dashboard data",
                             "--notes", "Data bundle read by the hosted dashboard. Replaced on every scheduled run; "
                                        "not a software release.", "--latest=false"], check=True)
    subprocess.run(gh + ["upload", RELEASE_TAG, "-R", REPO, "--clobber"] + [str(p) for p in BUNDLE.values()],
                   check=True)


def download(dest_root=ROOT, timeout: int = 60):
    """Download the bundle into the standard local locations (used by the hosted dashboard)."""
    import requests
    for name, path in BUNDLE.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{RELEASE_URL}/{name}", timeout=timeout)
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)


if __name__ == "__main__":
    build()
    upload()
    print(f"Published {len(BUNDLE)} files to {RELEASE_URL}", file=sys.stdout)
