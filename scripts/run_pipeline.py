"""Run the full GridPulse-TFT pipeline: ingest -> preprocess -> train -> export.

Each stage runs as a subprocess of the same interpreter (equivalent to the
individual ``make`` targets), fails fast on the first error, and prints a
timing summary at the end.

With --serve, the FastAPI server (:8000) and Streamlit dashboard (:8501)
are started once the pipeline finishes; Ctrl+C stops both.

Examples:
    python scripts/run_pipeline.py                       # full pipeline
    python scripts/run_pipeline.py --serve               # pipeline, then API + dashboard
    python scripts/run_pipeline.py --days 30 --no-wandb  # 30 days of data, no W&B
    python scripts/run_pipeline.py --skip-ingest         # reuse raw data on disk
    python scripts/run_pipeline.py --skip-train --skip-export
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_key(key: str) -> str:
    """Read a single key from .env without requiring python-dotenv."""
    if os.environ.get(key):
        return os.environ[key]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def run_stage(name: str, module: str, args: list[str]) -> float:
    """Run one pipeline stage; exit the process with its return code on failure."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n=== [{name}] {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        print(f"\nPipeline failed at stage '{name}' (exit {result.returncode}).", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"=== [{name}] done in {elapsed:.1f}s", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, help="days of raw data to preprocess")
    parser.add_argument("--epochs", type=int, default=None, help="override training epochs")
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    parser.add_argument("--skip-ingest", action="store_true", help="skip grid + weather ingestion")
    parser.add_argument("--skip-weather", action="store_true", help="skip weather ingestion only")
    parser.add_argument("--skip-train", action="store_true", help="stop after preprocessing")
    parser.add_argument("--skip-export", action="store_true", help="skip ONNX export")
    parser.add_argument(
        "--serve", action="store_true", help="start the API and dashboard after the pipeline"
    )
    args = parser.parse_args()

    timings: list[tuple[str, float]] = []

    if not args.skip_ingest:
        timings.append(("ingest:grid", run_stage("ingest:grid", "src.ingestion.grid_client", [])))
        skip_weather = args.skip_weather
        if not skip_weather and not load_dotenv_key("MET_OFFICE_API_KEY"):
            print(
                "\nMET_OFFICE_API_KEY is not set - skipping weather ingestion "
                "(the model trains without weather features; see .env.example)."
            )
            skip_weather = True
        if not skip_weather:
            timings.append(
                ("ingest:weather", run_stage("ingest:weather", "src.ingestion.weather_client", []))
            )

    # src.processing.features cleans raw data and writes the feature parquet.
    timings.append(
        (
            "preprocess",
            run_stage("preprocess", "src.processing.features", ["--days", str(args.days)]),
        )
    )

    if not args.skip_train:
        train_args = []
        if args.epochs is not None:
            train_args += ["--epochs", str(args.epochs)]
        if args.no_wandb:
            train_args.append("--no-wandb")
        timings.append(("train", run_stage("train", "src.training.trainer", train_args)))

        if not args.skip_export:
            timings.append(("export", run_stage("export", "src.training.export", [])))

    total = sum(t for _, t in timings)
    print("\n=== Pipeline complete ===")
    for name, elapsed in timings:
        print(f"  {name:<16}{elapsed:>8.1f}s")
    print(f"  {'total':<16}{total:>8.1f}s")
    if args.serve:
        serve()
    else:
        print("\nNext: make serve   (or: python scripts/run_pipeline.py --serve --skip-ingest)")
        print("      streamlit run dashboard/app.py")


def serve() -> None:
    """Run the FastAPI server and Streamlit dashboard until Ctrl+C."""
    api_cmd = [sys.executable, "-m", "uvicorn", "src.serving.api:app", "--port", "8000"]
    # Headless skips Streamlit's interactive first-run email prompt.
    dash_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.headless=true",
    ]
    dash_env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}

    print("\n=== Starting API on :8000 and dashboard on :8501 (Ctrl+C to stop) ===", flush=True)
    api = subprocess.Popen(api_cmd, cwd=REPO_ROOT)
    dash = subprocess.Popen(dash_cmd, cwd=REPO_ROOT, env=dash_env)
    try:
        # Exit when either process dies so a crashed server isn't silently masked.
        while api.poll() is None and dash.poll() is None:
            time.sleep(1)
        for proc, name in ((api, "API"), (dash, "dashboard")):
            if proc.poll() is not None:
                print(f"{name} exited with code {proc.returncode}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        for proc in (api, dash):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
