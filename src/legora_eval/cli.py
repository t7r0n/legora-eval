from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from legora_eval.dashboard import benchmark_summary, build_dashboard
from legora_eval.runner import export_demo_pack, init_demo, run_suite, verify_outputs

app = typer.Typer(help="Local legal-agent eval harness.")
console = Console()


@app.command("init-demo")
def init_demo_command(force: bool = typer.Option(False, "--force", help="Reset generated folders.")) -> None:
    console.print_json(data=init_demo(force=force))


@app.command("run")
def run_command(iterations: int = typer.Option(4, min=1, max=50, help="Times to repeat the suite.")) -> None:
    summary = run_suite(iterations=iterations)
    console.print_json(summary.model_dump_json(indent=2))


@app.command("verify")
def verify_command() -> None:
    report = verify_outputs()
    console.print_json(data=report)
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("dashboard")
def dashboard_command() -> None:
    path = build_dashboard()
    console.print(str(path))


@app.command("benchmark")
def benchmark_command() -> None:
    console.print_json(data=benchmark_summary())


@app.command("export-demo-pack")
def export_demo_pack_command() -> None:
    console.print(str(export_demo_pack()))


@app.command("inspect-db")
def inspect_db(path: Path = typer.Option(Path("runs/latest/results.duckdb"), help="DuckDB path.")) -> None:
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    rows = conn.execute(
        "select model, avg(citation_f1), avg(redline_delta), avg(jurisdiction_match), avg(plan_recall_at_k) "
        "from case_results group by model order by 2 desc"
    ).fetchall()
    conn.close()
    console.print(json.dumps(rows, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
