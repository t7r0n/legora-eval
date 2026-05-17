# Security Review

## Scope

Local CLI, deterministic synthetic fixtures, DuckDB run store, and generated static HTML dashboard.

## Current Assessment

The project is designed to run offline. It has no server-side request handlers, no authentication surface, no subprocess execution path, no deserialization of executable formats, and no external network client in the application code.

## Controls

- Input fixtures are parsed as JSON with Pydantic validation.
- DuckDB writes use parameterized inserts.
- HTML dashboard rendering uses Jinja autoescaping and only reads from generated local summaries.
- `.gitignore` excludes run databases, output bundles, virtual environments, caches, and build products.
- The CLI writes only under `runs/`, `outputs/`, and `data/` inside the project directory.

## Focused Scan

Reviewed the application for command execution, unsafe deserialization, external network clients, secret material, broad filesystem writes, and generated HTML injection. The only file-read path is the suite JSON loader, which is validated into Pydantic models. There is no subprocess, shell, socket, HTTP client, pickle, dynamic import from user data, or server runtime in the package code.

## Attack-Path Analysis

The realistic attacker-controlled input is a local JSON suite file. Malicious suite text can influence generated reports, but Jinja autoescaping prevents HTML execution in the dashboard. Suite content cannot reach a shell, a network client, credential material, or a privileged write path. The run database and demo artifacts stay inside project-local `runs/` and `outputs/` directories.

## Review Status

Passed focused local security review on 2026-05-17. No high-impact attacker-reachable path identified.
