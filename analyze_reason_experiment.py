from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a compact GitHub-style table without requiring tabulate."""
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    columns = list(text_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_results(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "reason_algorithm" not in df.columns and "ranking_score" in df.columns:
        df["reason_algorithm"] = df["ranking_score"]
    df["run_dir"] = str(run_dir)
    return df


def _read_details(run_dir: Path) -> pd.DataFrame:
    details_dir = run_dir / "scene_details"
    if not details_dir.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(details_dir.glob("scene_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        try:
            df["scene_id"] = int(path.stem.replace("scene_", ""))
        except ValueError:
            df["scene_id"] = path.stem.replace("scene_", "")
        df["detail_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "reason_algorithm" not in out.columns and "ranking_score" in out.columns:
        out["reason_algorithm"] = out["ranking_score"]
    out["run_dir"] = str(run_dir)
    return out


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    df = results.copy()
    if "cl_success" in df.columns:
        df["cl_success_num"] = df["cl_success"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["cl_success_num"] = False
    group_cols = ["model", "reason_algorithm", "branch"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        model, algo, branch = keys
        rows.append(
            {
                "model": model,
                "reason_algorithm": algo,
                "branch": branch,
                "num_targets": len(group),
                "success_rate": float(group["cl_success_num"].mean()) if "cl_success_num" in group else None,
                "avg_closed_loop_steps": float(pd.to_numeric(group.get("cl_num_steps"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_details(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    numeric_cols = [
        "P_s",
        "P_g",
        "P",
        "IG",
        "IG_normalized",
        "graspability",
        "cost",
        "score_legacy",
        "score_ig",
        "score_ig_graspability",
        "score_theory",
        "score",
    ]
    for col in numeric_cols:
        if col in details.columns:
            details[col] = pd.to_numeric(details[col], errors="coerce")
    group_cols = ["model", "reason_algorithm", "target_id"]
    keep_cols = [col for col in numeric_cols if col in details.columns]
    summary = (
        details.groupby(group_cols, dropna=False)[keep_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["num_candidate_rows"] = (
        details.groupby(group_cols, dropna=False).size().to_numpy()
    )
    return summary


def _summarize_reason_params(details: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    df = details.copy()
    if not results.empty:
        key_cols = ["model", "prior_prompt", "reason_algorithm", "scene_id", "target_id"]
        available_keys = [col for col in key_cols if col in df.columns and col in results.columns]
        branch_cols = available_keys + ["branch"]
        branch_df = results[branch_cols].drop_duplicates()
        df = df.merge(branch_df, on=available_keys, how="left")
    if "branch" not in df.columns:
        df["branch"] = "unknown"
    if "selected" in df.columns:
        df["selected_num"] = df["selected"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["selected_num"] = False
    numeric_cols = [
        "P_s",
        "P_g",
        "P",
        "IG",
        "IG_normalized",
        "graspability",
        "cost",
        "score_legacy",
        "score_ig",
        "score_ig_graspability",
        "score_theory",
        "score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    group_cols = ["model", "reason_algorithm", "branch"]
    keep_cols = [col for col in numeric_cols if col in df.columns]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        model, algo, branch = keys
        row = {
            "model": model,
            "reason_algorithm": algo,
            "branch": branch,
            "num_candidate_rows": int(len(group)),
            "num_selected_rows": int(group["selected_num"].sum()),
        }
        for col in keep_cols:
            row[f"avg_{col}"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize reason closed-loop comparison runs.")
    parser.add_argument("--root", default="runs_reason_compare")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    detail_frames = []
    for results_path in sorted(root.glob("*/*/*/results.csv")):
        run_dir = results_path.parent
        if not run_dir.is_dir():
            continue
        result_df = _read_results(run_dir)
        detail_df = _read_details(run_dir)
        result_frames.append(result_df)
        detail_frames.append(detail_df)

    results = pd.concat([df for df in result_frames if not df.empty], ignore_index=True) if result_frames else pd.DataFrame()
    details = pd.concat([df for df in detail_frames if not df.empty], ignore_index=True) if detail_frames else pd.DataFrame()

    if not results.empty:
        results.to_csv(out_dir / "all_results.csv", index=False)
    if not details.empty:
        details.to_csv(out_dir / "all_candidate_details.csv", index=False)

    result_summary = _summarize_results(results)
    detail_summary = _summarize_details(details)
    reason_param_summary = _summarize_reason_params(details, results)
    if not result_summary.empty:
        result_summary.to_csv(out_dir / "closed_loop_summary.csv", index=False)
    if not detail_summary.empty:
        detail_summary.to_csv(out_dir / "information_gain_summary.csv", index=False)
    if not reason_param_summary.empty:
        reason_param_summary.to_csv(out_dir / "reason_param_summary.csv", index=False)

    report = {
        "root": str(root),
        "out_dir": str(out_dir),
        "num_result_rows": int(len(results)),
        "num_detail_rows": int(len(details)),
        "closed_loop_summary": result_summary.to_dict(orient="records") if not result_summary.empty else [],
        "information_gain_summary_rows": int(len(detail_summary)),
        "reason_param_summary_rows": int(len(reason_param_summary)),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result_summary.empty:
        _write_markdown_table(result_summary, out_dir / "closed_loop_summary.md")
    if not detail_summary.empty:
        _write_markdown_table(detail_summary, out_dir / "information_gain_summary.md")
    if not reason_param_summary.empty:
        _write_markdown_table(reason_param_summary, out_dir / "reason_param_summary.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
