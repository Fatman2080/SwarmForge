#!/usr/bin/env python3
"""6-Stage 管线评测 — 多候选生成 → 测试生成 → 验证 → 修复 → 共识选择 → 最终验证

用法:
    python3 eval/run_pipeline.py --model qwen2.5-coder:32b
    python3 eval/run_pipeline.py --model qwen2.5-coder:32b --candidates 5 --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "HumanEval.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def ollama_chat(
    messages: list[dict],
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    from urllib.request import Request, urlopen

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()
    req = Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"]


def extract_function_body(raw: str, prompt: str) -> str:
    text = raw.strip()
    for fence in ("```python", "```Python", "```py", "```"):
        if fence in text:
            parts = text.split(fence, 1)
            if len(parts) > 1:
                text = parts[1].split("```")[0]
                break

    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            body_start = i + 1
            break
        if stripped.startswith("def ") and ":" in stripped:
            for j in range(i, min(i + 5, len(lines))):
                if lines[j].rstrip().endswith(":"):
                    body_start = j + 1
                    break
            break

    lines = lines[body_start:]
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and cleaned:
            break
        if stripped.startswith("# Example") or stripped.startswith("# Test"):
            break
        if stripped.startswith("print(") and "(" in stripped:
            break
        if stripped.startswith("if __name__"):
            break
        if stripped.startswith("assert ") and cleaned:
            break
        cleaned.append(line)

    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    text = "\n".join(cleaned)
    if not text.strip():
        return "    pass"

    first_code_line = ""
    for line in cleaned:
        if line.strip():
            first_code_line = line
            break
    if first_code_line and not first_code_line.startswith((" ", "\t")):
        text = "\n".join("    " + line for line in cleaned)
    return text


def run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True, timeout=timeout
        )
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout)[-500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Stage 1: Multi-candidate generation
# ---------------------------------------------------------------------------

def stage1_generate(task: dict, model: str, base_url: str, n: int, temp: float) -> list[str]:
    candidates = []
    for _ in range(n):
        try:
            raw = ollama_chat(
                [
                    {"role": "system", "content": (
                        "You are an expert Python programmer. "
                        "Complete the function. Output ONLY the function body code, "
                        "no explanation, no markdown fences, no extra text. "
                        "Continue from where the prompt ends."
                    )},
                    {"role": "user", "content": task["prompt"]},
                ],
                model, base_url, temperature=temp, max_tokens=1024,
            )
            body = extract_function_body(raw, task["prompt"])
            candidates.append(body)
        except Exception:
            candidates.append("    pass")
    return candidates


# ---------------------------------------------------------------------------
# Stage 2: Test generation
# ---------------------------------------------------------------------------

def stage2_gen_tests(task: dict, model: str, base_url: str) -> str:
    """Generate auxiliary assert-based tests for the function."""
    prompt_text = task["prompt"]
    entry = task["entry_point"]
    try:
        raw = ollama_chat(
            [
                {"role": "system", "content": (
                    "You are an expert Python test engineer. "
                    "Given a function signature and docstring, write 5 assert-based test cases. "
                    "Output ONLY assert statements, one per line. No markdown, no explanation. "
                    "Example format:\nassert func([1,2]) == 3\nassert func([]) == 0"
                )},
                {"role": "user", "content": (
                    f"Write 5 assert-based tests for this function:\n\n{prompt_text}\n\n"
                    f"The function name is: {entry}"
                )},
            ],
            model, base_url, temperature=0.3, max_tokens=512,
        )
        lines = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.startswith("```"):
                continue
            if line.startswith("assert "):
                lines.append(line)
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Stage 3: Verification — run candidates against original + auxiliary tests
# ---------------------------------------------------------------------------

def stage3_verify(
    task: dict, candidates: list[str], aux_tests: str
) -> list[dict]:
    """Return list of {idx, body, orig_pass, aux_pass, orig_err}."""
    results = []
    prompt = task["prompt"]
    test_code = task["test"]
    entry = task["entry_point"]

    for idx, body in enumerate(candidates):
        full_orig = prompt + body + "\n\n" + test_code + f"\n\ncheck({entry})\n"
        orig_pass, orig_err = run_code(full_orig)

        aux_pass = True
        if aux_tests:
            full_aux = prompt + body + "\n\n" + aux_tests + "\n"
            aux_pass, _ = run_code(full_aux, timeout=5)

        results.append({
            "idx": idx,
            "body": body,
            "orig_pass": orig_pass,
            "aux_pass": aux_pass,
            "orig_err": orig_err,
            "total_pass": int(orig_pass) + int(aux_pass),
        })
    return results


# ---------------------------------------------------------------------------
# Stage 4: Repair — fix failing candidates using error feedback
# ---------------------------------------------------------------------------

def stage4_repair(
    task: dict, body: str, error: str, model: str, base_url: str
) -> str:
    """Try to repair a failing solution."""
    try:
        raw = ollama_chat(
            [
                {"role": "system", "content": (
                    "You are an expert Python debugger. "
                    "Fix the buggy function body. Output ONLY the corrected function body code, "
                    "no explanation, no markdown fences."
                )},
                {"role": "user", "content": (
                    f"Function signature:\n{task['prompt']}\n\n"
                    f"Current (buggy) implementation:\n{body}\n\n"
                    f"Error:\n{error}\n\n"
                    f"Fix this function body. Output ONLY code."
                )},
            ],
            model, base_url, temperature=0.2, max_tokens=1024,
        )
        return extract_function_body(raw, task["prompt"])
    except Exception:
        return body


# ---------------------------------------------------------------------------
# Stage 5: Consensus selection
# ---------------------------------------------------------------------------

def stage5_select(verified: list[dict]) -> dict:
    """Pick the best candidate: prefer orig_pass, then total_pass, then first."""
    orig_passers = [v for v in verified if v["orig_pass"]]
    if orig_passers:
        best = sorted(orig_passers, key=lambda x: -x["total_pass"])
        return best[0]

    aux_passers = [v for v in verified if v["aux_pass"]]
    if aux_passers:
        return aux_passers[0]

    return verified[0]


# ---------------------------------------------------------------------------
# Full pipeline for one task
# ---------------------------------------------------------------------------

def run_task_pipeline(
    task: dict, model: str, base_url: str, n_candidates: int, gen_temp: float,
    enable_tests: bool, enable_repair: bool, max_repairs: int,
) -> dict:
    task_id = task["task_id"]
    t0 = time.monotonic()
    stages_log = {}

    # Stage 1
    candidates = stage1_generate(task, model, base_url, n_candidates, gen_temp)
    stages_log["s1_candidates"] = len(candidates)

    # Stage 2
    aux_tests = ""
    if enable_tests:
        aux_tests = stage2_gen_tests(task, model, base_url)
    stages_log["s2_aux_tests"] = len(aux_tests.splitlines()) if aux_tests else 0

    # Stage 3
    verified = stage3_verify(task, candidates, aux_tests)
    orig_pass_count = sum(1 for v in verified if v["orig_pass"])
    stages_log["s3_orig_pass"] = orig_pass_count

    # Stage 4: repair if no candidate passed original tests
    repaired_pass = False
    if enable_repair and orig_pass_count == 0:
        best_fail = min(verified, key=lambda v: len(v.get("orig_err", "")))
        for attempt in range(max_repairs):
            repaired_body = stage4_repair(
                task, best_fail["body"], best_fail["orig_err"], model, base_url
            )
            full = (
                task["prompt"] + repaired_body + "\n\n"
                + task["test"] + f"\n\ncheck({task['entry_point']})\n"
            )
            passed, err = run_code(full)
            if passed:
                verified.append({
                    "idx": len(verified),
                    "body": repaired_body,
                    "orig_pass": True,
                    "aux_pass": True,
                    "orig_err": "",
                    "total_pass": 2,
                })
                repaired_pass = True
                break
            best_fail = {"body": repaired_body, "orig_err": err, "orig_pass": False, "aux_pass": False, "total_pass": 0, "idx": -1}
    stages_log["s4_repaired"] = repaired_pass

    # Stage 5: select
    selected = stage5_select(verified)
    stages_log["s5_selected_idx"] = selected["idx"]

    # Stage 6: final verification (redundant but explicit)
    final_code = (
        task["prompt"] + selected["body"] + "\n\n"
        + task["test"] + f"\n\ncheck({task['entry_point']})\n"
    )
    final_pass, final_err = run_code(final_code)

    elapsed = time.monotonic() - t0
    return {
        "task_id": task_id,
        "passed": final_pass,
        "stages": stages_log,
        "selected_body": selected["body"][:1000],
        "error": final_err if not final_pass else "",
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="6-Stage 管线评测")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--candidates", type=int, default=5, help="Stage1 候选数 (N)")
    parser.add_argument("--gen-temp", type=float, default=0.6, help="生成温度")
    parser.add_argument("--no-tests", action="store_true", help="禁用 Stage2 测试生成")
    parser.add_argument("--no-repair", action="store_true", help="禁用 Stage4 修复")
    parser.add_argument("--max-repairs", type=int, default=2, help="最大修复尝试次数")
    parser.add_argument("--workers", type=int, default=1, help="并行任务数")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 题")
    args = parser.parse_args()

    tasks = []
    with open(DATA_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    if args.limit:
        tasks = tasks[:args.limit]

    total = len(tasks)
    print(f"{'='*60}")
    print(f"  6-Stage 管线评测")
    print(f"  模型: {args.model}")
    print(f"  题目数: {total}")
    print(f"  每题候选: {args.candidates}")
    print(f"  生成温度: {args.gen_temp}")
    print(f"  测试生成: {'✅' if not args.no_tests else '❌'}")
    print(f"  修复: {'✅' if not args.no_repair else '❌'}")
    print(f"  并行: {args.workers}")
    print(f"{'='*60}")
    print()

    all_results = []
    passed_count = 0
    repaired_count = 0
    completed = 0
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_task_pipeline,
                task, args.model, args.base_url, args.candidates, args.gen_temp,
                not args.no_tests, not args.no_repair, args.max_repairs,
            ): task["task_id"]
            for task in tasks
        }

        for future in as_completed(futures):
            tid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "task_id": tid,
                    "passed": False,
                    "stages": {},
                    "error": str(e),
                    "elapsed_s": 0,
                }
            all_results.append(result)
            completed += 1
            if result["passed"]:
                passed_count += 1
            if result.get("stages", {}).get("s4_repaired"):
                repaired_count += 1

            if completed % 5 == 0 or completed == total:
                elapsed = time.monotonic() - start
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                print(
                    f"  [{completed:3d}/{total}] "
                    f"✅ {passed_count} ({passed_count/completed*100:.1f}%) "
                    f"🔧 修复成功 {repaired_count} "
                    f"⏱ {rate:.1f} 题/分钟"
                )

    total_elapsed = time.monotonic() - start

    # Breakdown
    s3_direct = sum(
        1 for r in all_results
        if r["passed"] and not r.get("stages", {}).get("s4_repaired")
    )

    print()
    print("=" * 60)
    print(f"  📊 6-Stage 管线结果")
    print(f"  模型: {args.model}")
    print(f"  题目数: {total}")
    print(f"  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}分钟)")
    print()
    print(f"  ✅ 最终通过: {passed_count}/{total} ({passed_count/total*100:.1f}%)")
    print(f"     ├─ 直接通过 (Stage3): {s3_direct}")
    print(f"     └─ 修复通过 (Stage4): {repaired_count}")
    print()

    if passed_count < total:
        failed = [r for r in all_results if not r["passed"]]
        print(f"  ❌ 未通过: {len(failed)} 题")
        for r in failed[:10]:
            err_preview = r.get("error", "")[:80].replace("\n", " ")
            print(f"     - {r['task_id']}: {err_preview}")
        if len(failed) > 10:
            print(f"     ... 还有 {len(failed)-10} 题")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace(":", "_").replace("/", "_")
    out = RESULTS_DIR / f"pipeline_{slug}_n{args.candidates}.json"
    summary = {
        "model": args.model,
        "pipeline": "6-stage",
        "candidates": args.candidates,
        "gen_temp": args.gen_temp,
        "tests_enabled": not args.no_tests,
        "repair_enabled": not args.no_repair,
        "total_tasks": total,
        "passed": passed_count,
        "pass_rate": round(passed_count / total * 100, 2),
        "direct_pass": s3_direct,
        "repaired_pass": repaired_count,
        "elapsed_s": round(total_elapsed, 1),
        "task_results": all_results,
    }
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
