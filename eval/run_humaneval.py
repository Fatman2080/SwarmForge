#!/usr/bin/env python3
"""HumanEval 评测脚本 — 用于测量 Ollama 模型在 HumanEval 上的 pass@k。

用法:
    python3 eval/run_humaneval.py --model qwen2.5:0.5b
    python3 eval/run_humaneval.py --model qwen2.5-coder:7b --samples 5 --temperature 0.6
    python3 eval/run_humaneval.py --model qwen2.5-coder:7b --samples 10 --workers 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

DATA_PATH = Path(__file__).parent / "data" / "HumanEval.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def ollama_generate(
    prompt: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Python programmer. "
                    "Complete the function. Output ONLY the function body code, no explanation, "
                    "no markdown fences, no extra text. Continue from where the prompt ends."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    req = Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"]


def extract_function_body(raw: str, prompt: str) -> str:
    """从模型输出中提取函数体，处理各种格式问题。"""
    text = raw.strip()

    for fence in ("```python", "```Python", "```py", "```"):
        if fence in text:
            parts = text.split(fence, 1)
            if len(parts) > 1:
                text = parts[1].split("```")[0]
                break

    lines = text.splitlines()

    # 如果模型输出了完整的函数定义，只取函数体
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            body_start = i + 1
            break
        if stripped.startswith("def ") and ":" in stripped:
            # 可能是多行 def
            for j in range(i, min(i + 5, len(lines))):
                if lines[j].rstrip().endswith(":"):
                    body_start = j + 1
                    break
            break

    lines = lines[body_start:]

    # 去掉尾部噪音
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

    # 去掉尾部空行
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    text = "\n".join(cleaned)

    if not text.strip():
        return "    pass"

    # 检查缩进：函数体需要至少 4 格缩进
    first_code_line = ""
    for line in cleaned:
        if line.strip():
            first_code_line = line
            break

    if first_code_line and not first_code_line.startswith((" ", "\t")):
        text = "\n".join("    " + line for line in cleaned)

    return text


def run_code_safely(code: str, timeout: int = 10) -> tuple[bool, str]:
    """在子进程中执行代码，返回 (passed, error_msg)。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout)[-500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def evaluate_single(
    task: dict[str, Any],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    sample_idx: int,
) -> dict[str, Any]:
    """对单个 HumanEval 题目的单次采样进行评测。"""
    task_id = task["task_id"]
    prompt = task["prompt"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    started = time.monotonic()
    try:
        raw_output = ollama_generate(
            prompt, model, base_url, temperature, max_tokens,
        )
    except Exception as e:
        return {
            "task_id": task_id,
            "sample": sample_idx,
            "passed": False,
            "error": f"generation_failed: {e}",
            "latency_ms": int((time.monotonic() - started) * 1000),
        }

    gen_ms = int((time.monotonic() - started) * 1000)

    function_body = extract_function_body(raw_output, prompt)
    full_code = prompt + function_body + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    passed, error = run_code_safely(full_code)

    return {
        "task_id": task_id,
        "sample": sample_idx,
        "passed": passed,
        "error": error if not passed else "",
        "raw_output": raw_output[:2000],
        "extracted_body": function_body[:1000],
        "latency_ms": gen_ms,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    """计算 pass@k 指标。n=总采样数, c=通过数, k=选取数。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def main():
    parser = argparse.ArgumentParser(description="HumanEval 评测")
    parser.add_argument("--model", required=True, help="Ollama 模型名")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--samples", type=int, default=1, help="每题采样次数 (n)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 题 (0=全部)")
    args = parser.parse_args()

    tasks = []
    with open(DATA_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    if args.limit:
        tasks = tasks[:args.limit]

    total_tasks = len(tasks)
    total_samples = total_tasks * args.samples
    print(f"模型: {args.model}")
    print(f"题目数: {total_tasks}")
    print(f"每题采样: {args.samples}")
    print(f"总采样数: {total_samples}")
    print(f"temperature: {args.temperature}")
    print(f"并行 workers: {args.workers}")
    print()

    all_results: list[dict[str, Any]] = []
    completed = 0
    passed_count = 0

    jobs = [
        (task, sample_idx)
        for task in tasks
        for sample_idx in range(args.samples)
    ]

    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate_single,
                task, args.model, args.base_url,
                args.temperature, args.max_tokens, sample_idx,
            ): (task["task_id"], sample_idx)
            for task, sample_idx in jobs
        }

        for future in as_completed(futures):
            task_id, sample_idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "task_id": task_id,
                    "sample": sample_idx,
                    "passed": False,
                    "error": str(e),
                    "latency_ms": 0,
                }
            all_results.append(result)
            completed += 1
            if result["passed"]:
                passed_count += 1

            if completed % 10 == 0 or completed == total_samples:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{total_samples}] "
                    f"通过: {passed_count} "
                    f"({passed_count/completed*100:.1f}%) "
                    f"速度: {rate:.1f} samples/s"
                )

    total_elapsed = time.monotonic() - start_time

    task_pass_counts: dict[str, int] = {}
    task_sample_counts: dict[str, int] = {}
    for r in all_results:
        tid = r["task_id"]
        task_sample_counts[tid] = task_sample_counts.get(tid, 0) + 1
        if r["passed"]:
            task_pass_counts[tid] = task_pass_counts.get(tid, 0) + 1

    pass_1_values = []
    pass_5_values = []
    pass_10_values = []
    for tid in sorted(task_sample_counts.keys()):
        n = task_sample_counts[tid]
        c = task_pass_counts.get(tid, 0)
        pass_1_values.append(pass_at_k(n, c, 1))
        if n >= 5:
            pass_5_values.append(pass_at_k(n, c, 5))
        if n >= 10:
            pass_10_values.append(pass_at_k(n, c, 10))

    p1 = sum(pass_1_values) / len(pass_1_values) * 100
    p5 = sum(pass_5_values) / len(pass_5_values) * 100 if pass_5_values else 0
    p10 = sum(pass_10_values) / len(pass_10_values) * 100 if pass_10_values else 0

    task_passed = sum(1 for c in task_pass_counts.values() if c > 0)
    latencies = [r["latency_ms"] for r in all_results if r["latency_ms"] > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print()
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  题目数: {total_tasks}")
    print(f"  每题采样: {args.samples}")
    print(f"  temperature: {args.temperature}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  平均延迟: {avg_latency:.0f}ms")
    print()
    print(f"  pass@1:  {p1:.1f}%")
    if pass_5_values:
        print(f"  pass@5:  {p5:.1f}%")
    if pass_10_values:
        print(f"  pass@10: {p10:.1f}%")
    print()
    print(f"  至少通过 1 次的题目: {task_passed}/{total_tasks} ({task_passed/total_tasks*100:.1f}%)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace(":", "_").replace("/", "_")
    result_file = RESULTS_DIR / f"{model_slug}_n{args.samples}_t{args.temperature}.json"

    summary = {
        "model": args.model,
        "total_tasks": total_tasks,
        "samples_per_task": args.samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "total_elapsed_s": round(total_elapsed, 1),
        "avg_latency_ms": round(avg_latency),
        "pass_at_1": round(p1, 2),
        "pass_at_5": round(p5, 2) if pass_5_values else None,
        "pass_at_10": round(p10, 2) if pass_10_values else None,
        "tasks_passed_any": task_passed,
        "task_results": all_results,
    }
    with open(result_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()
