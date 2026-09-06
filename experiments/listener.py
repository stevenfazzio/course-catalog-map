"""The listener: a Claude model that answers the wayfinding lineups (step 2) and the name-intrusion items (step 3).

One structured-output call per question, retried on transient errors, with token usage accumulated so each run can
report what it cost. The model is Claude Sonnet 5, not the namer (Claude Opus 5): the no-self-preference rule from
Steven's Toponymy evaluation (TutteInstitute/toponymy discussion 177), where Sonnet was the listener for every
headline number. Results are cached to a JSONL file keyed by question id, so an interrupted run resumes without
paying twice.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

LISTENER_MODEL = "claude-sonnet-5"
LISTENER_EFFORT = "medium"
MAX_TOKENS = 8_000  # adaptive thinking is on by default on Sonnet 5 and counts against this
# USD per million tokens (input, output), claude-api skill table cached 2026-06-24; for the cost report only
PRICE_PER_MTOK = {"claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0), "claude-opus-5": (5.0, 25.0)}


def refused(answer: dict) -> bool:
    return "refused" in answer


class Listener:
    def __init__(self, model: str = LISTENER_MODEL, effort: str = LISTENER_EFFORT, cache_path: Path | None = None):
        self.model = model
        self.effort = effort
        self.client = anthropic.Anthropic(max_retries=6)  # ANTHROPIC_API_KEY from the environment
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        self.cache_path = cache_path
        # one entry per process that paid for calls, so the record can total the cost of a run resumed across processes
        self.runs_path = cache_path.with_suffix(".runs.json") if cache_path is not None else None
        self.cache: dict[str, dict] = {}
        self.cache_usage: dict[str, dict] = {}  # per-call tokens, so a resumed run still reports what it all cost
        self.run_ids: set[str] = set()
        self._lock = threading.Lock()
        if cache_path is not None and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["id"]] = rec["answer"]
                    if "usage" in rec:
                        self.cache_usage[rec["id"]] = rec["usage"]

    def ask(self, question_id: str, system: str, user: str, schema: dict) -> dict:
        """Structured answer to one question; served from the cache when the id was answered before."""
        with self._lock:
            self.run_ids.add(question_id)
        if question_id in self.cache:
            return self.cache[question_id]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}, "effort": self.effort},
        )
        if response.stop_reason == "refusal":
            # A safety classifier declined the request (HTTP 200). Recorded, not retried: the same input is declined
            # the same way, and a different listener for one item would mix instruments. Callers skip these.
            category = response.stop_details.category if response.stop_details else None
            answer = {"refused": category or "unspecified"}
        elif response.stop_reason != "end_turn":
            raise RuntimeError(f"{question_id}: stop_reason {response.stop_reason}")
        else:
            text = next(b.text for b in response.content if b.type == "text")
            answer = json.loads(text)
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += usage["input_tokens"]
            self.usage["output_tokens"] += usage["output_tokens"]
            self.usage["cache_read_input_tokens"] += response.usage.cache_read_input_tokens or 0
            self.cache[question_id] = answer
            self.cache_usage[question_id] = usage
            if self.cache_path is not None:
                with open(self.cache_path, "a") as fh:
                    fh.write(
                        json.dumps({"id": question_id, "answer": answer, "model": self.model, "usage": usage}) + "\n"
                    )
        return answer

    def ask_many(self, questions: list[tuple[str, str, str, dict]], workers: int = 8, label: str = "") -> list[dict]:
        """`questions` are (id, system, user, schema); answers come back in the same order."""
        t0 = time.time()
        done = 0
        results: list[dict | None] = [None] * len(questions)

        def run(i: int) -> None:
            nonlocal done
            results[i] = self.ask(*questions[i])
            with self._lock:
                done += 1
                if done % 50 == 0 or done == len(questions):
                    print(
                        f"  {label}{done}/{len(questions)} answered, {time.time() - t0:.0f}s, ${self.cost():.2f} so far"
                    )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(run, range(len(questions))))
        return results  # type: ignore[return-value]

    def cost(self, usage: dict | None = None) -> float:
        usage = self.usage if usage is None else usage
        p_in, p_out = PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        return (usage["input_tokens"] * p_in + usage["output_tokens"] * p_out) / 1e6

    def close(self) -> None:
        """Append this process's paid calls to the runs ledger. Call once, after the last question."""
        if self.runs_path is None or self.usage["calls"] == 0:
            return
        runs = json.loads(self.runs_path.read_text()) if self.runs_path.exists() else []
        runs.append(
            {
                "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "model": self.model,
                "calls": self.usage["calls"],
                "input_tokens": self.usage["input_tokens"],
                "output_tokens": self.usage["output_tokens"],
                "estimated_cost_usd": round(self.cost(), 2),
            }
        )
        self.runs_path.write_text(json.dumps(runs, indent=1))
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    def runs(self) -> list[dict]:
        return json.loads(self.runs_path.read_text()) if self.runs_path is not None and self.runs_path.exists() else []

    def run_usage(self) -> dict:
        """Tokens for every question asked in this run, including ones served from the cache of an earlier run."""
        known = [self.cache_usage[i] for i in self.run_ids if i in self.cache_usage]
        total = {
            "calls": len(self.run_ids),
            "calls_this_process": self.usage["calls"],
            "calls_with_usage_recorded": len(known),
            "input_tokens": int(sum(u["input_tokens"] for u in known)),
            "output_tokens": int(sum(u["output_tokens"] for u in known)),
        }
        total["estimated_cost_usd_for_recorded_calls"] = round(self.cost(total), 2)
        runs = self.runs()
        total["paid_runs"] = runs
        total["total_over_paid_runs"] = {
            "calls": int(sum(r["calls"] for r in runs)),
            "input_tokens": int(sum(r["input_tokens"] for r in runs)),
            "output_tokens": int(sum(r["output_tokens"] for r in runs)),
            "estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in runs), 2),
            "note": "sum of every process that paid for calls to this cache; entries flagged estimated were "
            "reconstructed from a run log rather than summed from per-call usage",
        }
        return total

    def describe(self) -> dict:
        return {
            "model": self.model,
            "effort": self.effort,
            "thinking": "adaptive (model default)",
            "sampling": "model default (Sonnet 5 takes no temperature parameter)",
            "anthropic_sdk": anthropic.__version__,
            "usage": self.run_usage(),
            "price_per_mtok_usd": PRICE_PER_MTOK.get(self.model),
        }


def have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
