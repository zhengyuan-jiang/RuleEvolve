"""RuleEvolve: self-evolving coding rules for AI coding agents.

RuleEvolve keeps a pool of m candidate coding rules. In every iteration it
allocates N mutations across the pool (OCBA-based strategic allocation), lets an
LLM mutator produce the new candidates, scores all m + N candidates with the
judge on a mini-batch of B coding tasks, and keeps the top m.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI

from agents import AGENTS, build_agent
from verifier import load_tasks, verify

MUTATION_GUIDANCE = (
    "Mutate this coding guidance for AI coding agent into one new variant: improve wording, "
    "reorder/combine steps, or vary style. Output exactly one complete guidance, no labels or separators."
)
NO_RULE = ""


# -----------------------------------------------------------------------------
# Rule pool
# -----------------------------------------------------------------------------
@dataclass
class Candidate:
    rule: str
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1)) if len(self.scores) >= 2 else 1.0


# -----------------------------------------------------------------------------
# Mutator M
# -----------------------------------------------------------------------------
class Mutator:
    """r_m = M(r, g): rewrites a coding rule under the mutation guidance g."""

    def __init__(self, model: str, temperature: float = 0.8, max_tokens: int = 4000):
        self.client = OpenAI(max_retries=5)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, rule: str) -> str | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MUTATION_GUIDANCE},
                    {"role": "user", "content": f"Mutate this coding guidance:\n\n{rule}"},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
        except Exception as e:
            print(f"  [mutator] {type(e).__name__}: {e}")
            return None
        return (response.choices[0].message.content or "").strip() or None


# -----------------------------------------------------------------------------
# Judge J
# -----------------------------------------------------------------------------
class Judge:
    """S = -|A(x, r)| - lambda * E_x[V(x, A(x, {})) - V(x, A(x, r))]_+   (Eq. 5)."""

    def __init__(self, agent, tasks: list[dict], lam: float, missing_output_length: int = 5000):
        self.agent = agent
        self.tasks = tasks
        self.lam = lam
        self.missing_output_length = missing_output_length

    def run(self, rule: str, task_ids: list[int]) -> tuple[list[int], list[bool]]:
        """Generate code for each task under `rule`; return code lengths and verifier outcomes."""
        lengths, passed = [], []
        for i in task_ids:
            task = self.tasks[i]
            try:
                code = self.agent.generate(rule, task["prompt"])
            except Exception as e:
                print(f"  [agent] task {task['task_id']}: {type(e).__name__}: {e}")
                code = ""
            lengths.append(len(code) if code.strip() else self.missing_output_length)
            passed.append(verify(code, task["test"]))
        return lengths, passed

    def score(self, rule: str, task_ids: list[int], baseline_passed: list[bool]) -> float:
        lengths, passed = self.run(rule, task_ids)
        penalty = np.mean([max(int(b) - int(p), 0) for b, p in zip(baseline_passed, passed)])
        return float(-np.mean(lengths) - self.lam * penalty)


# -----------------------------------------------------------------------------
# Strategic mutation allocation (Algorithm 2)
# -----------------------------------------------------------------------------
def allocate_mutations(pool: list[Candidate], n_mutations: int, eps: float = 1e-6) -> list[int]:
    mu = np.array([c.mean for c in pool])
    sigma = np.maximum([c.std for c in pool], 1e-12)
    b = int(np.argmax(mu))
    others = np.array([i for i in range(len(pool)) if i != b], dtype=int)

    w = np.zeros(len(pool))
    w[others] = (sigma[others] / np.maximum(mu[b] - mu[others], eps)) ** 2
    w[b] = sigma[b] * np.sqrt(np.sum(w[others] ** 2 / sigma[others] ** 2)) if len(others) else 1.0
    if w.sum() <= 0:
        w = np.ones(len(pool))
    return _round_to_total(n_mutations * w / w.sum(), n_mutations)


def _round_to_total(x: np.ndarray, total: int) -> list[int]:
    """Round non-negative reals to integers that sum to `total` (largest remainder)."""
    n = np.floor(x).astype(int)
    n[np.argsort(n - x)[: total - n.sum()]] += 1
    return n.tolist()


# -----------------------------------------------------------------------------
# RuleEvolve (Algorithm 1)
# -----------------------------------------------------------------------------
def rule_evolve(
    initial_rule: str,
    judge: Judge,
    mutator: Mutator,
    train_ids: list[int],
    pool_size: int = 5,
    n_mutations: int = 20,
    iterations: int = 10,
    batch_size: int = 10,
    eps: float = 1e-6,
    seed: int = 42,
    output_dir: Path | None = None,
) -> Candidate:
    rng = np.random.default_rng(seed)

    def sample_batch() -> tuple[list[int], list[bool]]:
        task_ids = rng.choice(train_ids, size=batch_size, replace=False).tolist()
        _, baseline_passed = judge.run(NO_RULE, task_ids)
        return task_ids, baseline_passed

    def evaluate(candidates: list[Candidate]) -> None:
        task_ids, baseline_passed = sample_batch()
        for c in candidates:
            c.scores.append(judge.score(c.rule, task_ids, baseline_passed))

    # Initialize the pool with m mutations of the initial rules and score it.
    print(f"Initializing pool with {pool_size} mutations of the initial rules")
    pool = [Candidate(mutator(initial_rule) or initial_rule) for _ in range(pool_size)]
    evaluate(pool)
    _save(pool, output_dir, 0)

    for t in range(1, iterations + 1):
        allocation = allocate_mutations(pool, n_mutations, eps)
        print(f"Iteration {t}/{iterations}: mutation allocation {allocation}")

        # Each new candidate inherits the score set of its parent.
        children = [
            Candidate(rule, list(parent.scores))
            for parent, n in zip(pool, allocation)
            for _ in range(n)
            if (rule := mutator(parent.rule))
        ]

        # Score all m + N candidates on a fresh mini-batch, then keep the top m.
        evaluate(pool + children)
        pool = sorted(pool + children, key=lambda c: c.mean, reverse=True)[:pool_size]
        print(f"  best mean score: {pool[0].mean:.1f}")
        _save(pool, output_dir, t)

    return max(pool, key=lambda c: c.mean)


def _save(pool: list[Candidate], output_dir: Path | None, iteration: int) -> None:
    if output_dir is None:
        return
    d = output_dir / f"iter_{iteration:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(pool):
        (d / f"rule_{k}.md").write_text(c.rule, encoding="utf-8")
    (d / "scores.json").write_text(json.dumps([c.scores for c in pool], indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="RuleEvolve: self-evolving coding rules for AI coding agents.")
    p.add_argument("--initial-rule", type=Path, required=True, help="Initial coding rules (e.g. an AGENTS.md).")
    p.add_argument("--agent", choices=sorted(AGENTS), default="openhands", help="Coding-agent framework.")
    p.add_argument("--model", default="gpt-4.1-mini", help="Backbone LLM of the coding agent.")
    p.add_argument("--agent-base-url", default=None, help="API base URL of the backbone LLM (e.g. a local server).")
    p.add_argument("--mutator-model", default="gpt-5.1", help="LLM used by the mutator.")
    p.add_argument("--iterations", "-T", type=int, default=10)
    p.add_argument("--mutations", "-N", type=int, default=20)
    p.add_argument("--pool-size", "-m", type=int, default=5)
    p.add_argument("--batch-size", "-B", type=int, default=10)
    p.add_argument("--lam", type=float, default=3000.0, help="Weight of the correctness penalty (lambda).")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--train-size", type=int, default=1000, help="Tasks [0, train-size) of BigCodeBench are used.")
    p.add_argument("--dataset-split", default="v0.1.4", help="BigCodeBench version on Hugging Face.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = p.parse_args()

    tasks = load_tasks(args.dataset_split)
    judge = Judge(build_agent(args.agent, args.model, args.agent_base_url), tasks, args.lam)
    best = rule_evolve(
        initial_rule=args.initial_rule.read_text(encoding="utf-8"),
        judge=judge,
        mutator=Mutator(args.mutator_model),
        train_ids=list(range(min(args.train_size, len(tasks)))),
        pool_size=args.pool_size,
        n_mutations=args.mutations,
        iterations=args.iterations,
        batch_size=args.batch_size,
        eps=args.eps,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "best_rule.md").write_text(best.rule, encoding="utf-8")
    print(f"Best rules (mean score {best.mean:.1f}) saved to {args.output_dir / 'best_rule.md'}")


if __name__ == "__main__":
    main()
