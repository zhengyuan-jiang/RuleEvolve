# RuleEvolve

The official repository for the NeurIPS 2026 paper "Self-Evolving Coding Rules for AI Coding Agents".

[[Paper]](https://arxiv.org/abs/2610.00650)

RuleEvolve evolves the coding rules (e.g. `AGENTS.md`) of an AI coding agent automatically. It maintains a pool of `m` candidate rules and improves them iteratively:

1. **Mutation allocation.** `N` mutations are distributed across the pool with an OCBA-based strategy. Candidates with a higher score variance, or with a mean score close to the current best, receive more mutations.
2. **Mutator.** An LLM rewrites each selected candidate into a new variant.
3. **Judge.** Every candidate is scored on a mini-batch of `B` coding tasks. The score rewards shorter generated code and penalizes any drop in correctness relative to the agent without rules:

   `S = -|A(x, r)| - λ · E_x[ V(x, A(x, ∅)) - V(x, A(x, r)) ]_+`

4. **Pruning.** A new candidate inherits its parent's score history. The top `m` candidates by mean score are kept.

After `T` iterations, RuleEvolve returns the candidate with the highest mean score.

## Repository layout

| File | Content |
|---|---|
| `rule_evolve.py` | RuleEvolve: rule pool, mutator, judge, mutation allocation, CLI |
| `agents.py` | Coding-agent backends: OpenHands (rules via `AGENTS.md`) and OpenAI SDK (rules as system prompt) |
| `verifier.py` | BigCodeBench loading and execution-based verification |

## Setup

Python 3.12 or later is required.

```bash
pip install -r requirements.txt
```

The verifier runs the BigCodeBench unit tests, so the libraries those tests import must be installed too. See the [BigCodeBench repository](https://github.com/bigcode-project/bigcodebench) for its evaluation requirements.

> **Warning:** the agents and the verifier execute model-generated code on the local machine. Run them inside an isolated environment such as a container.

Set your API key:

```bash
export OPENAI_API_KEY=...
```

## Usage

```bash
python rule_evolve.py --initial-rule path/to/AGENTS.md --agent openhands --model gpt-4.1-mini
```

| Argument | Default | Description |
|---|---|---|
| `--initial-rule` | (required) | Initial coding rules |
| `--agent` | `openhands` | `openhands` or `openai` |
| `--model` | `gpt-4.1-mini` | Backbone LLM of the coding agent |
| `--agent-base-url` | none | Base URL for a self-hosted backbone, e.g. an Ollama server |
| `--mutator-model` | `gpt-5.1` | LLM used by the mutator |
| `-T`, `--iterations` | `10` | Number of iterations |
| `-N`, `--mutations` | `20` | Mutations per iteration |
| `-m`, `--pool-size` | `5` | Rule pool size |
| `-B`, `--batch-size` | `10` | Coding tasks per evaluation |
| `--allocation` | `ocba` | Mutation allocation: `ocba` (ours), or the `even` / `proportional` baselines |
| `--lam` | `3000` | Weight λ of the correctness penalty |
| `--eps` | `1e-6` | Minimum gap ε in the mutation allocation |
| `--train-size` | `1000` | BigCodeBench tasks `[0, train-size)` used for evolution |

The pool from each iteration is written to `runs/iter_XX/`, and the best rules to `runs/best_rule.md`.

The baseline allocation strategies use the same pipeline. `even` gives every candidate `N / m` mutations, and `proportional` allocates mutations in proportion to each candidate's mean score:

```bash
python rule_evolve.py --initial-rule path/to/AGENTS.md --allocation even
```

### Local backbone models

The coding agent can use a self-hosted model, for example Qwen3-Coder served by [Ollama](https://ollama.com). The mutator still uses the OpenAI API. The two agent backends expect different model names and base URLs:

```bash
# OpenHands
python rule_evolve.py --initial-rule path/to/AGENTS.md --agent openhands \
    --model ollama/qwen3-coder:30b --agent-base-url http://localhost:11434

# OpenAI SDK
python rule_evolve.py --initial-rule path/to/AGENTS.md --agent openai \
    --model qwen3-coder:30b --agent-base-url http://localhost:11434/v1
```

## Citation

```bibtex
@article{jiang2026self,
  title={Self-Evolving Coding Rules for AI Coding Agents},
  author={Jiang, Zhengyuan and Wang, Reachal and Hu, Yuepeng and Wang, Yupu and Jia, Yuqi and Gong, Neil Zhenqiang},
  journal={arXiv preprint arXiv:2610.00650},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
