"""BigCodeBench tasks and the execution-based code verifier V(x, y)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

TEST_TIMEOUT = 30  # seconds per task


def load_tasks(split: str = "v0.1.4") -> list[dict]:
    """Load BigCodeBench from Hugging Face as a list of {task_id, prompt, test}."""
    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench", split=split)
    return [
        {"task_id": row["task_id"], "prompt": row["instruct_prompt"], "test": row["test"]}
        for row in ds
    ]


def verify(code: str, test: str, timeout: int = TEST_TIMEOUT) -> bool:
    """Return True iff `code` passes all unit tests in `test`.

    The generated code is executed as a separate Python process. Run this inside
    an isolated environment (e.g. a container): it executes model-generated code.
    """
    if not code.strip():
        return False
    program = f"{code}\n\n{test}"
    if "unittest.main" not in test:
        program += '\n\nif __name__ == "__main__":\n    import unittest\n    unittest.main()\n'
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir, "test_task.py")
        path.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=workdir,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0
