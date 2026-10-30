"""Coding-agent backends A_theta used by the RuleEvolve judge.

Each backend maps (coding rules, task description) to the generated code.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

SAVE_PROMPT = "\nPlease finish this task and save the code to a file named 'task.py'."
NO_RULE_SYSTEM_PROMPT = "You are a Python expert. Complete the task."


class OpenHandsAgent:
    """OpenHands agent. The coding rules are provided through AGENTS.md in the workspace."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, rule: str, prompt: str) -> str:
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.sdk.context import AgentContext
        from openhands.sdk.context.skills import load_project_skills
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        with tempfile.TemporaryDirectory() as workspace:
            if rule:
                Path(workspace, "AGENTS.md").write_text(rule, encoding="utf-8")
            agent = Agent(
                llm=LLM(model=self.model, base_url=self.base_url, api_key=self.api_key),
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                    Tool(name=TaskTrackerTool.name),
                ],
                agent_context=AgentContext(skills=load_project_skills(workspace)),
            )
            conversation = Conversation(agent=agent, workspace=workspace, visualizer=None)
            try:
                conversation.send_message(prompt + SAVE_PROMPT)
                conversation.run()
            finally:
                conversation.close()
            task_file = Path(workspace, "task.py")
            return task_file.read_text(encoding="utf-8") if task_file.exists() else ""


class OpenAIAgent:
    """Tool-calling agent built on the OpenAI SDK. The coding rules are the system prompt."""

    TOOL = {
        "type": "function",
        "function": {
            "name": "save_code_to_file",
            "description": "Save the generated Python code to a file. Call with filename 'task.py' "
                           "and the complete runnable code as content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename, e.g. task.py"},
                    "content": {"type": "string", "description": "Full code to save"},
                },
                "required": ["filename", "content"],
            },
        },
    }

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=5)

    def generate(self, rule: str, prompt: str) -> str:
        system = (rule or NO_RULE_SYSTEM_PROMPT) + (
            "\n\nYou must call save_code_to_file with filename 'task.py' and the full runnable code. "
            "Do not put code in the message—only use the tool."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt + SAVE_PROMPT},
            ],
            tools=[self.TOOL],
            tool_choice="auto",
        )
        for call in response.choices[0].message.tool_calls or []:
            if call.function.name != "save_code_to_file":
                continue
            args = json.loads(call.function.arguments)
            if args.get("filename") == "task.py":
                return args.get("content", "")
        return ""


AGENTS = {"openhands": OpenHandsAgent, "openai": OpenAIAgent}


def build_agent(name: str, model: str, base_url: str | None = None):
    return AGENTS[name](model=model, base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY"))
