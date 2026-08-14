"""Message construction. Every agent sees byte-identical input for a given task and role."""

from __future__ import annotations

from ..schemas import Role, Task

SYSTEM_PROMPT = (
    "You are answering an evaluation question. Follow the response format exactly. "
    "Be concise: no preamble, no restatement of the question."
)


def build_messages(
    task: Task, role: Role | None = None, transcript: str = ""
) -> list[dict[str, str]]:
    if role is None:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]
    content = role.prompt_template.format(
        transcript=transcript or "(no prior turns)", query=task.prompt
    )
    return [
        {"role": "system", "content": role.contract},
        {"role": "user", "content": content},
    ]


def verifier_ground_truth(worker_was_correct: bool) -> str:
    """The verifier is graded on agreeing with ground truth, not on the task answer itself."""
    return "ACCEPT" if worker_was_correct else "REVISE"
