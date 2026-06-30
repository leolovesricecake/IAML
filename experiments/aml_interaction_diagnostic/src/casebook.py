from pathlib import Path
from typing import List

from src.io_utils import read_jsonl


def _format_words(example) -> str:
    scores = example.get("word_attributions", [])
    words = example.get("words", [])
    rendered = []
    for idx, word in enumerate(words):
        score = scores[idx] if idx < len(scores) else 0.0
        rendered.append(f"**{word}**({score:.3f})" if score >= 0.5 else word)
    return " ".join(rendered)


def build_casebook(run_dir) -> str:
    """Build a Markdown casebook from per-example diagnostic output."""
    run_dir = Path(run_dir)
    examples = list(read_jsonl(run_dir / "per_example.jsonl"))
    sections = [
        ("High interaction", sorted(examples, key=lambda row: row.get("interaction_strength", 0.0), reverse=True)[:10]),
        ("Low interaction", sorted(examples, key=lambda row: row.get("interaction_strength", 0.0))[:10]),
        ("Worst faithfulness", sorted(examples, key=lambda row: row.get("faithfulness_error", 0.0), reverse=True)[:10]),
        ("Best faithfulness", sorted(examples, key=lambda row: row.get("faithfulness_error", 0.0))[:10]),
    ]
    lines: List[str] = ["# AML Interaction Diagnostic Casebook", ""]
    for title, rows in sections:
        lines.extend([f"## {title}", ""])
        for row in rows:
            lines.extend(
                [
                    f"### Example {row['id']}",
                    "",
                    f"- Bucket: `{row.get('bucket', 'unknown')}`",
                    f"- Predicted label: `{row.get('predicted_label')}`",
                    f"- Original target score: `{row.get('original_target_score')}`",
                    f"- Interaction strength: `{row.get('interaction_strength')}`",
                    f"- Text: {row.get('text', '')}",
                    f"- AML attribution: {_format_words(row)}",
                    f"- Top positive interactions: `{row.get('top_interactions_signed', [])[:5]}`",
                    f"- Top absolute interactions: `{row.get('top_interactions_absolute', [])[:5]}`",
                    "",
                ]
            )
    content = "\n".join(lines)
    (run_dir / "casebook.md").write_text(content, encoding="utf-8")
    return content
