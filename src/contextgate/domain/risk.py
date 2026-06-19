from __future__ import annotations

import re

from contextgate.domain.gateway import AbstentionReason, RiskReport


class RuleBasedRiskPolicy:
    """Fail-closed injection gate used before optional classifier integrations."""

    version = "rules-v1"
    _rules = {
        "ignore_instructions": re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|system|developer)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "prompt_exfiltration": re.compile(
            r"\b(reveal|show|print|repeat|extract)\b.{0,40}\b(system prompt|developer message|hidden instructions)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "role_override": re.compile(
            r"\b(you are now|act as)\b.{0,50}\b(system|developer|unrestricted)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    }

    def assess_query(self, text: str) -> RiskReport:
        return self._assess(text, AbstentionReason.UNSAFE_QUERY)

    def assess_contexts(self, contexts: list[str]) -> RiskReport:
        return self._assess("\n".join(contexts), AbstentionReason.UNSAFE_CONTEXT)

    def _assess(self, text: str, reason: AbstentionReason) -> RiskReport:
        matches = tuple(name for name, pattern in self._rules.items() if pattern.search(text))
        return RiskReport(
            score=1.0 if matches else 0.0,
            blocked=bool(matches),
            reason=reason if matches else None,
            matched_rules=matches,
        )
