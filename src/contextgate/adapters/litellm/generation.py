from __future__ import annotations

import re

from contextgate.config import Settings, get_settings
from contextgate.domain.gateway import AnswerResult, Citation
from contextgate.domain.retrieval import RetrievalResult


class AnswerGenerator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _extractive_answer(self, retrieval: RetrievalResult) -> str:
        if retrieval.abstained or not retrieval.hits:
            return "I could not find enough grounded evidence in the knowledge base."
        lines = [
            f"{hit.text[:500].strip()} [{index}]"
            for index, hit in enumerate(retrieval.hits[:3], start=1)
        ]
        return "\n\n".join(lines)

    def abstain(self, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(
            answer="I could not answer from grounded evidence in the knowledge base.",
            citations=[],
            retrieval=retrieval,
            provider="abstention",
            grounded=False,
        )

    def generate(
        self,
        retrieval: RetrievalResult,
        *,
        system_prompt: str | None = None,
    ) -> AnswerResult:
        citations = [
            Citation(index=index, chunk_id=hit.chunk_id, source=hit.source)
            for index, hit in enumerate(retrieval.hits, start=1)
        ]
        if retrieval.abstained:
            return self.abstain(retrieval)
        if not self.settings.llm_model:
            answer = self._extractive_answer(retrieval)
            return AnswerResult(
                answer=answer,
                citations=citations[:3],
                retrieval=retrieval,
                provider="extractive",
                grounded=True,
            )

        from litellm import completion

        context = "\n\n".join(
            f"[{index}] {hit.source}\n{hit.text}"
            for index, hit in enumerate(retrieval.hits, start=1)
        )
        prompt = system_prompt or (
            "Answer only from the supplied evidence. Cite every factual claim with [n]. "
            "If evidence is insufficient, say that you cannot answer."
        )
        response = completion(
            model=self.settings.llm_model,
            api_base=self.settings.llm_api_base,
            api_key=self.settings.llm_api_key,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Question: {retrieval.query}\n\nEvidence:\n{context}",
                },
            ],
            temperature=0,
        )
        answer = str(response.choices[0].message.content)
        cited_indices = {int(value) for value in re.findall(r"\[(\d+)]", answer)}
        valid_indices = set(range(1, len(retrieval.hits) + 1))
        grounded = bool(cited_indices) and cited_indices.issubset(valid_indices)
        selected_citations = [citation for citation in citations if citation.index in cited_indices]
        return AnswerResult(
            answer=answer,
            citations=selected_citations,
            retrieval=retrieval,
            provider=self.settings.llm_model,
            grounded=grounded,
        )
