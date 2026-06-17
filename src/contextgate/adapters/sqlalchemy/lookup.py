from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from contextgate.adapters.sqlalchemy import KnowledgeBase


def get_knowledge_base(session: Session, identifier: str) -> KnowledgeBase:
    knowledge_base = session.scalar(
        select(KnowledgeBase).where(
            or_(KnowledgeBase.id == identifier, KnowledgeBase.slug == identifier)
        )
    )
    if knowledge_base is None:
        raise ValueError(f"Knowledge base not found: {identifier}")
    return knowledge_base
