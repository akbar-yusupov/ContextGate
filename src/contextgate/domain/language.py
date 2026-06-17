from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42

UZBEK_MARKERS = {
    "uchun",
    "bilan",
    "qanday",
    "qachon",
    "buyurtma",
    "to'lov",
    "to‘lov",
    "bekor",
    "mumkin",
    "kerak",
    "yetkazib",
    "qaytarish",
}


def detect_language(text: str) -> str:
    tokens = set(re.findall(r"[\w'‘]+", text.lower(), flags=re.UNICODE))
    if len(tokens & UZBEK_MARKERS) >= 1:
        return "uz"
    if re.search(r"[а-яё]", text.lower()):
        return "ru"
    try:
        language = detect(text)
    except LangDetectException:
        return "unknown"
    return language if language in {"en", "ru", "uz"} else "unknown"
