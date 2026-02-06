import re
from typing import List, Optional
from urllib.parse import urlparse

# OpenAI 사용은 선택(없어도 동작)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# 본문 확인(조건부)용
import requests
from bs4 import BeautifulSoup


# =========================
# OpenAI client
# =========================
def _get_client():
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


# =========================
# Helpers
# =========================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s+")


def _is_en(language: str) -> bool:
    return (language or "ko").strip().lower().startswith("en")


def _norm_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = re.sub(r"[\"'“”‘’]", "", s)
    return s


def _is_image_file_url(url: str) -> bool:
    try:
        path = urlparse(url or "").path.lower()
    except Exception:
        path = (url or "").lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))


def _is_meaningless_summary(summary: str) -> bool:
    """
    summary가 사실상 '내용 없음'에 가까운 문구인지 판별(보수적).
    """
    s = _norm_text(summary).lower()
    if not s:
        return True

    meaningless_patterns = [
        "자세한 내용", "자세히 보기", "자세히보기",
        "기사 보기", "기사보기", "원문 보기", "원문보기",
        "더보기", "보기", "바로가기",
        "사진", "이미지", "영상", "동영상",
        "관련 기사", "관련기사",
        "클릭", "확인",
    ]
    return any(p in s for p in meaningless_patterns)


def _is_summary_same_as_title(title: str, summary: str) -> bool:
    t = _norm_text(title).lower()
    s = _norm_text(summary).lower()
    if not t or not s:
        return False
    return t == s or t in s or s in t


def _fetch_html(url: str, timeout: int = 10) -> Optional[str]:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def _extract_text_and_imgcount(html: str) -> (str, int):
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    img_count = len(soup.find_all("img"))
    return text, img_count


def _is_image_only_ad_page(body_text: str, img_count: int) -> bool:
    body = _norm_text(body_text)
    return len(body) < 40 and img_count >= 1


def _enforce_sentence_and_length(text: str, max_sentences: int, max_chars: int) -> str:
    """
    - 모델이 길게 쓰거나 문장 수가 늘어나는 경우를 방지하기 위한 최종 안전망.
    - 1~3문장 범위로만 잘라서 반환 (가능한 한 원문 보존).
    """
    s = _norm_text(text)
    if not s:
        return s

    parts = [p.strip() for p in _SENT_SPLIT_RE.split(s) if p.strip()]
    if parts:
        s = " ".join(parts[:max_sentences]).strip()

    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s


def _auto_sentence_target(n_items: int) -> int:
    # 기존 정책: 2~3문장 (기사 수가 늘어도 3문장 유지)
    if n_items <= 3:
        return 2
    return 3


# =========================
# Prompts (KO/EN)
# =========================
def _prompt_title_only(title: str, language: str) -> str:
    if _is_en(language):
        return f"""
You are writing a factual daily newsletter summary for executives in the contact lens / optical industry.

ABSOLUTE RULES (MOST IMPORTANT):
- Use ONLY what is explicitly stated in the Title.
- Do NOT add any facts, numbers, entities, brands, causes, outcomes, or interpretations not present.
- No exaggeration, no speculation, no forecasting.
- Keep proper nouns as-is (Korean names/brands are allowed as proper nouns).
- Output MUST be in English.
- Output 2–3 short sentences.

[Title]
{title}

[Output]
""".strip()

    return f"""
너는 콘택트렌즈/안경 업계 데일리 뉴스레터를 임원에게 보고하는 비서다.
아래 [제목]만을 근거로 2~3문장 요약을 작성하라.

🚫 절대 규칙:
- 제목에 없는 사실/숫자/주체/브랜드/원인/결과 절대 추가 금지
- 과장/추측/전망/평가 금지
- '출시'라는 단어가 제목에 명확히 있는 경우만 사용
- 2~3문장

[제목]
{title}

[출력]
""".strip()


def _prompt_compress_long_summary(title: str, summary: str, language: str) -> str:
    if _is_en(language):
        return f"""
You are writing a factual daily newsletter summary for executives in the contact lens / optical industry.

ABSOLUTE RULES (MOST IMPORTANT):
- Use ONLY what is explicitly stated in the Input Summary.
- Do NOT add any facts, numbers, entities, brands, causes, outcomes, or interpretations not present.
- No exaggeration, no speculation, no forecasting.
- Keep proper nouns as-is (Korean names/brands are allowed as proper nouns).
- Output MUST be in English.
- Output 2–3 short sentences.

[Title]
{title}

[Input Summary]
{summary}

[Output]
""".strip()

    return f"""
너는 콘택트렌즈/안경 업계 데일리 뉴스레터를 임원에게 보고하는 비서다.
아래 [제목/요약]을 근거로 '긴 요약을 2~3문장으로 압축'하라.

🚫 절대 규칙:
- 입력에 없는 사실/숫자/주체/브랜드/원인/결과 절대 추가 금지
- 과장/추측/전망/평가 금지
- '출시 예정'인 경우만 '출시'라는 단어 사용
- 2~3문장

[제목]
{title}

[요약]
{summary}

[출력]
""".strip()


def _prompt_summarize_from_body(title: str, body_text: str, language: str) -> str:
    if _is_en(language):
        return f"""
You are writing a factual daily newsletter summary for executives in the contact lens / optical industry.

ABSOLUTE RULES (MOST IMPORTANT):
- Use ONLY what is explicitly stated in the Article Body.
- Do NOT add any facts, numbers, entities, brands, causes, outcomes, or interpretations not present.
- No exaggeration, no speculation, no forecasting.
- Keep proper nouns as-is (Korean names/brands are allowed as proper nouns).
- Output MUST be in English.
- Output 2–3 short sentences.

[Title]
{title}

[Article Body]
{body_text}

[Output]
""".strip()

    return f"""
너는 콘택트렌즈/안경 업계 데일리 뉴스레터를 임원에게 보고하는 비서다.
아래 [기사 본문]만을 근거로 2~3문장 요약을 작성하라.

🚫 절대 규칙:
- 기사에 없는 사실/숫자/주체/브랜드/원인/결과를 절대 추가하지 말 것
- 과장/추측/전망/평가 금지
- '출시 예정'인 경우만 '출시'라는 단어 사용
- 안경테/렌즈/제품의 브랜드명은 본문에 명확히 언급된 경우에만 사용
- 브랜드가 불명확하면 특정 주체를 단정하지 말 것
- 2~3문장

[제목]
{title}

[기사 본문]
{body_text}

[출력]
""".strip()


def _call_openai(client, prompt: str, temperature: float = 0.2) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (r.choices[0].message.content or "").strip()


def _ensure_min_chars_english(summary: str, title: str, min_chars: int, max_chars: int, client) -> str:
    """
    영어 요약이 너무 짧아 should_exclude_article()의 '요약 짧음' 필터(<40)에 걸리는 문제 방지용.
    ✅ 팩트 추가 없이 '같은 의미를 더 풀어서' 쓰도록 재작성한다.
    """
    s = _norm_text(summary)
    if len(s) >= min_chars:
        return s

    # 1) AI로 동일 의미 확장
    if client is not None:
        try:
            prompt = f"""Rewrite the following summary in English.
Rules:
- Keep EXACTLY the same meaning; do NOT add any new facts.
- Keep it 2–3 sentences.
- Make it at least {min_chars} characters but not more than {max_chars} characters.
- Keep proper nouns as-is (Korean names/brands are allowed as proper nouns).
Input:
Title: {title}
Summary: {s}
Output:"""
            s2 = _call_openai(client, prompt, temperature=0.2)
            s2 = _enforce_sentence_and_length(s2, max_sentences=3, max_chars=max_chars)
            if len(_norm_text(s2)) >= min_chars:
                return _norm_text(s2)
        except Exception:
            pass

    # 2) 최후 수단: 제목을 괄호로 덧붙여 길이 확보 (팩트 추가 없음)
    if title:
        suffix = f" (Title: {title})"
        out = (s + suffix).strip()
        if len(out) > max_chars:
            out = out[:max_chars].rstrip() + "…"
        return out

    return s


def _fallback_overall(language: str = "ko") -> str:
    if _is_en(language):
        return "A briefing could not be generated due to missing AI access; please refer to the article list below."
    return "AI 요약을 생성할 수 없어 기사 목록만 공유드립니다."


# =========================
# A. 기사별 summary 정제/생성
# =========================
def refine_article_summaries(articles: List, language: str = "ko") -> None:
    """
    ✅ 각 기사 summary 정책(핵심 로직 유지)
    - 영어 모드에서 요약이 너무 짧아 기사 자체가 제외되는 문제를 막기 위해
      (팩트 추가 없이) '같은 의미로 더 풀어쓰는' 최소 길이 보정만 추가.
    """
    client = _get_client()

    if _is_en(language):
        LONG_SUMMARY_THRESHOLD = 260
        MAX_SUMMARY_CHARS = 220
        MIN_SUMMARY_CHARS = 60
    else:
        LONG_SUMMARY_THRESHOLD = 150
        MAX_SUMMARY_CHARS = 105
        MIN_SUMMARY_CHARS = 0

    for a in articles:
        title = _norm_text(getattr(a, "title", "") or "")
        summary_raw = getattr(a, "summary", "") or ""
        summary = _norm_text(summary_raw)
        link = (getattr(a, "link", "") or "").strip()

        # ✅ 네이버(OpenAPI 포함) 판별 플래그
        is_naver = bool(getattr(a, "is_naver", False))

        # 링크가 이미지 파일이면: 광고/배너로 보고 summary는 빈값
        if _is_image_file_url(link):
            a.summary = ""
            continue

        # 3) summary 없음/무의미 -> 본문 확인
        if (not summary) or _is_meaningless_summary(summary):
            html = _fetch_html(link)
            if not html:
                a.summary = ""
                continue

            body_text, img_count = _extract_text_and_imgcount(html)

            # 3-1) 이미지만 광고 -> 빈값
            if _is_image_only_ad_page(body_text, img_count):
                a.summary = ""
                continue

            # 3-2) 본문 텍스트 -> AI 요약(가능하면)
            if client is not None:
                try:
                    prompt = _prompt_summarize_from_body(title, body_text, language)
                    summary = _call_openai(client, prompt, temperature=0.2)
                except Exception:
                    summary = _norm_text(body_text)
            else:
                summary = _norm_text(body_text)

            summary = _enforce_sentence_and_length(summary, max_sentences=3, max_chars=MAX_SUMMARY_CHARS)
            if _is_en(language) and MIN_SUMMARY_CHARS:
                summary = _ensure_min_chars_english(summary, title, MIN_SUMMARY_CHARS, MAX_SUMMARY_CHARS, client)

            a.summary = summary
            continue

        # 2) summary == title -> 제목 정보만으로 2~3문장(추측 절대 금지)
        if _is_summary_same_as_title(title, summary):
            if client is not None:
                try:
                    prompt = _prompt_title_only(title, language)
                    summary = _call_openai(client, prompt, temperature=0.2)
                except Exception:
                    summary = title
            else:
                summary = title

            summary = _enforce_sentence_and_length(summary, max_sentences=3, max_chars=MAX_SUMMARY_CHARS)
            if _is_en(language) and MIN_SUMMARY_CHARS:
                summary = _ensure_min_chars_english(summary, title, MIN_SUMMARY_CHARS, MAX_SUMMARY_CHARS, client)

            a.summary = summary
            continue

        # 1) summary가 길면 -> 압축 요약
        if len(summary) >= LONG_SUMMARY_THRESHOLD:
            if client is not None:
                try:
                    prompt = _prompt_compress_long_summary(title, summary, language)
                    summary = _call_openai(client, prompt, temperature=0.2)
                except Exception:
                    pass

        # ✅ 네이버는 길이와 상관없이 한번 더 AI 정리(기존 정책 유지)
        if is_naver and client is not None:
            try:
                prompt = _prompt_compress_long_summary(title, summary, language)
                summary = _call_openai(client, prompt, temperature=0.2)
            except Exception:
                pass

        summary = _enforce_sentence_and_length(summary, max_sentences=3, max_chars=MAX_SUMMARY_CHARS)
        if _is_en(language) and MIN_SUMMARY_CHARS:
            summary = _ensure_min_chars_english(summary, title, MIN_SUMMARY_CHARS, MAX_SUMMARY_CHARS, client)

        a.summary = summary


# =========================
# B. 상단 전체 요약
# =========================
def summarize_overall(articles: List, language: str = "ko") -> str:
    """
    ✅ 임원용 "어제 기사 AI 브리핑"
    - 입력(제목/요약) 범위 내에서만 이슈 단위로 묶어 요약
    - 영어 모드에서 너무 짧게 잘려 빈약해지는 문제를 줄이기 위해 char limit만 현실적으로 조정
    """
    if not articles:
        if _is_en(language):
            return "There were no relevant articles collected for yesterday, so there is nothing additional to brief."
        return "어제 기준으로 수집된 관련 기사가 없어 별도 공유 사항은 없습니다."

    client = _get_client()
    if client is None:
        return _fallback_overall(language=language)

    items = []
    for a in articles[:10]:
        t = (getattr(a, "title", "") or "").strip()
        s = (getattr(a, "summary", "") or "").strip()
        s = re.sub(r"\s+", " ", s).strip()

        if not s:
            continue

        if _is_en(language):
            items.append(f"- Title: {t}\n  Summary: {s}")
        else:
            items.append(f"- 제목: {t}\n  요약: {s}")

    if not items:
        if _is_en(language):
            return "Yesterday’s collected items did not contain usable text summaries, so there is nothing to consolidate."
        return "어제는 수집된 기사 중 텍스트 요약이 가능한 항목이 없어, 주요 이슈를 요약할 내용이 없습니다."

    target_sentences = _auto_sentence_target(len(items))

    if _is_en(language):
        max_chars = 650
        min_chars = 220
        prompt = f"""
You are an executive assistant writing a daily briefing for executives in the contact lens / optical industry.
Write a "Yesterday AI Briefing" using ONLY the input [Titles/Summaries] below.

ABSOLUTE RULES (MOST IMPORTANT):
- Use ONLY the facts stated in the input. Do NOT add any new facts, numbers, entities, brands, causes, or outcomes.
- No exaggeration, no speculation, no forecasting, no interpretation.
- Keep proper nouns as-is (Korean names/brands are allowed as proper nouns).
- Output MUST be in English.

OUTPUT FORMAT (IMPORTANT):
- Exactly {target_sentences} sentences.
- Sentence 1: One-sentence overall wrap-up (yesterday’s main flow within the input).
- Sentences 2–{target_sentences}: Summarize by distinct issues (group similar items into ONE sentence per issue).
- Do NOT list one sentence per article.
- Aim for at least {min_chars} characters but not more than {max_chars} characters.

[Titles/Summaries]
{chr(10).join(items)}
""".strip()
    else:
        max_chars = 420
        prompt = f"""
너는 콘택트렌즈/안경 업계 데일리 뉴스레터를 임원에게 보고하는 비서다.
아래 [기사 제목/요약]만을 근거로 '어제 기사 AI 브리핑'을 작성하라.

🚫 절대 규칙 (가장 중요):
- 아래 입력에 없는 사실/숫자/주체/브랜드/원인/결과를 절대 추가하지 말 것
- 과장/추측/전망/평가 금지
- 유사한 기사/동일 사건은 하나의 이슈로 묶어서 1문장으로만 작성

✅ 출력 형식(중요):
- 총 {target_sentences}문장 (문장 수 정확히 지킬 것)
- 1문장째: 전체 총평(어제 핵심 흐름/경향을 1문장으로)
- 2~{target_sentences}문장째: 서로 다른 '이슈' 단위로 요약
- 전체 {max_chars}자 이내

[기사 제목/요약]
{chr(10).join(items)}
""".strip()

    try:
        text = _call_openai(client, prompt, temperature=0.2)
        text = _enforce_sentence_and_length(text, max_sentences=3, max_chars=max_chars)

        # 영어 모드에서 너무 짧게 나왔으면 같은 의미로 보강 (팩트 추가 없이)
        if _is_en(language) and len(_norm_text(text)) < 220:
            text = _ensure_min_chars_english(text, title="Yesterday AI Briefing", min_chars=220, max_chars=max_chars, client=client)

        return text
    except Exception:
        return _fallback_overall(language=language)
