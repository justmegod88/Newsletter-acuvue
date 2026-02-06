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

    for p in meaningless_patterns:
        if p in s:
            return True
    return False


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
    # 텍스트가 거의 없고 이미지가 많으면 광고/배너로 판정
    body = _norm_text(body_text)
    if len(body) < 40 and img_count >= 1:
        return True
    return False


def _enforce_2to3_sentences(text: str, max_sentences: int = 3, max_chars: int = 105) -> str:
    t = _norm_text(text)
    if not t:
        return ""

    # 문장 분리 (영/한 혼용 대응)
    sents = re.split(r"(?<=[.!?。])\s+|(?<=[가-힣])\.\s+|(?<=[가-힣])\s+", t)
    sents = [s.strip() for s in sents if s.strip()]

    # 문장이 너무 없으면 그대로
    if not sents:
        sents = [t]

    sents = sents[:max_sentences]
    out = " ".join(sents).strip()

    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "…"
    return out


def _auto_sentence_target(n_items: int) -> int:
    # 기존 정책 유지: 1~3문장
    if n_items <= 3:
        return 2
    if n_items <= 6:
        return 3
    return 3


# =========================
# Prompts (KO/EN)
# =========================
def _is_en(language: str) -> bool:
    return (language or "ko").lower().startswith("en")


def _prompt_title_only(title: str, language: str) -> str:
    if _is_en(language):
        return f"""
You are writing a factual daily newsletter summary for executives in the contact lens / optical industry.

Rules (MOST IMPORTANT):
- Use ONLY what is explicitly stated in the title.
- Do NOT add any facts, numbers, entities, brands, causes, or outcomes that are not present.
- No exaggeration, no speculation, no forecasting.
- Only use the word "launch" if the title clearly states it; otherwise do not use it.
- Output 2–3 short sentences, within 105 characters if possible.

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
- 2~3문장, 105자 이내

[제목]
{title}

[출력]
""".strip()


def _prompt_compress_long_summary(title: str, summary: str, language: str) -> str:
    if _is_en(language):
        return f"""
You are writing a factual daily newsletter summary for executives in the contact lens / optical industry.

Rules (MOST IMPORTANT):
- Use ONLY what is explicitly stated in the input summary.
- Do NOT add any facts, numbers, entities, brands, causes, or outcomes that are not present.
- No exaggeration, no speculation, no forecasting.
- Only use the word "launch" if the input clearly states it; otherwise do not use it.
- Output 2–3 short sentences, within 105 characters if possible.

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
- 2~3문장, 105자 이내
- 가능한 한 팩트 중심으로

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

Rules (MOST IMPORTANT):
- Use ONLY what is explicitly stated in the article body.
- Do NOT add any facts, numbers, entities, brands, causes, or outcomes that are not present.
- No exaggeration, no speculation, no forecasting.
- Only use the word "launch" if the body explicitly states it; otherwise do not use it.
- Output 2–3 short sentences, within 105 characters if possible.
- Focus on hard facts (who/what/which action/what happened).

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
- 기사에 없는 단어 절대 사용 금지
- 2~3문장, 105자 이내
- 가능한 한 팩트 중심으로

[제목]
{title}

[기사 본문]
{body_text}

[출력]
""".strip()


def _call_openai_2to3_sentences(client, prompt: str, max_chars: int = 105) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = (r.choices[0].message.content or "").strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _fallback_overall(articles: List, language: str = "ko") -> str:
    if _is_en(language):
        return "A brief could not be generated due to missing AI access; please refer to the article list below."
    return "AI 요약을 생성할 수 없어 기사 목록만 공유드립니다."


# =========================
# ✅ A. 기사별 summary 정제/생성
# =========================
def refine_article_summaries(articles: List, language: str = "ko") -> None:
    """
    ✅ 각 기사 summary 정책(확정본) - 로직 유지
    (변경점: language='en'일 때 프롬프트만 영어로)
    """
    client = _get_client()

    LONG_SUMMARY_THRESHOLD = 150
    MAX_SUMMARY_CHARS = 105

    for a in articles:
        title = _norm_text(getattr(a, "title", "") or "")
        summary_raw = getattr(a, "summary", "") or ""
        summary = _norm_text(summary_raw)
        link = (getattr(a, "link", "") or "").strip()

        # ✅ 네이버(OpenAPI 포함) 판별 플래그
        is_naver = bool(getattr(a, "is_naver", False))

        # 링크가 이미지 파일이면: 광고/배너로 보고 summary는 빈값
        if _is_image_file_url(link):
            try:
                a.summary = ""
            except Exception:
                pass
            continue

        # 3) summary 없음/무의미 -> 본문 확인
        if not summary or _is_meaningless_summary(summary):
            html = _fetch_html(link)
            if not html:
                try:
                    a.summary = ""
                except Exception:
                    pass
                continue

            body_text, img_count = _extract_text_and_imgcount(html)

            # 3-1) 이미지만 광고 -> 빈값
            if _is_image_only_ad_page(body_text, img_count):
                try:
                    a.summary = ""
                except Exception:
                    pass
                continue

            # 3-2) 본문 텍스트 -> AI 요약(가능하면)
            if client is not None:
                try:
                    prompt = _prompt_summarize_from_body(title, body_text, language)
                    summary = _call_openai_2to3_sentences(client, prompt, max_chars=MAX_SUMMARY_CHARS)
                except Exception:
                    summary = _norm_text(body_text)[:MAX_SUMMARY_CHARS].rstrip()
            else:
                summary = _norm_text(body_text)[:MAX_SUMMARY_CHARS].rstrip()

            summary = _enforce_2to3_sentences(summary, max_sentences=3, max_chars=MAX_SUMMARY_CHARS)

            try:
                a.summary = summary
            except Exception:
                pass
            continue

        # 2) summary == title -> 제목 정보만으로 2~3문장(추측 절대 금지)
        if _is_summary_same_as_title(title, summary):
            if client is not None:
                try:
                    prompt = _prompt_title_only(title, language)
                    summary = _call_openai_2to3_sentences(client, prompt, max_chars=MAX_SUMMARY_CHARS)
                except Exception:
                    summary = title
            else:
                summary = title

            summary = _enforce_2to3_sentences(summary, max_sentences=3, max_chars=MAX_SUMMARY_CHARS)

            try:
                a.summary = summary
            except Exception:
                pass
            continue

        # 1) summary가 길면 -> 압축 요약
        if len(summary) >= LONG_SUMMARY_THRESHOLD:
            if client is not None:
                try:
                    prompt = _prompt_compress_long_summary(title, summary, language)
                    summary = _call_openai_2to3_sentences(client, prompt, max_chars=MAX_SUMMARY_CHARS)
                except Exception:
                    summary = summary[:MAX_SUMMARY_CHARS].rstrip() + "…"
            else:
                summary = summary[:MAX_SUMMARY_CHARS].rstrip() + "…"

        # ✅ 네이버는 길이와 상관없이 한번 더 AI 정리(기존 정책 유지)
        if is_naver and client is not None:
            try:
                prompt = _prompt_compress_long_summary(title, summary, language)
                summary = _call_openai_2to3_sentences(client, prompt, max_chars=MAX_SUMMARY_CHARS)
            except Exception:
                pass

        summary = _enforce_2to3_sentences(summary, max_sentences=3, max_chars=MAX_SUMMARY_CHARS)

        try:
            a.summary = summary
        except Exception:
            pass


# =========================
# ✅ B. 상단 전체 요약
# =========================
def summarize_overall(articles: List, language: str = "ko") -> str:
    """
    ✅ 임원용 "어제 기사 AI 브리핑" (이슈 묶기형)
    - 정책 유지, 출력 언어만 선택
    """
    if not articles:
        if _is_en(language):
            return "There were no relevant articles collected for yesterday, so there is nothing additional to brief."
        return "어제 기준으로 수집된 관련 기사가 없어 별도 공유 사항은 없습니다."

    client = _get_client()
    if client is None:
        return _fallback_overall(articles, language=language)

    items = []
    for a in articles[:10]:
        t = (getattr(a, "title", "") or "").strip()
        s = (getattr(a, "summary", "") or "").strip()
        s = re.sub(r"\s+", " ", s).strip()

        if len(s) > 150:
            s = s[:150].rstrip() + "…"

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
        prompt = f"""
You are an executive assistant writing a daily briefing for executives in the contact lens / optical industry.
Write a "Yesterday AI Briefing" using ONLY the input [Titles/Summaries] below.

ABSOLUTE RULES (MOST IMPORTANT):
- Do NOT add any facts, numbers, entities, brands, causes, or outcomes that are not in the input.
- No exaggeration, no speculation, no forecasting, no interpretation.
  * Forbidden examples: "it suggests", "it indicates", "it is likely", "expected to", "may lead to"
- Only use the word "launch" if the input explicitly states it; otherwise do not use it.
- Trend wording is allowed ONLY within what is observable from the input.
  * Allowed: "coverage continued", "this topic appeared repeatedly across multiple items"
  * Not allowed: "will expand", "strategically important", "will lead to growth" (future/interpretation)

OUTPUT FORMAT (IMPORTANT):
- Exactly {target_sentences} sentences.
- Sentence 1: One-sentence overall wrap-up (yesterday’s main flow within the input).
- Sentences 2–{target_sentences}: Summarize by distinct "issues" (group similar items into ONE sentence per issue).
- Do NOT list one sentence per article.
- Total within 420 characters. Keep sentences short and definitive.

[Titles/Summaries]
{chr(10).join(items)}
""".strip()
    else:
        prompt = f"""
너는 콘택트렌즈/안경 업계 데일리 뉴스레터를 임원에게 보고하는 비서다.
아래 [기사 제목/요약]만을 근거로 '어제 기사 AI 브리핑'을 작성하라.

🚫 절대 규칙 (가장 중요):
- 아래 입력에 없는 사실/숫자/주체/브랜드/원인/결과를 절대 추가하지 말 것
- 과장/추측/전망/평가 금지
  * 금지 예: "~로 보인다", "~할 것으로 예상", "~가능성이 높다", "~시사한다", "~의미가 크다"
- 기사에 '출시'라는 단어를 명확히 언급한 경우만 사용, 아니면 사용 절대 금지
- 트렌드/경향 언급은 가능하나, 반드시 입력에서 관찰되는 범위로만 표현할 것
  * 허용 예: "관련 보도가 이어졌다", "○○ 주제가 다수 기사에서 반복됐다"
  * 금지 예: "시장 확대/축소로 이어질 것", "전략적으로 중요해질 것" (미래/해석)

✅ 출력 형식(중요):
- 총 {target_sentences}문장 (문장 수 정확히 지킬 것)
- 1문장째: 전체 총평(어제 핵심 흐름/경향을 1문장으로)
- 2~{target_sentences}문장째: 서로 다른 '이슈' 단위로 요약
- 유사한 기사/동일 사건은 하나의 이슈로 묶어서 1문장으로만 작성
- 문장마다 특정 기사 1개를 그대로 옮겨 적는 '나열형' 금지
- 전체 420자 이내, 문장은 짧고 단정하게

[기사 제목/요약]
{chr(10).join(items)}
""".strip()

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = (r.choices[0].message.content or "").strip()
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return _fallback_overall(articles, language=language)

        if len(text) > 420:
            text = text[:420].rstrip() + "…"

        text = _enforce_2to3_sentences(text, max_sentences=3, max_chars=420)
        return text
    except Exception:
        return _fallback_overall(articles, language=language)
