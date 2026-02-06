import datetime as dt
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import re
import difflib

from jinja2 import Environment, FileSystemLoader

from scrapers import (
    load_config,
    fetch_all_articles,
    filter_yesterday_articles,
    filter_out_finance_articles,
    filter_out_yakup_articles,
    deduplicate_articles,        # (scrapers.py의 URL+제목 dedup: 1차)
    should_exclude_article,      # ✅ 최종 안전 필터용
)
from categorizer import categorize_articles
from summarizer import refine_article_summaries, summarize_overall
from mailer import send_email_html


# =========================
# ✅ (A) URL/제목 정규화
# =========================
def _normalize_url(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url)
    path = (p.path or "").rstrip("/")
    scheme = p.scheme or "https"
    return f"{scheme}://{p.netloc.lower()}{path}"


def _normalize_title(title: str) -> str:
    t = (title or "").lower().strip()
    t = re.sub(r"\[[^\]]+\]", " ", t)      # [단독]
    t = re.sub(r"\([^)]*\)", " ", t)       # (종합)
    t = re.sub(r"[^\w가-힣]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _similarity(a: str, b: str) -> float:
    a = _normalize_text(a)
    b = _normalize_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _title_bucket_keys(title: str):
    """
    중복 후보 비교 대상을 좁히기 위한 버킷 키.
    """
    t = _normalize_title(title)
    tokens = t.split()
    keys = []
    if tokens:
        keys.append(tokens[0])
        if len(tokens) >= 2:
            keys.append(" ".join(tokens[:2]))
    return keys or [""]


# =========================
# ✅ (B) 리스트용 dedupe & group (기존 유지)
# =========================
def dedupe_and_group_articles(articles, threshold=0.75):
    if not articles:
        return []

    # 1) URL 정규화 기준으로 먼저 그룹
    url_map = {}
    for a in articles:
        u = _normalize_url(getattr(a, "link", "") or "")
        url_map.setdefault(u, []).append(a)

    unique = []
    for u, group in url_map.items():
        # 동일 URL이면 1개만 남김(가장 summary가 긴 걸 우선)
        group_sorted = sorted(group, key=lambda x: len(getattr(x, "summary", "") or ""), reverse=True)
        unique.append(group_sorted[0])

    # 2) 제목 유사도 기반으로 추가 중복 그룹핑
    buckets = {}
    for a in unique:
        keys = _title_bucket_keys(getattr(a, "title", "") or "")
        for k in keys:
            buckets.setdefault(k, []).append(a)

    seen = set()
    final = []
    for k, group in buckets.items():
        for a in group:
            if id(a) in seen:
                continue
            # a를 대표로 삼고, 유사한 것들을 묶어서 하나만 남김
            rep = a
            seen.add(id(a))
            for b in group:
                if id(b) in seen:
                    continue
                if _similarity(getattr(rep, "title", ""), getattr(b, "title", "")) >= threshold:
                    seen.add(id(b))
            final.append(rep)

    # 버킷 중복으로 final에 중복이 생길 수 있어 URL 기준 한번 더 정리
    out = []
    out_seen = set()
    for a in final:
        u = _normalize_url(getattr(a, "link", "") or "")
        if u in out_seen:
            continue
        out_seen.add(u)
        out.append(a)

    return out


# =========================
# ✅ (C) 브리핑용 pick (기존 유지)
# =========================
def select_articles_for_brief(
    acuvue_articles,
    company_articles,
    product_articles,
    trend_articles,
    eye_health_articles,
    max_items=10,
):
    combined = []
    combined += list(acuvue_articles or [])
    combined += list(company_articles or [])
    combined += list(product_articles or [])
    combined += list(trend_articles or [])
    combined += list(eye_health_articles or [])

    # 요약이 비어있거나 너무 짧은 건 제외
    filtered = []
    for a in combined:
        s = (getattr(a, "summary", "") or "").strip()
        if len(s) < 10:
            continue
        filtered.append(a)

    # 제목+URL 기준으로 안정적으로 중복 제거
    seen = set()
    unique = []
    for a in filtered:
        key = (_normalize_url(getattr(a, "link", "") or ""), _normalize_title(getattr(a, "title", "") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)

    return unique[:max_items]


def build_yesterday_ai_brief(
    acuvue_articles,
    company_articles,
    product_articles,
    trend_articles,
    eye_health_articles,
    output_language="ko",
):
    picked = select_articles_for_brief(
        acuvue_articles,
        company_articles,
        product_articles,
        trend_articles,
        eye_health_articles,
        max_items=10,
    )

    if not picked:
        if (output_language or "ko").lower().startswith("en"):
            return "There were no relevant articles collected for yesterday, so there is nothing additional to brief."
        return "어제는 수집된 기사가 없어 주요 이슈를 요약할 내용이 없습니다."

    return summarize_overall(picked, language=output_language)


def main():
    cfg = load_config()
    tz = ZoneInfo(cfg.get("timezone", "Asia/Seoul"))
    output_language = (cfg.get("output_language", "ko") or "ko").strip()

    # 1) 수집
    articles = fetch_all_articles(cfg)

    # 2) 약업신문 제외 + 투자/재무 제외
    articles = filter_out_yakup_articles(articles)
    articles = filter_out_finance_articles(articles)

    # 3) 날짜 필터: 어제 기사만
    articles = filter_yesterday_articles(articles, cfg)

    # 4) 1차 중복 제거(빠른 제거: URL+제목)
    articles = deduplicate_articles(articles)

    # 5) 기사별 요약(summary 정제/생성)  ✅ 언어만 영어로 옵션
    refine_article_summaries(articles, language=output_language)

    # 6) 최종 안전 필터
    articles = [a for a in articles if not should_exclude_article(a.title, a.summary)]

    # ✅ 7) 기사 리스트용 중복 묶기(기존 유지: 0.75)
    articles = dedupe_and_group_articles(articles, threshold=0.75)

    # 8) 분류
    categorized = categorize_articles(articles)

    # 9) 카테고리 간 중복 제거
    acuvue_list, company_list, product_list, trend_list, eye_health_list = remove_cross_category_duplicates(
        categorized.acuvue,
        categorized.company,
        categorized.product,
        categorized.trend,
        categorized.eye_health,
    )

    # ✅ 10) 상단 브리핑(브리핑 전용)
    summary = build_yesterday_ai_brief(
        acuvue_list,
        company_list,
        product_list,
        trend_list,
        eye_health_list,
        output_language=output_language,
    )

    # 11) 템플릿 렌더링
    env = Environment(loader=FileSystemLoader("."), autoescape=True)
    template = env.get_template("template_newsletter.html")

    html = template.render(
        today_date=dt.datetime.now(tz).strftime("%Y-%m-%d"),
        yesterday_summary=summary,
        acuvue_articles=acuvue_list,
        company_articles=company_list,
        product_articles=product_list,
        trend_articles=trend_list,
        eye_health_articles=eye_health_list,

        # ✅ 템플릿에서 영어 라벨을 쓰고 싶을 때를 위해 옵션 제공(템플릿 수정은 아래 참고)
        output_language=output_language,
    )

    # 12) 메일 제목
    email = cfg["email"]
    yesterday_str = (dt.datetime.now(tz).date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    subject_prefix = email.get("subject_prefix", "[Daily News]")

    if output_language.lower().startswith("en"):
        subject = f"{subject_prefix} Yesterday Briefing - {yesterday_str}"
    else:
        subject = f"{subject_prefix} 어제 기사 브리핑 - {yesterday_str}"

    # 13) 발송
    send_email_html(
        subject=subject,
        html_body=html,
        from_addr=email["from"],
        to_addrs=email["to"],
    )


# =========================
# ✅ (D) 카테고리 간 중복 제거 (기존 유지)
# =========================
def remove_cross_category_duplicates(acuvue, company, product, trend, eye_health):
    def make_key(a):
        return (_normalize_url(getattr(a, "link", "") or ""), _normalize_title(getattr(a, "title", "") or ""))

    seen = set()
    out = []
    for lst in [acuvue, company, product, trend, eye_health]:
        new_list = []
        for a in lst:
            k = make_key(a)
            if k in seen:
                continue
            seen.add(k)
            new_list.append(a)
        out.append(new_list)

    return tuple(out)


if __name__ == "__main__":
    main()
