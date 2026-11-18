from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


# -----------------------------
# Swissdox config (keywords + query builder)
# -----------------------------
DEFAULT_API_BASE_URL = "https://swissdox.linguistik.uzh.ch/api"
DEFAULT_COLUMNS = [
    "id", "pubtime", "medium_code", "medium_name", "rubric", "regional",
    "doctype", "doctype_description", "language", "char_count", "dateline",
    "head", "subhead", "content_id", "content",
]

# -----------------------------
# Mots-clés de la requête Swissdox : tout est défini en toutes lettres dans
# src/keywords_list.py (départements + unités fédérales, agences
# indépendantes, conseillers fédéraux ; déclinaisons allemandes et variantes
# d'accents / de casse comprises).
#
# Ce module ne fait que rassembler les formes et ajouter l'autre apostrophe
# (' <-> ’) aux termes qui en contiennent une.
#
# NB : le step 2 ne lit PAS ces listes ; il fait son propre matching via
# src/target_taxonomy.py.
# -----------------------------
try:  # src/ sur le sys.path
    from keywords_list import (  # noqa: E402
        COUNCILLORS as _COUNCILLORS_LIST,
        FEDERAL_UNITS as _FEDERAL_UNITS_LIST,
        INDEPENDENT_AGENCIES as _INDEPENDENT_AGENCIES_LIST,
    )
except ImportError:  # import via le package "src"
    from src.keywords_list import (  # noqa: E402
        COUNCILLORS as _COUNCILLORS_LIST,
        FEDERAL_UNITS as _FEDERAL_UNITS_LIST,
        INDEPENDENT_AGENCIES as _INDEPENDENT_AGENCIES_LIST,
    )

_APOSTROPHES = ("'", "’")


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def apostrophe_variants(term: str) -> List[str]:
    if not any(a in term for a in _APOSTROPHES):
        return [term]
    base = term.replace("’", "'")
    return [base, base.replace("'", "’")]


def _query_terms(terms: List[str]) -> List[str]:
    """Normalise les espaces, ajoute les variantes d'apostrophe, dédoublonne (ordre stable)."""
    return list(dict.fromkeys(
        v for t in terms if _norm_spaces(t) for v in apostrophe_variants(_norm_spaces(t))
    ))


def _entity_surfaces(entities: List[Dict[str, Any]]) -> List[str]:
    """Toutes les formes (noms FR/DE déclinés, abréviations, variantes de casse)
    d'une liste d'entités au format de keywords_list.py (FEDERAL_UNITS,
    INDEPENDENT_AGENCIES)."""
    terms: List[str] = []
    for u in entities:
        for variants in u["fr_names"]:          # [forme de référence, variantes casse/accents...]
            terms.extend(variants)
        for declined in u["de_names"]:          # [forme de référence, déclinaisons...]
            terms.extend(declined)
        case_variants = u.get("abbrev_case_variants", {})
        for lang in ("fr", "de"):
            for ab in [u[f"{lang}_abbrev"]] + u[f"{lang}_alt_abbrevs"]:
                if ab:
                    terms.append(ab)
                    terms.extend(case_variants.get(ab, []))
    return _query_terms(terms)


FEDERAL_UNIT_TERMS: List[str] = _entity_surfaces(_FEDERAL_UNITS_LIST)
INDEPENDENT_AGENCY_TERMS: List[str] = _entity_surfaces(_INDEPENDENT_AGENCIES_LIST)
COUNCILLOR_TERMS: List[str] = _query_terms([f for forms in _COUNCILLORS_LIST for f in forms])


def build_query_payload(
    *,
    start_date: str,
    end_date: str,
    languages: List[str],
    sources: List[str],
    max_results: int,
    columns: List[str],
    query_name: str,
    comment: str,
    expiration_date: str,
    version: str = "1.2",
) -> Dict[str, Any]:
    query_block = {
        "sources": sources,
        "dates": [{"from": start_date, "to": end_date}],
        "languages": languages,
        "content": {
            "OR": [
                {"OR": FEDERAL_UNIT_TERMS},
                {"OR": INDEPENDENT_AGENCY_TERMS},
                {"OR": COUNCILLOR_TERMS},
            ]
        },
    }
    yaml_payload = {
        "query": query_block,
        "result": {"format": "TSV", "maxResults": max_results, "columns": columns},
        "version": version,
    }
    return {
        "yaml_payload": yaml_payload,
        "meta": {"name": query_name, "comment": comment, "expirationDate": expiration_date},
    }


# -----------------------------
# Swissdox API client
# -----------------------------
@dataclass
class SwissdoxClient:
    api_key: str
    api_secret: str
    base_url: str = DEFAULT_API_BASE_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.query_url = f"{self.base_url}/query"
        self.status_url = f"{self.base_url}/status"
        self.headers = {"X-API-Key": self.api_key, "X-API-Secret": self.api_secret}
        self.session = requests.Session()

    def submit_query(
        self,
        yaml_payload: Dict[str, Any],
        *,
        name: str,
        comment: str,
        expiration_date: str,
        test: bool = False,
    ) -> str:
        yaml_query = yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True)
        data = {
            "query": yaml_query,
            "name": name,
            "comment": comment,
            "expirationDate": expiration_date,
        }
        if test:
            data["test"] = "1"

        r = self.session.post(self.query_url, headers=self.headers, data=data, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Swissdox /query failed ({r.status_code}): {r.text[:2000]}")

        resp = r.json()
        if resp.get("result") != "ok":
            raise RuntimeError(f"Swissdox non-ok response: {resp}")

        qid = resp.get("queryId") or resp.get("id")
        if not qid:
            raise RuntimeError(f"Swissdox missing queryId: {resp}")
        return str(qid)

    # Erreurs serveur passagères (500, 502, 503, 504, coupure réseau) : on
    # réessaie au lieu de faire échouer un job de plusieurs heures. Une erreur
    # 4xx (clé API refusée, requête invalide) reste immédiatement fatale.
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    def _get_with_retry(self, url: str, *, what: str, max_tries: int = 10, base_sleep_s: int = 30):
        for attempt in range(1, max_tries + 1):
            try:
                r = self.session.get(url, headers=self.headers, timeout=self.timeout)
                if r.status_code not in self._RETRY_STATUS:
                    r.raise_for_status()
                    return r
                err = f"HTTP {r.status_code}"
            except (requests.ConnectionError, requests.Timeout) as e:
                err = type(e).__name__
            if attempt == max_tries:
                raise RuntimeError(f"Swissdox {what}: échec après {max_tries} essais ({err})")
            wait = min(base_sleep_s * 2 ** (attempt - 1), 30 * 60)   # 30 s, 1 min, 2 min, ... max 30 min
            print(f"[Swissdox] {what}: {err}, nouvel essai {attempt + 1}/{max_tries} dans {wait}s", flush=True)
            time.sleep(wait)

    def wait_for_download_url(
        self,
        query_id: str,
        *,
        max_wait_s: int = 12 * 60 * 60,
        poll_every_s: int = 30,
    ) -> str:
        start = time.time()
        deadline = start + max_wait_s
        last_log = float("-inf")   # 1re ligne de suivi tout de suite
        missing_since = None
        while time.time() < deadline:
            rs = self._get_with_retry(self.status_url, what="/status")
            status_list = rs.json()
            now = time.time()

            job = next((j for j in status_list if str(j.get("id")) == str(query_id)), None)
            if not job:
                # La requête n'apparaît pas (encore / plus) dans /status. Juste après
                # une soumission c'est normal quelques minutes ; au-delà de 30 min,
                # elle a disparu (expirée, supprimée ou échouée côté Swissdox) :
                # on arrête au lieu d'attendre en silence jusqu'à max_wait_s.
                missing_since = missing_since or now
                if now - missing_since > 30 * 60:
                    ids = [str(j.get("id")) for j in status_list][-10:]
                    raise RuntimeError(
                        f"Swissdox: queryId {query_id} absent de /status depuis 30 min "
                        f"(10 derniers ids listés: {ids}). Relancer sans --query-id pour soumettre à nouveau."
                    )
            else:
                missing_since = None
                if job.get("error"):
                    raise RuntimeError(f"Swissdox job error: {job['error']}")
                url = job.get("downloadUrl")
                if url:
                    print(f"[Swissdox] résultat prêt après {(now - start) / 3600:.1f} h d'attente", flush=True)
                    return self._normalize_download_url(url)

            if now - last_log >= 10 * 60:   # une ligne de suivi toutes les 10 min
                state = ({k: v for k, v in job.items() if k not in ("query", "downloadUrl")}
                         if job else "requête absente de /status")
                print(f"[Swissdox] attente {(now - start) / 3600:.1f} h — état: {state}", flush=True)
                last_log = now

            time.sleep(poll_every_s)

        raise TimeoutError(f"Swissdox: no downloadUrl after {max_wait_s}s (query_id={query_id})")

    def download_tsv_xz(self, download_url: str) -> pd.DataFrame:
        r = self._get_with_retry(download_url, what="téléchargement", max_tries=6)
        return pd.read_csv(BytesIO(r.content), sep="\t", compression="xz")

    def _normalize_download_url(self, download_url: str) -> str:
        if download_url.startswith("http"):
            return download_url
        if download_url.startswith("/"):
            return f"{self.base_url}{download_url}"
        return f"{self.base_url}/download/{download_url}"


# -----------------------------
# Text cleaning
# -----------------------------
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(' "“”„\'')


def clean_xml_keep_paragraphs(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = html.unescape(s)

    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n\n", s)
    s = re.sub(r"(?i)<p[^>]*>", "", s)
    s = re.sub(r"(?i)</div\s*>", "\n\n", s)
    s = re.sub(r"(?i)<div[^>]*>", "", s)

    s = re.sub(r"<[^>]+>", " ", s)

    lines = s.splitlines()
    cleaned_lines = []
    for line in lines:
        line = _WS_RE.sub(" ", line).strip()
        cleaned_lines.append(line)

    s = "\n".join(cleaned_lines)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = _MULTI_NL_RE.sub("\n\n", s).strip()

    return s


def extract_first_paragraph(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.strip()
    if not s:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    if paragraphs:
        return paragraphs[0]

    return clean_text(s)


def clean_articles_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "pubtime" in df.columns:
        df["pubtime"] = pd.to_datetime(df["pubtime"].astype(str), errors="coerce", utc=True).dt.date

    for c in ["medium_name", "rubric", "dateline", "head", "subhead"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_text)

    if "content" in df.columns:
        df["content"] = df["content"].apply(clean_xml_keep_paragraphs)

    return df


# -----------------------------
# Article-level output:
# one row = one article with title + lead + full text
# -----------------------------
def build_article_leads(df_articles: pd.DataFrame, *, content_col: str = "content") -> pd.DataFrame:
    if content_col not in df_articles.columns:
        raise ValueError(f"Missing '{content_col}' column")

    if "content_id" in df_articles.columns:
        article_id = df_articles["content_id"].astype(str)
    elif "id" in df_articles.columns:
        article_id = df_articles["id"].astype(str)
    else:
        article_id = df_articles.index.astype(str)

    title = df_articles["head"].fillna("").astype(str) if "head" in df_articles.columns else ""
    lead = df_articles[content_col].fillna("").astype(str).apply(extract_first_paragraph)
    text = df_articles[content_col].fillna("").astype(str)

    out = pd.DataFrame({
        "article_id": article_id,
        "title": title,
        "lead": lead,
        "text": text,
    })

    meta_cols = [
        "pubtime",
        "medium_code",
        "medium_name",
        "rubric",
        "regional",
        "doctype",
        "doctype_description",
        "language",
        "char_count",
        "dateline",
    ]

    for col in meta_cols:
        if col in df_articles.columns:
            out[col] = df_articles[col]

    return out


# -----------------------------
# Top-level pipeline
# -----------------------------
ARTICLES_FILE = "swissdox_articles_lead.parquet"


def run_pipeline(
    *,
    start_date: str,
    end_date: str,
    languages: List[str],
    sources: List[str],
    max_results: int,
    expiration_date: str,
    query_name: str,
    comment: str,
    out_dir: Path,
    test: bool = False,
    max_wait_s: int = 12 * 60 * 60,
    query_id: str | None = None,
) -> Dict[str, Path]:
    load_dotenv()
    api_key = os.getenv("SWISSDOX_API_KEY")
    api_secret = os.getenv("SWISSDOX_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing SWISSDOX_API_KEY / SWISSDOX_API_SECRET in .env")

    payload = build_query_payload(
        start_date=start_date,
        end_date=end_date,
        languages=languages,
        sources=sources,
        max_results=max_results,
        columns=DEFAULT_COLUMNS,
        query_name=query_name,
        comment=comment,
        expiration_date=expiration_date,
    )

    client = SwissdoxClient(api_key=api_key, api_secret=api_secret)
    if query_id:
        # Reprise : la requête a déjà été soumise (ex. job précédent interrompu),
        # on ne la renvoie pas, on attend / télécharge son résultat.
        qid = str(query_id)
        print("[Swissdox] reprise de la requête existante (pas de nouvelle soumission)")
    else:
        print(f"[Swissdox] requête : {len(FEDERAL_UNIT_TERMS)} termes départements/unités, "
              f"{len(INDEPENDENT_AGENCY_TERMS)} agences, {len(COUNCILLOR_TERMS)} conseillers")
        qid = client.submit_query(
            payload["yaml_payload"],
            name=payload["meta"]["name"],
            comment=payload["meta"]["comment"],
            expiration_date=payload["meta"]["expirationDate"],
            test=test,
        )
    print(f"[Swissdox] queryId={qid}")

    url = client.wait_for_download_url(qid, max_wait_s=max_wait_s)
    print(f"[Swissdox] downloadUrl={url}")

    df_raw = client.download_tsv_xz(url)
    print(f"[Swissdox] raw shape={df_raw.shape}")

    if len(df_raw) >= max_results:
        print(
            f"[Swissdox] WARNING: result count ({len(df_raw)}) hit the max-results cap "
            f"({max_results}). Some articles may have been truncated."
        )

    df_articles = clean_articles_df(df_raw)
    print(f"[Swissdox] cleaned articles shape={df_articles.shape}")

    df_leads = build_article_leads(df_articles)
    print(f"[Swissdox] article-level output shape={df_leads.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)

    p_articles_parquet = out_dir / ARTICLES_FILE
    p_articles_csv = out_dir / "swissdox_articles_lead.csv"

    df_leads.to_parquet(p_articles_parquet, index=False)
    df_leads.to_csv(p_articles_csv, index=False)

    return {
        "articles_lead_parquet": p_articles_parquet,
        "articles_lead_csv": p_articles_csv,
    }
