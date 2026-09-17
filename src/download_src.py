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

DE_TERMS = [
    "Bürokratie", "Berner Verwaltung", "Papierkrieg", "Verwaltung", "Bundesverwaltung",
    "Beamtenapparat", "Amtsschimmel", "Regulierungsdichte", "Behörden", "Bürokraten",
    "Beamte", "Staatsangestellte",
]
DE_LEVEL = ["Bund", "Bundes", "Kanton", "kantonal", "Schweiz"]

FR_TERMS = [
    "Bureaucratie", "Administration publique", "Administration fédérale", "Appareil administratif",
    "Appareil étatique", "Appareil de l'État", "Autorités administratives", "Services de l'État",
    "Services publics", "Fonction publique", "Pouvoir administratif", "Autorités cantonales",
    "Administration centrale", "Départements fédéraux", "Offices fédéraux", "Organes de l'État",
    "Technocratie", "Bureaucrates", "Fonctionnaires", "Employés de l'État",
]
FR_LEVEL = ["fédéral", "federal", "federale", "cantonal", "cantonale", "Suisse"]

DEPARTMENTS = [
    "VBS", "DDPS", "Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport",
    "Département fédéral de la défense, de la protection de la population et des sports",
    "EDA", "DFAE", "Eidgenössische Departement für auswärtige Angelegenheiten",
    "Département fédéral des affaires étrangères",
    "UVEK", "DETEC", "Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation",
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication",
    "EJPD", "DFJP", "Eidgenössische Justiz- und Polizeidepartement",
    "Département fédéral de justice et police",
    "EDI", "DFI", "Eidgenössische Departement des Innern", "Département fédéral de l'intérieur",
    "EFD", "DFF", "Eidgenössische Finanzdepartement", "Département fédéral des finances",
    "WBF", "DEFR", "Eidgenössische Departement für Wirtschaft, Bildung und Forschung",
    "Département fédéral de l'économie, de la formation et de la recherche",
]

# Federal councillors 2000–present (from run2_prompts.py compositions)
COUNCILLORS = [
    "Joseph Deiss", "Ruth Dreifuss", "Ruth Metzler-Arnold", "Adolf Ogi",
    "Kaspar Villiger", "Pascal Couchepin", "Moritz Leuenberger", "Samuel Schmid",
    "Micheline Calmy-Rey", "Christoph Blocher", "Hans-Rudolf Merz", "Doris Leuthard",
    "Eveline Widmer-Schlumpf", "Ueli Maurer", "Didier Burkhalter", "Simonetta Sommaruga",
    "Johann Schneider-Ammann", "Alain Berset", "Guy Parmelin", "Ignazio Cassis",
    "Karin Keller-Sutter", "Viola Amherd", "Elisabeth Baume-Schneider", "Albert Rösti",
    "Beat Jans", "Martin Pfister",
]

# Administrative units of each department (FR + DE names and abbreviations)
# NOTE: 2-letter abbreviations (DV, DC, KD, DR, OA) and ambiguous short ones
# (BAR, BAG, BIT, BWL) are intentionally omitted — they are common standalone
# words in FR/DE press and would generate false positives.
ADMIN_UNITS = [
    # ── DFAE / EDA ──────────────────────────────────────────────────────────
    "SG-DFAE", "GS-EDA",
    "Direction du droit international public", "DDIP",
    "Direktion für Völkerrecht",
    "Direction consulaire",
    "Konsularische Direktion",
    "Direction du développement et de la coopération", "DDC",
    "Direktion für Entwicklung und Zusammenarbeit", "DEZA",
    "Direction des ressources",
    "Direktion für Ressourcen",
    # ── DFI / EDI ───────────────────────────────────────────────────────────
    "SG-DFI", "GS-EDI",
    "Bureau fédéral de l'égalité entre femmes et hommes", "BFEG",
    "Eidgenössisches Büro für die Gleichstellung von Frau und Mann", "EBG",
    "Office fédéral de la culture", "OFC",
    "Bundesamt für Kultur", "BAK",
    "Archives fédérales suisses", "AFS",
    "Schweizerisches Bundesarchiv",
    "Office fédéral de météorologie et de climatologie", "MétéoSuisse",
    "Bundesamt für Meteorologie und Klimatologie", "MeteoSchweiz",
    "Office fédéral de la santé publique", "OFSP",
    "Bundesamt für Gesundheit", "BAG",
    "Office fédéral de la sécurité alimentaire et des affaires vétérinaires", "OSAV",
    "Bundesamt für Lebensmittelsicherheit und Veterinärwesen", "BLV",
    "Office fédéral de la statistique", "OFS",
    "Bundesamt für Statistik", "BFS",
    "Office fédéral des assurances sociales", "OFAS",
    "Bundesamt für Sozialversicherungen", "BSV",
    # ── DFJP / EJPD ─────────────────────────────────────────────────────────
    "SG-DFJP", "GS-EJPD",
    "Secrétariat d'État aux migrations", "SEM",
    "Staatssekretariat für Migration",
    "Office fédéral de la justice", "OFJ",
    "Bundesamt für Justiz",
    "Office fédéral de la police", "fedpol",
    "Bundesamt für Polizei",
    "Service Surveillance de la correspondance par poste et télécommunication", "Service SCPT",
    "Dienst Überwachung Post- und Fernmeldeverkehr", "ÜPF",
    # ── DDPS / VBS ──────────────────────────────────────────────────────────
    "SG-DDPS", "GS-VBS",
    "Office fédéral de la protection de la population", "OFPP",
    "Bundesamt für Bevölkerungsschutz", "BABS",
    "Office fédéral de l'armement", "armasuisse",
    "Bundesamt für Rüstung",
    "Office fédéral de topographie", "swisstopo",
    "Bundesamt für Landestopografie",
    "Office fédéral du sport", "OFSPO",
    "Bundesamt für Sport", "BASPO",
    "Office fédéral de la cybersécurité", "OFCS",
    "Bundesamt für Cybersicherheit", "BACS",
    "Secrétariat d'État à la politique de sécurité", "SEPOS",
    "Staatssekretariat für Sicherheitspolitik",
    "Armée suisse", "Schweizer Armee",
    "Service de renseignement de la Confédération", "SRC",
    "Nachrichtendienst des Bundes", "NDB",
    "Office de l'auditeur en chef", "OAC",
    "Oberauditorat",
    # ── DFF / EFD ───────────────────────────────────────────────────────────
    "SG-DFF", "GS-EFD",
    "Secrétariat d'État aux questions financières internationales", "SFI",
    "Staatssekretariat für internationale Finanzfragen", "SIF",
    "Administration fédérale des finances", "AFF",
    "Eidgenössische Finanzverwaltung", "EFV",
    "Office fédéral du personnel", "OFPER",
    "Eidgenössisches Personalamt", "EPA",
    "Administration fédérale des contributions", "AFC",
    "Eidgenössische Steuerverwaltung", "ESTV",
    "Office fédéral de la douane et de la sécurité des frontières", "OFDF",
    "Bundesamt für Zoll und Grenzsicherheit", "BAZG",
    "Office fédéral de l'informatique et de la télécommunication", "OFIT",
    "Bundesamt für Informatik und Telekommunikation",
    "Office fédéral des constructions et de la logistique", "OFCL",
    "Bundesamt für Bauten und Logistik", "BBL",
    # ── DEFR / WBF ──────────────────────────────────────────────────────────
    "SG-DEFR", "GS-WBF",
    "Secrétariat d'État à l'économie", "SECO",
    "Staatssekretariat für Wirtschaft",
    "Secrétariat d'État à la formation, à la recherche et à l'innovation", "SEFRI",
    "Staatssekretariat für Bildung, Forschung und Innovation", "SBFI",
    "Office fédéral de l'agriculture", "OFAG",
    "Bundesamt für Landwirtschaft", "BLW",
    "Office fédéral pour l'approvisionnement économique du pays", "OFAE",
    "Bundesamt für wirtschaftliche Landesversorgung",
    "Office fédéral du logement", "OFL",
    "Bundesamt für Wohnungswesen", "BWO",
    "Office fédéral du service civil", "CIVI",
    "Bundesamt für Zivildienst", "ZIVI",
    # ── DETEC / UVEK ────────────────────────────────────────────────────────
    "SG-DETEC", "GS-UVEK",
    "Office fédéral des transports", "OFT",
    "Bundesamt für Verkehr", "BAV",
    "Office fédéral de l'aviation civile", "OFAC",
    "Bundesamt für Zivilluftfahrt", "BAZL",
    "Office fédéral de l'énergie", "OFEN",
    "Bundesamt für Energie", "BFE",
    "Office fédéral des routes", "OFROU",
    "Bundesamt für Strassen", "ASTRA",
    "Office fédéral de la communication", "OFCOM",
    "Bundesamt für Kommunikation", "BAKOM",
    "Office fédéral de l'environnement", "OFEV",
    "Bundesamt für Umwelt", "BAFU",
    "Office fédéral du développement territorial", "ARE",
    "Bundesamt für Raumentwicklung",
]

# Independent regulatory agencies and extra-departmental units (FR + DE)
INDEPENDENT_AGENCIES = [
    "Inspection fédérale de la sécurité nucléaire", "IFSN",
    "Eidgenössisches Nuklearsicherheitsinspektorat", "ENSI",
    "Inspection fédérale des installations à courant fort",
    "Eidgenössisches Starkstrominspektorat", "ESTI",
    "Service suisse d'enquête de sécurité", "SESE",
    "Schweizerische Sicherheitsuntersuchungsstelle", "SUST",
    "Commission fédérale de l'électricité", "EICom",
    "Eidgenössische Elektrizitätskommission", "ElCom",
    "Commission fédérale de la communication", "ComCom",
    "Eidgenössische Kommunikationskommission",
    "Autorité indépendante d'examen des plaintes en matière de radio-télévision", "AIEP",
    "Unabhängige Beschwerdeinstanz für Radio und Fernsehen", "UBI",
    "Commission fédérale de la poste", "PostCom",
    "Eidgenössische Postkommission",
    "Commission des chemins de fer", "RailCom",
    "Kommission für den Eisenbahnverkehr",
    "Surveillance des prix", "SPR",
    "Preisüberwachung", "PUE",
    "Commission de la concurrence", "COMCO",
    "Wettbewerbskommission", "WEKO",
    "Domaine des Écoles polytechniques fédérales", "domaine des EPF",
    "Bereich der Eidgenössischen Technischen Hochschulen", "ETH-Bereich",
    "Haute école fédérale en formation professionnelle", "HEFP",
    "Eidgenössisches Hochschulinstitut für Berufsbildung", "EHB",
    "Agence suisse pour l'encouragement de l'innovation", "Innosuisse",
    "Schweizerische Agentur für Innovationsförderung",
    "Autorité fédérale de surveillance des marchés financiers", "FINMA",
    "Eidgenössische Finanzmarktaufsicht",
    "Contrôle fédéral des finances", "CDF",
    "Eidgenössische Finanzkontrolle", "EFK",
    "Caisse fédérale de pensions PUBLICA", "Pensionskasse des Bundes PUBLICA", "PUBLICA",
    "Institut fédéral de la propriété intellectuelle", "IPI",
    "Eidgenössisches Institut für Geistiges Eigentum", "IGE",
    "Institut fédéral de métrologie", "METAS",
    "Eidgenössisches Institut für Metrologie",
    "Institut suisse de droit comparé", "ISDC",
    "Schweizerisches Institut für Rechtsvergleichung", "SIR",
    "Autorité fédérale de surveillance en matière de révision", "ASR",
    "Eidgenössische Revisionsaufsichtsbehörde", "RAB",
    "Commission fédérale des maisons de jeu", "CFMJ",
    "Eidgenössische Spielbankenkommission", "ESBK",
    "Commission arbitrale fédérale pour la gestion de droits d'auteur et de droits voisins", "CAF",
    "Eidgenössische Schiedskommission für die Verwertung von Urheberrechten", "ESchK",
    "Commission nationale de prévention de la torture", "CNPT",
    "Nationale Kommission zur Verhütung von Folter", "NKVF",
    "Commission fédérale des migrations", "CFM",
    "Eidgenössische Migrationskommission", "EKM",
    "Institut suisse des produits thérapeutiques", "Swissmedic",
    "Schweizerisches Heilmittelinstitut",
    "Musée national suisse", "MNS",
    "Schweizerisches Nationalmuseum", "SNM",
    "Fondation suisse pour la culture Pro Helvetia",
    "Schweizerische Kulturstiftung Pro Helvetia", "Pro Helvetia",
]


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
                # {"OR": DE_TERMS},   # temporarily disabled
                # {"OR": FR_TERMS},   # temporarily disabled
                # {"OR": DE_LEVEL},   # temporarily disabled
                # {"OR": FR_LEVEL},   # temporarily disabled
                {"OR": DEPARTMENTS},
                {"OR": ADMIN_UNITS},
                {"OR": INDEPENDENT_AGENCIES},
                {"OR": COUNCILLORS},
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

    def wait_for_download_url(
        self,
        query_id: str,
        *,
        max_wait_s: int = 24 * 60 * 60,
        poll_every_s: int = 60,
    ) -> str:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            rs = self.session.get(self.status_url, headers=self.headers, timeout=self.timeout)
            rs.raise_for_status()
            status_list = rs.json()

            job = next((j for j in status_list if str(j.get("id")) == str(query_id)), None)
            if not job:
                time.sleep(poll_every_s)
                continue

            if job.get("error"):
                raise RuntimeError(f"Swissdox job error: {job['error']}")

            url = job.get("downloadUrl")
            if url:
                return self._normalize_download_url(url)

            time.sleep(poll_every_s)

        raise TimeoutError(f"Swissdox: no downloadUrl after {max_wait_s}s (query_id={query_id})")

    def download_tsv_xz(self, download_url: str) -> pd.DataFrame:
        r = self.session.get(download_url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
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
    return s.strip(' """„\'')


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
    resume_query_id: str | None = None,
    test: bool = False,
) -> Dict[str, Path]:
    load_dotenv()
    api_key = os.getenv("SWISSDOX_API_KEY")
    api_secret = os.getenv("SWISSDOX_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing SWISSDOX_API_KEY / SWISSDOX_API_SECRET in .env")

    client = SwissdoxClient(api_key=api_key, api_secret=api_secret)

    if resume_query_id:
        print(f"[Swissdox] Resuming existing queryId={resume_query_id} (skipping submission)")
        qid = resume_query_id
    else:
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
        qid = client.submit_query(
            payload["yaml_payload"],
            name=payload["meta"]["name"],
            comment=payload["meta"]["comment"],
            expiration_date=payload["meta"]["expirationDate"],
            test=test,
        )
        print(f"[Swissdox] queryId={qid}")

    url = client.wait_for_download_url(qid)
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

    run_tag = time.strftime("%Y%m%d_%H%M%S")

    # One full parquet for downstream pipeline use
    p_parquet = out_dir / f"swissdox_all_{run_tag}.parquet"
    df_leads.to_parquet(p_parquet, index=False)
    print(f"[Swissdox] saved full parquet → {p_parquet}")

    # One CSV per year for easy inspection
    out_paths: Dict[str, Path] = {"full_parquet": p_parquet}
    if "pubtime" in df_leads.columns:
        df_leads["_year"] = df_leads["pubtime"].apply(
            lambda d: d.year if hasattr(d, "year") else int(str(d)[:4])
        )
        for year, df_year in df_leads.drop(columns="_year").groupby(df_leads["_year"]):
            p_csv = out_dir / f"swissdox_{year}_{run_tag}.csv"
            df_year.to_csv(p_csv, index=False)
            print(f"[Swissdox] saved year {year} CSV ({len(df_year):,} rows) → {p_csv}")
            out_paths[f"csv_{year}"] = p_csv
    else:
        p_csv = out_dir / f"swissdox_all_{run_tag}.csv"
        df_leads.to_csv(p_csv, index=False)
        out_paths["csv_all"] = p_csv

    pointer_path = out_dir / ".last_download"
    pointer_path.write_text(str(p_parquet), encoding="utf-8")
    print(f"[Swissdox] pointer → {pointer_path} (-> {p_parquet})")

    return out_paths
