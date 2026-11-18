# src/step2_targets.py
"""
Step 2 — reconnaissance des cibles (départements, unités, agences, conseillers)
dans le texte des articles, avec les mêmes mots-clés que la requête Swissdox
(src/keywords_list.py) et les règles de nettoyage de SWISSDOX_REPO
(scripts/step2_sentence_cutting.py), adaptées au niveau ARTICLE.

Une « cible » = une entité réelle, identifiée par son libellé `keyword`
(ex. "OFSP/BAG", "DDPS/VBS", "SECO", "Karin Keller-Sutter"). Tous les alias
d'une même cible (FR/DE, nom complet/abréviation, déclinaisons, variantes de
casse) donnent le même `keyword`.

Libellés : pour toute cible qui existait déjà dans l'ancienne taxonomie
(src/target_taxonomy.py), le libellé historique est conservé — les fichiers
GOLD sont joints sur (article_id, keyword). Les nouvelles cibles reçoivent un
libellé au même format ("abrév. FR/abrév. DE", sinon "nom FR/nom DE").

Règles de matching (identiques à SWISSDOX_REPO) :
  - noms complets (toutes déclinaisons) : insensibles à la casse, mot entier ;
  - abréviations : SENSIBLES à la casse, mot entier ; seules les graphies
    listées dans keywords_list.py sont acceptées (SECO et Seco, mais pas
    "seco") — sinon "BAR" capterait "bar", "DR" "Dr." ... ;
  - ' et ’ interchangeables, espaces multiples tolérés ;
  - plus long d'abord : « Administration fédérale des finances » l'emporte sur
    « Administration fédérale ».

Règles de nettoyage, appliquées phrase par phrase (la phrase ne sert qu'à
juger le contexte : la sortie reste une ligne par article × cible) :
  - crédits photo retirés de la phrase avant la recherche (« Name / EPA »,
    « (Bild: ...) », « © ... ») et légendes ignorées ;
  - phrases > 1000 caractères (tableaux) ou contenant une entité HTML ignorées ;
  - termes ignorés (STEP2_IGNORED_KEYWORDS) : reconnus mais sans cible, car
    ils ne désignent pas une entité précise (« Generalsekretariat » ...) ;
  - abréviations ambiguës : règles de contexte dans la même phrase
    (STEP2_ABBREV_RULES) ;
  - termes partagés entre deux cibles : attribution selon la date
    (STEP2_DATE_RULES) ; cible passée d'un département / statut à un autre :
    fiche choisie selon la date (STEP2_TRANSFER_DATES).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from src.keywords_list import COUNCILLORS, FEDERAL_UNITS, INDEPENDENT_AGENCIES
    from src.target_taxonomy import (
        ADMIN_UNIT_GROUPS, ADMINISTRATIVE_UNIT, DEPARTMENT_ALIASES, FEDERAL_COUNCILLOR,
        FEDERAL_DEPARTMENT, INDEPENDENT_AGENCY, INDEPENDENT_AGENCY_GROUPS,
        get_council_for_date, get_councillor_term,
    )
except ImportError:  # src/ directement sur le sys.path
    from keywords_list import COUNCILLORS, FEDERAL_UNITS, INDEPENDENT_AGENCIES
    from target_taxonomy import (
        ADMIN_UNIT_GROUPS, ADMINISTRATIVE_UNIT, DEPARTMENT_ALIASES, FEDERAL_COUNCILLOR,
        FEDERAL_DEPARTMENT, INDEPENDENT_AGENCY, INDEPENDENT_AGENCY_GROUPS,
        get_council_for_date, get_councillor_term,
    )


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def name_key(s: str) -> str:
    """Clé d'un nom complet : minuscules, apostrophes unifiées, espaces normalisés."""
    return _norm_spaces(s).replace("’", "'").lower()


def abbrev_key(s: str) -> str:
    """Clé d'une abréviation : casse CONSERVÉE, apostrophes unifiées, espaces normalisés."""
    return _norm_spaces(s).replace("’", "'")


# ════════════════════════════════════════════════════════════════════════════
# RÈGLES DE NETTOYAGE (reprises de SWISSDOX_REPO/scripts/step2_sentence_cutting.py,
# complétées pour les agences indépendantes)
# ════════════════════════════════════════════════════════════════════════════

# ── Termes ignorés ───────────────────────────────────────────────────────────
# Restent dans la requête Swissdox mais ne créent aucune ligne, car ils ne
# permettent pas d'identifier UNE cible :
#   - Secrétariat général / Generalsekretariat : commun aux 7 départements
#     (les abréviations SG-DFAE, GS-EDA, ... restent actives) ;
#   - Secrétariat d'Etat / Staatssekretariat / SEE / STS : noms et abréviations
#     alternatifs du Secrétariat d'Etat du DFAE (SEE-DFAE / STS-EDA et les noms
#     complets restent actifs) ;
#   - unités de l'armée (OLD), dans les deux langues (« Heer », « Luftwaffe » ...) ;
#   - [ajout accountability] Écoles polytechniques fédérales / Eidgenössische
#     Technische Hochschulen (pluriel) : nom alternatif commun à l'EPFZ et à
#     l'EPFL (EPFZ, EPFL, ETH-Bereich, ... restent actifs).
# Comparaison sur la forme de référence (1re forme du groupe) : toutes les
# déclinaisons, variantes de casse et apostrophes sont ignorées aussi.
STEP2_IGNORED_KEYWORDS = [
    "Secrétariat général", "Generalsekretariat",
    "Secrétariat d’Etat", "Staatssekretariat", "SEE", "STS",
    "Forces aériennes", "Luftwaffe",
    "Forces terrestres", "Heer",
    "Etat-major général", "Generalstab",
    "Écoles polytechniques fédérales", "Eidgenössische Technische Hochschulen",
]

# ── Règles de contexte pour les abréviations ambiguës ────────────────────────
# Une abréviation listée ici n'est retenue que si SA PHRASE remplit la règle.
#   "require" : la phrase doit contenir un de ces motifs ;
#   "exclude" : la phrase ne doit contenir aucun de ces motifs ;
#   "lang"    : retenue seulement dans les articles de cette langue.
# Clé = abréviation de référence (les variantes « Seco », « Astra » ... suivent
# la règle de leur abréviation). (?i:...) = insensible à la casse.
STEP2_ABBREV_RULES = {
    # ── Reprises telles quelles de SWISSDOX_REPO ─────────────────────────────
    "DC":    {"require": r"(?i:consulaire|konsular|Auslandschweizer|Suisses? de l.étranger|\bEDA\b|\bDFAE\b)"},
    "DR":    {"require": r"(?i:ressources|Ressourcen|Aussennetz|réseau extérieur|\bEDA\b|\bDFAE\b)"},
    "DV":    {"require": r"(?i:Völkerrecht|droit international|\bEDA\b|\bDFAE\b)"},
    "KD":    {"require": r"(?i:konsular|consulaire|Auslandschweizer|\bEDA\b)"},
    "OA":    {"require": r"(?i:Oberauditor|auditeur en chef|Militärjustiz|justice militaire)"},
    "AFS":   {"require": r"(?i:Archiv|archives)"},
    "EPA":   {"require": r"(?i:Personalamt|Bundespersonal|personnel de la Confédération|Bundesangestellt|Lohn|Löhne|salaire|\bEFD\b|\bDFF\b)"},
    "AFC":   {"lang": "fr",
              "exclude": r"AFC\s+[A-Z][a-z]|(?i:Sunderland|Bournemouth|Ajax|Wimbledon|Fussball|football|Fifa|Uefa|asiati|Asien|Champions League|Ligue des champions|Asian Cup|M23|Congo|Kongo|America First|cantonal)"},
    "BIT":   {"lang": "de", "exclude": r"(?i:travail|chômage|Arbeitsorganisation|\bILO\b|\bOIT\b|Bureau international)"},
    "BWL":   {"lang": "de", "exclude": r"BWL-|VWL|(?i:studi|Betriebswirtschaft|Universität|Hochschule|Bachelor|Master)"},
    "BBL":   {"lang": "de", "exclude": r"(?i:Béjart|Ballet|Butt|Brazilian)"},
    "ARE":   {"exclude": r"(?i:kantonal|\bKanton|cantonal|\bcanton)"},
    "ASTRA": {"exclude": r"(?i:Opel|Vauxhall|Satellit|satellite|\bSES\b|Zeneca)"},
    "BAR":   {"require": r"(?i:archiv|Akten|Dossier|Dokument|document|Bundesarchiv)"},
    "BFM":   {"exclude": r"BFM\s*TV|BFMTV|BFM\s+(?:Business|Paris|Lyon)|«BFM|(?i:chaîne|télévision)"},
    "BAK":   {"exclude": r"\b(?:[Dd]ie|der)\s+BAK\b(?!-)|BAK\s*(?:Basel|Economics)|BAKBASEL|BAK-Konjunktur|(?i:Konjunktur|Ecoplan|Forschungsinstitut|Wirtschaftsforsch|institut de recherche|recherches économiques|bâlois|BAK-Experten|BAK-Ökonom|Prognose|prévision|prophezei|Ökonom|économiste|Wachstum|croissance|\bBIP\b|\bPIB\b)"},
    "BSV":   {"exclude": r"BSV\s+(?:Bern|Muri|Stans|Wacker|Kriens|Borba)|(?i:Handball)|\b\d{1,2}:\d{1,2}\b"},
    "METAS": {"exclude": r"Metas\s+[A-ZÄÖÜ]|(?i:Zuckerberg|Facebook|Instagram|WhatsApp)"},

    # ── Ajouts accountability (abréviations courtes / homonymes fréquents) ───
    # Règles écrites a priori, à vérifier sur un vrai run (voir l'échantillon
    # de contrôle conseillé dans le README).
    "BJ":    {"require": r"(?i:Justiz|Rechtshilfe|Auslieferung|Gesetz|Bundesamt|\bEJPD\b)"},          # initiales
    "ST":    {"require": r"(?i:touris)"},                                                              # « ST » = Saint, sigles divers
    "SAS":   {"require": r"(?i:sillon|ferroviaire|Trassen|Eisenbahn|\bCFF\b|\bSBB\b)"},                 # compagnie aérienne SAS, société SAS
    "TVS":   {"require": r"(?i:Trassen|Eisenbahn|sillon|ferroviaire|\bSBB\b|\bCFF\b)"},
    "RFA":   {"require": r"(?i:alcool|spiritueux|eau-de-vie|absinthe|\bRégie\b)"},                    # République fédérale d'Allemagne
    "EAV":   {"require": r"(?i:Alkohol|Spirituosen|Schnaps|Absinth)"},
    "PSI":   {"require": r"(?i:Scherrer|Villigen|Forschung|recherche|Institut|Neutron|Proton|Synchrotron|\bETH\b|\bEPF)"},   # livres/pouce², partis PSI
    "IPS":   {"require": r"(?i:Scherrer|Villigen|recherche|Institut|Neutron|Proton|Synchrotron|\bEPF)"},
    "SCH":   {"require": r"(?i:hôtel|crédit|touris)"},
    "SGH":   {"require": r"(?i:Hotel|Kredit|Touris|Beherbergung)"},
    "FK":    {"require": r"(?i:Finanzkommission|Nationalrat|Ständerat|Parlament|Budget|Voranschlag)"},     # clubs de foot « FK ... »
    "Cdf":   {"require": r"(?i:commission|Conseil national|Conseil des [ÉE]tats|parlement|budget)"},
    "CFB":   {"require": r"(?i:banque|bancaire|surveillance)"},
    "EBK":   {"require": r"(?i:Bank|Aufsicht|Finanzmarkt)"},
    "UBI":   {"require": r"(?i:Radio|Fernseh|Beschwerde|\bSRG\b|\bSRF\b|Sendung|Programm)"},             # revenu de base (UBI)
    "SIR":   {"require": r"(?i:Rechtsvergleich|Institut)"},
    "ASR":   {"require": r"(?i:révision|réviseur|audit)"},
    "RAB":   {"require": r"(?i:Revision|Revisor|Aufsicht|Prüf)"},
    "SERV":  {"require": r"(?i:Export|Ausfuhr|Versicherung)"},
    "ASRE":  {"require": r"(?i:export|assurance)"},
    "MNS":   {"require": r"(?i:musée|exposition|collection)"},
    "SNM":   {"require": r"(?i:Museum|Ausstellung|Sammlung)"},
    "FNP":   {"require": r"(?i:forêt|neige|paysage|avalanche|\bWSL\b)"},
    "WSL":   {"require": r"(?i:Wald|Schnee|Lawine|Landschaft|forêt|neige|avalanche|paysage|Forschung|recherche|Birmensdorf|\bSLF\b)"},
    "IGE":   {"require": r"(?i:Eigentum|Patent|Marke|Urheber)"},
    "IPI":   {"require": r"(?i:propriété|brevet|marque|droit d.auteur)"},
}

# ── Termes partagés entre deux cibles : attribution selon la date ────────────
# L'Office fédéral des assurances privées dépendait du DFJP jusqu'au 30 juin
# 2003, puis du DFF (où le CSV le range comme nom alternatif de la « Caisse
# fédérale d'assurance »). Format : forme de référence -> liste de
# (date limite incluse "AAAA-MM-JJ", département EN de la fiche à retenir) ;
# la 1re qui convient gagne.
STEP2_DATE_RULES = {
    "Office fédéral des assurances privées": [("2003-06-30", "FDJP"), ("9999-12-31", "FDF")],
    "Bundesamt für Privatversicherungen":    [("2003-06-30", "FDJP"), ("9999-12-31", "FDF")],
}

# ── Cibles passées d'une fiche « OLD » à une fiche actuelle ──────────────────
# Même cible (même libellé), deux fiches dans keywords_list.py : jusqu'à la
# date incluse -> fiche OLD, ensuite -> fiche actuelle.
#   - Domaine des EPF (EPFZ, EPFL, PSI, WSL, Empa, Eawag) : DFI -> DEFR au
#     1er janvier 2013 ;
#   - METAS : office fédéral -> établissement autonome au 1er janvier 2013.
# Clé = libellé `keyword` de la cible (vérifié au chargement).
STEP2_TRANSFER_DATES = {
    "domaine des EPF/ETH-Bereich": "2012-12-31",
    "EPFZ/ETHZ": "2012-12-31",
    "EPFL/ETHL": "2012-12-31",
    "IPS/PSI": "2012-12-31",
    "FNP/WSL": "2012-12-31",
    "LFEM/Empa": "2012-12-31",
    "IFAEPE/EAWAG": "2012-12-31",
    "METAS": "2012-12-31",
}

# ── Phrases ignorées pour la recherche ──────────────────────────────────────
STEP2_MAX_SENTENCE_CHARS = 1000      # au-delà : tableaux de chiffres, code de graphiques
_HTML_ENTITY = re.compile(r"&(?:[A-Za-z]{2,8}|#[xX]?[0-9A-Fa-f]{1,6});")


# ════════════════════════════════════════════════════════════════════════════
# FICHES ET LIBELLÉS
# ════════════════════════════════════════════════════════════════════════════
_TYPE_MAP = {
    "Federal Department": FEDERAL_DEPARTMENT,
    "Federal Offices": ADMINISTRATIVE_UNIT,
    "OLD Federal Offices": ADMINISTRATIVE_UNIT,
    "Independent Agency": INDEPENDENT_AGENCY,
    "OLD Independent Agency": INDEPENDENT_AGENCY,
}

# Département : code EN (keywords_list) -> libellé "FR/DE" (format historique de parent_dept)
DEPT_LABEL: dict[str, str] = {
    u["upper_level"]: f'{u["fr_abbrev"]}/{u["de_abbrev"]}'
    for u in FEDERAL_UNITS if u["type"] == "Federal Department"
}
DEPT_EN: dict[str, str] = {label: code for code, label in DEPT_LABEL.items()}


@dataclass
class Form:
    reference: str            # 1re forme du groupe (sert de matched_alias)
    surfaces: list[str]       # toutes les graphies cherchées
    is_abbrev: bool
    lang: str                 # "fr" | "de" | "both"


@dataclass
class Record:
    """Une fiche de keywords_list.py (une ligne du CSV, ou un conseiller)."""
    rid: int
    target_type: str
    status: str               # "current" | "OLD"
    dept_en: str              # FDFA, FDHA, ... ("" pour un conseiller)
    name_fr: str
    name_de: str
    name_en: str
    abbrev_fr: str
    abbrev_de: str
    abbrev_en: str
    forms: list[Form] = field(default_factory=list)
    keyword: str = ""         # libellé de la cible (rempli après coup)

    @property
    def department(self) -> Optional[str]:
        return DEPT_LABEL.get(self.dept_en) if self.dept_en else None


def _entity_record(rid: int, u: dict) -> Record:
    rec = Record(
        rid=rid,
        target_type=_TYPE_MAP[u["type"]],
        status="OLD" if u["type"].upper().startswith("OLD") else "current",
        dept_en=u["upper_level"],
        name_fr=u["fr_names"][0][0] if u["fr_names"] else "",
        name_de=u["de_names"][0][0] if u["de_names"] else "",
        name_en=u["en_name"], abbrev_fr=u["fr_abbrev"], abbrev_de=u["de_abbrev"], abbrev_en=u["en_abbrev"],
    )
    for lang in ("fr", "de"):
        for variants in u[f"{lang}_names"]:
            rec.forms.append(Form(variants[0], list(variants), False, lang))
    case_variants = u.get("abbrev_case_variants", {})
    for lang in ("fr", "de"):
        for ab in [u[f"{lang}_abbrev"]] + u[f"{lang}_alt_abbrevs"]:
            if ab:
                rec.forms.append(Form(ab, [ab] + case_variants.get(ab, []), True, lang))
    return rec


def _councillor_record(rid: int, forms: list[str]) -> Record:
    name = forms[0]
    return Record(
        rid=rid, target_type=FEDERAL_COUNCILLOR, status="current", dept_en="",
        name_fr=name, name_de=name, name_en=name, abbrev_fr="", abbrev_de="", abbrev_en="",
        forms=[Form(name, list(forms), False, "both")],
    )


def _legacy_alias_index() -> tuple[dict[str, str], dict[str, str]]:
    """Alias de l'ancienne taxonomie -> libellé historique (noms, abréviations)."""
    pairs = list(DEPARTMENT_ALIASES.items())
    pairs += [(a, c) for c, _p, aliases in ADMIN_UNIT_GROUPS for a in aliases]
    pairs += [(a, c) for c, aliases in INDEPENDENT_AGENCY_GROUPS for a in aliases]
    by_name = {name_key(a): c for a, c in pairs}
    by_abbrev = {abbrev_key(a): c for a, c in pairs}
    return by_name, by_abbrev


def _new_label(rec: Record) -> str:
    """"abrév. FR/abrév. DE" (une seule si identiques, casse ignorée), sinon "nom FR/nom DE"."""
    fr, de = rec.abbrev_fr, rec.abbrev_de
    if not (fr or de):
        fr, de = rec.name_fr, rec.name_de
    if not de or fr.lower() == de.lower():
        return fr
    return de if not fr else f"{fr}/{de}"


def _assign_labels(records: list[Record]) -> None:
    by_name, by_abbrev = _legacy_alias_index()
    for rec in records:
        if rec.target_type == FEDERAL_COUNCILLOR:
            rec.keyword = rec.name_fr
            continue
        legacy = set()
        for f in rec.forms:
            for s in f.surfaces:
                hit = by_abbrev.get(abbrev_key(s)) if f.is_abbrev else by_name.get(name_key(s))
                if hit:
                    legacy.add(hit)
        if len(legacy) > 1:
            raise AssertionError(f"Fiche {rec.name_fr or rec.name_de!r} : plusieurs libellés historiques {legacy}")
        rec.keyword = legacy.pop() if legacy else _new_label(rec)


RECORDS: list[Record] = []
for _u in FEDERAL_UNITS + INDEPENDENT_AGENCIES:
    RECORDS.append(_entity_record(len(RECORDS), _u))
for _forms in COUNCILLORS:
    RECORDS.append(_councillor_record(len(RECORDS), _forms))
_assign_labels(RECORDS)

RECORDS_BY_KEYWORD: dict[str, list[Record]] = {}
for _r in RECORDS:
    RECORDS_BY_KEYWORD.setdefault(_r.keyword, []).append(_r)


# ════════════════════════════════════════════════════════════════════════════
# INDEX DE RECHERCHE
# ════════════════════════════════════════════════════════════════════════════
# Clé de surface -> [(rid, n° de forme)]. Noms : name_key ; abréviations :
# abbrev_key (casse conservée), préfixées pour ne jamais se confondre.
_SURFACE_INDEX: dict[str, list[tuple[int, int]]] = {}
for _r in RECORDS:
    for _fi, _f in enumerate(_r.forms):
        for _s in _f.surfaces:
            _k = ("A|" + abbrev_key(_s)) if _f.is_abbrev else ("N|" + name_key(_s))
            if (_r.rid, _fi) not in _SURFACE_INDEX.setdefault(_k, []):
                _SURFACE_INDEX[_k].append((_r.rid, _fi))


def _to_regex(term: str) -> str:
    pat = re.escape(_norm_spaces(term)).replace("\\ ", r"\s+")
    return re.sub(r"['’]", "['’]", pat)


_WORD = re.compile(r"(?<!\w)\w+")


class Matcher:
    """Équivalent rapide d'une grande alternance « plus long d'abord » :

        (?<!\\w)(terme1|terme2|...)(?!\\w)   — noms (?i:...), abréviations sensibles à la casse

    Les termes sont rangés par premier mot (en minuscules) ; à chaque début de
    mot de la phrase, on n'essaie que les termes qui commencent par ce mot, du
    plus long au plus court, et le 1er qui correspond gagne (puis on reprend
    après lui). Même résultat que l'alternance, ~100x plus rapide.
    """

    def __init__(self) -> None:
        by_first: dict[str, list[tuple[int, re.Pattern]]] = {}
        for key in _SURFACE_INDEX:
            kind, surface = key[0], key[2:]
            first = _WORD.match(surface)
            if first is None:
                raise AssertionError(f"Terme sans mot initial : {surface!r}")
            core = _to_regex(surface) + r"(?!\w)"
            rx = re.compile(core, 0 if kind == "A" else re.IGNORECASE)
            by_first.setdefault(first.group(0).lower(), []).append((len(surface), rx))
        self._by_first = {w: [rx for _, rx in sorted(c, key=lambda e: e[0], reverse=True)]
                          for w, c in by_first.items()}

    def finditer(self, text: str):
        """Yield (start, matched text), comme re.finditer sur l'alternance."""
        pos = 0
        while True:
            w = _WORD.search(text, pos)
            if w is None:
                return
            end = None
            for rx in self._by_first.get(w.group(0).lower(), ()):
                m = rx.match(text, w.start())
                if m:
                    end = m.end()
                    yield w.start(), m.group(0)
                    break
            pos = end if end is not None else w.end()


def build_pattern() -> Matcher:
    return Matcher()


def _lookup(text: str) -> list[tuple[int, int]]:
    """Fiches/formes correspondant au texte trouvé (abréviation exacte prioritaire)."""
    return _SURFACE_INDEX.get("A|" + abbrev_key(text)) or _SURFACE_INDEX.get("N|" + name_key(text), [])


# ── Règles compilées + vérifications au chargement ──────────────────────────
_ALL_REFERENCES = {f.reference for r in RECORDS for f in r.forms}
_ALL_REF_KEYS = {name_key(x) for x in _ALL_REFERENCES}

_IGNORED_KEYS = {name_key(k) for k in STEP2_IGNORED_KEYWORDS}
_unknown = _IGNORED_KEYS - _ALL_REF_KEYS
if _unknown:
    raise AssertionError(f"STEP2_IGNORED_KEYWORDS absents de keywords_list.py : {_unknown}")

_ABBREV_RULES = {
    k: {kind: (rx if kind == "lang" else re.compile(rx)) for kind, rx in rule.items()}
    for k, rule in STEP2_ABBREV_RULES.items()
}
_abbrev_refs = {f.reference for r in RECORDS for f in r.forms if f.is_abbrev}
_unknown = set(_ABBREV_RULES) - _abbrev_refs
if _unknown:
    raise AssertionError(f"STEP2_ABBREV_RULES : abréviations absentes de keywords_list.py : {_unknown}")

_DATE_RULES = {name_key(k): v for k, v in STEP2_DATE_RULES.items()}
_unknown = set(_DATE_RULES) - _ALL_REF_KEYS
if _unknown:
    raise AssertionError(f"STEP2_DATE_RULES absents de keywords_list.py : {_unknown}")

_unknown = set(STEP2_TRANSFER_DATES) - set(RECORDS_BY_KEYWORD)
if _unknown:
    raise AssertionError(f"STEP2_TRANSFER_DATES : libellés inconnus {_unknown}")
for _kw, _recs in RECORDS_BY_KEYWORD.items():
    if len(_recs) > 1:
        if _kw not in STEP2_TRANSFER_DATES or sorted(r.status for r in _recs) != ["OLD", "current"]:
            raise AssertionError(
                f"Libellé {_kw!r} porté par {len(_recs)} fiches sans règle de transfert "
                f"(STEP2_TRANSFER_DATES, une fiche OLD + une actuelle)"
            )


def _form_is_ignored(form: Form) -> bool:
    return name_key(form.reference) in _IGNORED_KEYS


# Tout terme partagé par plusieurs cibles doit être ignoré ou tranché par une
# règle de date — sinon on ne saurait pas à quelle cible attribuer le match.
for _k, _hits in _SURFACE_INDEX.items():
    _kws = {RECORDS[rid].keyword for rid, _fi in _hits
            if not _form_is_ignored(RECORDS[rid].forms[_fi])}
    if len(_kws) > 1 and not any(
        name_key(RECORDS[rid].forms[fi].reference) in _DATE_RULES for rid, fi in _hits
    ):
        raise AssertionError(f"Terme {_k[2:]!r} partagé par {_kws} sans règle (ignoré / date)")


def _day(pubtime) -> str:
    s = "" if pubtime is None else str(pubtime)
    return "" if s in ("", "nan", "NaT", "None") else s[:10]


def _passes_context(form: Form, sentence: str, lang) -> bool:
    rule = _ABBREV_RULES.get(form.reference) if form.is_abbrev else None
    if rule is None:
        return True
    if "lang" in rule and str(lang).strip().lower() != rule["lang"]:
        return False
    if "require" in rule and not rule["require"].search(sentence):
        return False
    if "exclude" in rule and rule["exclude"].search(sentence):
        return False
    return True


def _pick_by_date(hits: list[tuple[int, int]], day: str) -> list[tuple[int, int]]:
    """Terme partagé entre cibles : garde la fiche désignée par STEP2_DATE_RULES."""
    for rid, fi in hits:
        rule = _DATE_RULES.get(name_key(RECORDS[rid].forms[fi].reference))
        if rule is None:
            continue
        if not day:
            return []          # pas de date : impossible de trancher
        for limit, dept_en in rule:
            if day <= limit:
                return [(r, f) for r, f in hits if RECORDS[r].dept_en == dept_en]
    return hits


def resolve_sentence(sentence: str, pattern: Matcher, lang, day: str,
                     offset: int, found: dict[str, dict[str, int]], counts: dict[str, int]) -> None:
    """Ajoute à `found` {keyword: {forme telle qu'écrite dans le texte: 1re
    position}} les cibles de la phrase (et compte les mentions dans `counts`).
    On garde le mot exact de l'article (déclinaison, accents, apostrophe,
    casse) pour que les prompts des étapes suivantes l'ancrent dans le texte."""
    for start, text in pattern.finditer(sentence):
        hits = [(rid, fi) for rid, fi in _lookup(text)
                if not _form_is_ignored(RECORDS[rid].forms[fi])]
        hits = [(rid, fi) for rid, fi in hits if _passes_context(RECORDS[rid].forms[fi], sentence, lang)]
        if len({RECORDS[rid].keyword for rid, _ in hits}) > 1:
            hits = _pick_by_date(hits, day)
        if not hits:
            continue
        rid, fi = hits[0]
        kw = RECORDS[rid].keyword
        found.setdefault(kw, {}).setdefault(" ".join(text.split()), offset + start)
        counts[kw] = counts.get(kw, 0) + 1


def best_alias(aliases: dict[str, int]) -> str:
    """Alias le plus long (le plus explicite) ; à égalité, le premier apparu."""
    return max(aliases, key=lambda a: (len(a), -aliases[a]))


# ════════════════════════════════════════════════════════════════════════════
# MÉTADONNÉES
# ════════════════════════════════════════════════════════════════════════════
def record_for(keyword: str, pubtime=None) -> Record:
    """Fiche à utiliser pour la cible à la date de l'article."""
    recs = RECORDS_BY_KEYWORD[keyword]
    if len(recs) == 1:
        return recs[0]
    day, limit = _day(pubtime), STEP2_TRANSFER_DATES[keyword]
    status = "current" if (not day or day > limit) else "OLD"
    return next(r for r in recs if r.status == status)


def target_metadata(keyword: str, pubtime=None, language=None) -> dict:
    """Colonnes de métadonnées d'une ligne article × cible.

    parent_dept : format historique (identique aux fichiers GOLD) — le
                  département lui-même, celui d'une unité, celui dirigé par un
                  conseiller à la date de l'article ; vide pour une agence.
    department  : département de rattachement pour TOUS les types (agences
                  comprises, d'après le CSV), même format "FR/DE".
    """
    rec = record_for(keyword, pubtime)
    if rec.target_type == FEDERAL_COUNCILLOR:
        comp = get_council_for_date(pubtime)
        dept = next((d for d, name in comp.items() if name == keyword), None)
        parent = dept
    else:
        dept = rec.department
        parent = None if rec.target_type == INDEPENDENT_AGENCY else dept
    lang = str(language).strip().lower() if language is not None else ""
    if rec.target_type == FEDERAL_COUNCILLOR:
        kw_abbrev = keyword
    elif lang == "de":
        kw_abbrev = rec.abbrev_de or rec.abbrev_fr or keyword
    else:
        kw_abbrev = rec.abbrev_fr or rec.abbrev_de or keyword
    return {
        "keyword_abbrev": kw_abbrev,
        "target_type": rec.target_type,
        "target_status": rec.status,
        "parent_dept": parent,
        "department": dept,
        "department_en": DEPT_EN.get(dept) if dept else None,
        "target_name_fr": rec.name_fr or None,
        "target_name_de": rec.name_de or None,
        "target_name_en": rec.name_en or None,
        "target_abbrev_fr": rec.abbrev_fr or None,
        "target_abbrev_de": rec.abbrev_de or None,
        "target_abbrev_en": rec.abbrev_en or None,
    }


def reference_table():
    """Une ligne par fiche : tout ce qu'il faut savoir sur chaque cible
    (fichier keywords_reference.csv du step 2)."""
    import pandas as pd
    rows = []
    for rec in RECORDS:
        valid_from = valid_to = None
        if rec.keyword in STEP2_TRANSFER_DATES:
            limit = STEP2_TRANSFER_DATES[rec.keyword]
            if rec.status == "OLD":
                valid_to = limit
            else:
                valid_from = str((pd.Timestamp(limit) + pd.Timedelta(days=1)).date())
        if rec.target_type == FEDERAL_COUNCILLOR:
            term = get_councillor_term(rec.keyword)
            if term:
                valid_from, valid_to = str(term[0]), (None if term[1].year == 9999 else str(term[1]))
        active = [f for f in rec.forms if not _form_is_ignored(f)]
        rows.append({
            "keyword": rec.keyword,
            "target_type": rec.target_type,
            "target_status": rec.status,
            "department": rec.department,
            "department_en": rec.dept_en or None,
            "target_name_fr": rec.name_fr or None,
            "target_name_de": rec.name_de or None,
            "target_name_en": rec.name_en or None,
            "target_abbrev_fr": rec.abbrev_fr or None,
            "target_abbrev_de": rec.abbrev_de or None,
            "target_abbrev_en": rec.abbrev_en or None,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "abbrev_rules": " | ".join(f"{f.reference}: {STEP2_ABBREV_RULES[f.reference]}"
                                       for f in active if f.is_abbrev and f.reference in STEP2_ABBREV_RULES) or None,
            "ignored_forms": " | ".join(f.reference for f in rec.forms if _form_is_ignored(f)) or None,
            "n_surfaces": sum(len(f.surfaces) for f in active),
            "surfaces": " | ".join(s for f in active for s in f.surfaces),
        })
    return pd.DataFrame(rows)


def legacy_targets_lost() -> list[str]:
    """Libellés de l'ancienne taxonomie qui ne correspondent plus à aucune fiche."""
    legacy = set(DEPARTMENT_ALIASES.values()) | {c for c, _p, _a in ADMIN_UNIT_GROUPS} \
        | {c for c, _a in INDEPENDENT_AGENCY_GROUPS}
    return sorted(legacy - set(RECORDS_BY_KEYWORD))
