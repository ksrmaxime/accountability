# src/target_taxonomy.py
"""
Single source of truth for the "targets" the accountability pipeline tracks
in Swiss press articles: the 7 federal departments, their administrative
units, the independent agencies, and the federal councillors (2000-present).

This module replaces, in one place, what used to be split across:
  - src/download_src.py      (DEPARTMENTS / ADMIN_UNITS / INDEPENDENT_AGENCIES /
                               COUNCILLORS literal lists used to build the
                               Swissdox query)
  - scripts/tag_keywords.py  (per-alias language tagging + regex matching +
                               abbreviation-vs-full-name deduplication)
  - src/analysis_config.py   (alias -> target_type / parent_dept mapping)
  - src/run5_prompts.py      (federal council compositions by date)

The key addition versus the old pipeline: every alias of the *same*
real-world entity (its French and German, abbreviated and spelled-out
forms) is grouped under one CANONICAL label. Matching an article now
returns canonical target identities directly, so "OFT" (found in a French
article) and "BAV" (found in a German article) are recognised as the same
target instead of two different ones, and an article that name-drops both
"VBS" and its full German name only ever produces one row for that target.

Public API
----------
  TARGET_TYPES                : the 4 category labels used downstream
  match_targets(text, lang)   -> list[str] of canonical target names found
  classify_target(canonical)  -> dict with target_type / parent_dept
  COUNCILLORS                 : list[str] of the 26 councillor names
  get_council_for_date(pubtime)      -> {dept_code: councillor_name}
  get_councillor_term(name)          -> (start_date, end_date) | None
  self_check()                       -> raises AssertionError on any gap
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Target type labels (exactly as used downstream, e.g. in analysis output)
# ---------------------------------------------------------------------------
FEDERAL_DEPARTMENT = "Federal Department"
ADMINISTRATIVE_UNIT = "Administrative Unit"
INDEPENDENT_AGENCY = "Independent Agency"
FEDERAL_COUNCILLOR = "Federal Councillor"

TARGET_TYPES = [FEDERAL_DEPARTMENT, ADMINISTRATIVE_UNIT, INDEPENDENT_AGENCY, FEDERAL_COUNCILLOR]


# ---------------------------------------------------------------------------
# 1) Federal departments — alias -> canonical department code.
#    The canonical code IS the department's own "keyword" AND its own
#    parent_dept (a department is its own parent).
# ---------------------------------------------------------------------------
DEPARTMENT_CODES: list[str] = [
    "DFAE/EDA", "DFI/EDI", "DFJP/EJPD", "DDPS/VBS",
    "DFF/EFD", "DEFR/WBF", "DETEC/UVEK",
]

DEPARTMENT_ALIASES: dict[str, str] = {
    "VBS": "DDPS/VBS", "DDPS": "DDPS/VBS",
    "Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport": "DDPS/VBS",
    "Département fédéral de la défense, de la protection de la population et des sports": "DDPS/VBS",

    "EDA": "DFAE/EDA", "DFAE": "DFAE/EDA",
    "Eidgenössische Departement für auswärtige Angelegenheiten": "DFAE/EDA",
    "Département fédéral des affaires étrangères": "DFAE/EDA",

    "UVEK": "DETEC/UVEK", "DETEC": "DETEC/UVEK",
    "Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation": "DETEC/UVEK",
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication": "DETEC/UVEK",

    "EJPD": "DFJP/EJPD", "DFJP": "DFJP/EJPD",
    "Eidgenössische Justiz- und Polizeidepartement": "DFJP/EJPD",
    "Département fédéral de justice et police": "DFJP/EJPD",

    "EDI": "DFI/EDI", "DFI": "DFI/EDI",
    "Eidgenössische Departement des Innern": "DFI/EDI",
    "Département fédéral de l'intérieur": "DFI/EDI",

    "EFD": "DFF/EFD", "DFF": "DFF/EFD",
    "Eidgenössische Finanzdepartement": "DFF/EFD",
    "Département fédéral des finances": "DFF/EFD",

    "WBF": "DEFR/WBF", "DEFR": "DEFR/WBF",
    "Eidgenössische Departement für Wirtschaft, Bildung und Forschung": "DEFR/WBF",
    "Département fédéral de l'économie, de la formation et de la recherche": "DEFR/WBF",
}


# ---------------------------------------------------------------------------
# 2) Administrative units — grouped by real-world entity.
#    Each tuple: (canonical_label, parent_dept_code, [aliases...])
# ---------------------------------------------------------------------------
ADMIN_UNIT_GROUPS: list[tuple[str, str, list[str]]] = [
    # ── DFAE / EDA ──────────────────────────────────────────────────────────
    ("SG-DFAE/GS-EDA", "DFAE/EDA", ["SG-DFAE", "GS-EDA"]),
    ("DDIP", "DFAE/EDA", ["Direction du droit international public", "DDIP", "Direktion für Völkerrecht"]),
    ("Direction consulaire/Konsularische Direktion", "DFAE/EDA",
     ["Direction consulaire", "Konsularische Direktion"]),
    ("DDC/DEZA", "DFAE/EDA",
     ["Direction du développement et de la coopération", "DDC",
      "Direktion für Entwicklung und Zusammenarbeit", "DEZA"]),
    ("Direction des ressources/Direktion für Ressourcen", "DFAE/EDA",
     ["Direction des ressources", "Direktion für Ressourcen"]),

    # ── DFI / EDI ───────────────────────────────────────────────────────────
    ("SG-DFI/GS-EDI", "DFI/EDI", ["SG-DFI", "GS-EDI"]),
    ("BFEG/EBG", "DFI/EDI",
     ["Bureau fédéral de l'égalité entre femmes et hommes", "BFEG",
      "Eidgenössisches Büro für die Gleichstellung von Frau und Mann", "EBG"]),
    ("OFC/BAK", "DFI/EDI",
     ["Office fédéral de la culture", "OFC", "Bundesamt für Kultur", "BAK"]),
    ("AFS", "DFI/EDI",
     ["Archives fédérales suisses", "AFS", "Schweizerisches Bundesarchiv"]),
    ("MétéoSuisse/MeteoSchweiz", "DFI/EDI",
     ["Office fédéral de météorologie et de climatologie", "MétéoSuisse",
      "Bundesamt für Meteorologie und Klimatologie", "MeteoSchweiz"]),
    ("OFSP/BAG", "DFI/EDI",
     ["Office fédéral de la santé publique", "OFSP", "Bundesamt für Gesundheit", "BAG"]),
    ("OSAV/BLV", "DFI/EDI",
     ["Office fédéral de la sécurité alimentaire et des affaires vétérinaires", "OSAV",
      "Bundesamt für Lebensmittelsicherheit und Veterinärwesen", "BLV"]),
    ("OFS/BFS", "DFI/EDI",
     ["Office fédéral de la statistique", "OFS", "Bundesamt für Statistik", "BFS"]),
    ("OFAS/BSV", "DFI/EDI",
     ["Office fédéral des assurances sociales", "OFAS", "Bundesamt für Sozialversicherungen", "BSV"]),

    # ── DFJP / EJPD ─────────────────────────────────────────────────────────
    ("SG-DFJP/GS-EJPD", "DFJP/EJPD", ["SG-DFJP", "GS-EJPD"]),
    ("SEM", "DFJP/EJPD",
     ["Secrétariat d'État aux migrations", "SEM", "Staatssekretariat für Migration"]),
    ("OFJ", "DFJP/EJPD", ["Office fédéral de la justice", "OFJ", "Bundesamt für Justiz"]),
    ("fedpol", "DFJP/EJPD", ["Office fédéral de la police", "fedpol", "Bundesamt für Polizei"]),
    ("Service SCPT/ÜPF", "DFJP/EJPD",
     ["Service Surveillance de la correspondance par poste et télécommunication", "Service SCPT",
      "Dienst Überwachung Post- und Fernmeldeverkehr", "ÜPF"]),

    # ── DDPS / VBS ──────────────────────────────────────────────────────────
    ("SG-DDPS/GS-VBS", "DDPS/VBS", ["SG-DDPS", "GS-VBS"]),
    ("OFPP/BABS", "DDPS/VBS",
     ["Office fédéral de la protection de la population", "OFPP",
      "Bundesamt für Bevölkerungsschutz", "BABS"]),
    ("armasuisse", "DDPS/VBS",
     ["Office fédéral de l'armement", "armasuisse", "Bundesamt für Rüstung"]),
    ("swisstopo", "DDPS/VBS",
     ["Office fédéral de topographie", "swisstopo", "Bundesamt für Landestopografie"]),
    ("OFSPO/BASPO", "DDPS/VBS",
     ["Office fédéral du sport", "OFSPO", "Bundesamt für Sport", "BASPO"]),
    ("OFCS/BACS", "DDPS/VBS",
     ["Office fédéral de la cybersécurité", "OFCS", "Bundesamt für Cybersicherheit", "BACS"]),
    ("SEPOS", "DDPS/VBS",
     ["Secrétariat d'État à la politique de sécurité", "SEPOS", "Staatssekretariat für Sicherheitspolitik"]),
    ("Armée suisse/Schweizer Armee", "DDPS/VBS", ["Armée suisse", "Schweizer Armee"]),
    ("SRC/NDB", "DDPS/VBS",
     ["Service de renseignement de la Confédération", "SRC",
      "Nachrichtendienst des Bundes", "NDB"]),
    ("OAC/Oberauditorat", "DDPS/VBS", ["Office de l'auditeur en chef", "OAC", "Oberauditorat"]),

    # ── DFF / EFD ───────────────────────────────────────────────────────────
    ("SG-DFF/GS-EFD", "DFF/EFD", ["SG-DFF", "GS-EFD"]),
    ("SFI/SIF", "DFF/EFD",
     ["Secrétariat d'État aux questions financières internationales", "SFI",
      "Staatssekretariat für internationale Finanzfragen", "SIF"]),
    ("AFF/EFV", "DFF/EFD",
     ["Administration fédérale des finances", "AFF", "Eidgenössische Finanzverwaltung", "EFV"]),
    ("OFPER/EPA", "DFF/EFD",
     ["Office fédéral du personnel", "OFPER", "Eidgenössisches Personalamt", "EPA"]),
    ("AFC/ESTV", "DFF/EFD",
     ["Administration fédérale des contributions", "AFC", "Eidgenössische Steuerverwaltung", "ESTV"]),
    ("OFDF/BAZG", "DFF/EFD",
     ["Office fédéral de la douane et de la sécurité des frontières", "OFDF",
      "Bundesamt für Zoll und Grenzsicherheit", "BAZG"]),
    ("OFIT", "DFF/EFD",
     ["Office fédéral de l'informatique et de la télécommunication", "OFIT",
      "Bundesamt für Informatik und Telekommunikation"]),
    ("OFCL/BBL", "DFF/EFD",
     ["Office fédéral des constructions et de la logistique", "OFCL",
      "Bundesamt für Bauten und Logistik", "BBL"]),

    # ── DEFR / WBF ──────────────────────────────────────────────────────────
    ("SG-DEFR/GS-WBF", "DEFR/WBF", ["SG-DEFR", "GS-WBF"]),
    ("SECO", "DEFR/WBF", ["Secrétariat d'État à l'économie", "SECO", "Staatssekretariat für Wirtschaft"]),
    ("SEFRI/SBFI", "DEFR/WBF",
     ["Secrétariat d'État à la formation, à la recherche et à l'innovation", "SEFRI",
      "Staatssekretariat für Bildung, Forschung und Innovation", "SBFI"]),
    ("OFAG/BLW", "DEFR/WBF",
     ["Office fédéral de l'agriculture", "OFAG", "Bundesamt für Landwirtschaft", "BLW"]),
    ("OFAE", "DEFR/WBF",
     ["Office fédéral pour l'approvisionnement économique du pays", "OFAE",
      "Bundesamt für wirtschaftliche Landesversorgung"]),
    ("OFL/BWO", "DEFR/WBF",
     ["Office fédéral du logement", "OFL", "Bundesamt für Wohnungswesen", "BWO"]),
    ("CIVI/ZIVI", "DEFR/WBF",
     ["Office fédéral du service civil", "CIVI", "Bundesamt für Zivildienst", "ZIVI"]),

    # ── DETEC / UVEK ────────────────────────────────────────────────────────
    ("SG-DETEC/GS-UVEK", "DETEC/UVEK", ["SG-DETEC", "GS-UVEK"]),
    ("OFT/BAV", "DETEC/UVEK", ["Office fédéral des transports", "OFT", "Bundesamt für Verkehr", "BAV"]),
    ("OFAC/BAZL", "DETEC/UVEK",
     ["Office fédéral de l'aviation civile", "OFAC", "Bundesamt für Zivilluftfahrt", "BAZL"]),
    ("OFEN/BFE", "DETEC/UVEK", ["Office fédéral de l'énergie", "OFEN", "Bundesamt für Energie", "BFE"]),
    ("OFROU/ASTRA", "DETEC/UVEK",
     ["Office fédéral des routes", "OFROU", "Bundesamt für Strassen", "ASTRA"]),
    ("OFCOM/BAKOM", "DETEC/UVEK",
     ["Office fédéral de la communication", "OFCOM", "Bundesamt für Kommunikation", "BAKOM"]),
    ("OFEV/BAFU", "DETEC/UVEK",
     ["Office fédéral de l'environnement", "OFEV", "Bundesamt für Umwelt", "BAFU"]),
    ("ARE", "DETEC/UVEK",
     ["Office fédéral du développement territorial", "ARE", "Bundesamt für Raumentwicklung"]),
]


# ---------------------------------------------------------------------------
# 3) Independent agencies — grouped the same way. No parent department.
#    NOTE: this single category covers both classic sector regulators
#    (FINMA, COMCO/WEKO, ElCom, ...) and non-regulatory extra-departmental
#    units (Innosuisse, PUBLICA, Pro Helvetia, ...). The old pipeline named
#    it "Independent Regulatory Agency", which wrongly implied every entry
#    was a regulator — renamed here to "Independent Agency" with no change
#    to which entities are included.
# ---------------------------------------------------------------------------
INDEPENDENT_AGENCY_GROUPS: list[tuple[str, list[str]]] = [
    ("IFSN/ENSI", ["Inspection fédérale de la sécurité nucléaire", "IFSN",
                   "Eidgenössisches Nuklearsicherheitsinspektorat", "ENSI"]),
    ("ESTI", ["Inspection fédérale des installations à courant fort",
              "Eidgenössisches Starkstrominspektorat", "ESTI"]),
    ("SESE/SUST", ["Service suisse d'enquête de sécurité", "SESE",
                   "Schweizerische Sicherheitsuntersuchungsstelle", "SUST"]),
    ("EICom/ElCom", ["Commission fédérale de l'électricité", "EICom",
                     "Eidgenössische Elektrizitätskommission", "ElCom"]),
    ("ComCom", ["Commission fédérale de la communication", "ComCom",
                "Eidgenössische Kommunikationskommission"]),
    ("AIEP/UBI", ["Autorité indépendante d'examen des plaintes en matière de radio-télévision", "AIEP",
                  "Unabhängige Beschwerdeinstanz für Radio und Fernsehen", "UBI"]),
    ("PostCom", ["Commission fédérale de la poste", "PostCom", "Eidgenössische Postkommission"]),
    ("RailCom", ["Commission des chemins de fer", "RailCom", "Kommission für den Eisenbahnverkehr"]),
    ("SPR/PUE", ["Surveillance des prix", "SPR", "Preisüberwachung", "PUE"]),
    ("COMCO/WEKO", ["Commission de la concurrence", "COMCO", "Wettbewerbskommission", "WEKO"]),
    ("domaine des EPF/ETH-Bereich", ["Domaine des Écoles polytechniques fédérales", "domaine des EPF",
                                      "Bereich der Eidgenössischen Technischen Hochschulen", "ETH-Bereich"]),
    ("HEFP/EHB", ["Haute école fédérale en formation professionnelle", "HEFP",
                  "Eidgenössisches Hochschulinstitut für Berufsbildung", "EHB"]),
    ("Innosuisse", ["Agence suisse pour l'encouragement de l'innovation", "Innosuisse",
                    "Schweizerische Agentur für Innovationsförderung"]),
    ("FINMA", ["Autorité fédérale de surveillance des marchés financiers", "FINMA",
               "Eidgenössische Finanzmarktaufsicht"]),
    ("CDF/EFK", ["Contrôle fédéral des finances", "CDF", "Eidgenössische Finanzkontrolle", "EFK"]),
    ("PUBLICA", ["Caisse fédérale de pensions PUBLICA", "Pensionskasse des Bundes PUBLICA", "PUBLICA"]),
    ("IPI/IGE", ["Institut fédéral de la propriété intellectuelle", "IPI",
                 "Eidgenössisches Institut für Geistiges Eigentum", "IGE"]),
    ("METAS", ["Institut fédéral de métrologie", "METAS", "Eidgenössisches Institut für Metrologie"]),
    ("ISDC/SIR", ["Institut suisse de droit comparé", "ISDC",
                  "Schweizerisches Institut für Rechtsvergleichung", "SIR"]),
    ("ASR/RAB", ["Autorité fédérale de surveillance en matière de révision", "ASR",
                 "Eidgenössische Revisionsaufsichtsbehörde", "RAB"]),
    ("CFMJ/ESBK", ["Commission fédérale des maisons de jeu", "CFMJ",
                   "Eidgenössische Spielbankenkommission", "ESBK"]),
    ("CAF/ESchK", ["Commission arbitrale fédérale pour la gestion de droits d'auteur et de droits voisins", "CAF",
                   "Eidgenössische Schiedskommission für die Verwertung von Urheberrechten", "ESchK"]),
    ("CNPT/NKVF", ["Commission nationale de prévention de la torture", "CNPT",
                   "Nationale Kommission zur Verhütung von Folter", "NKVF"]),
    ("CFM/EKM", ["Commission fédérale des migrations", "CFM", "Eidgenössische Migrationskommission", "EKM"]),
    ("Swissmedic", ["Institut suisse des produits thérapeutiques", "Swissmedic",
                    "Schweizerisches Heilmittelinstitut"]),
    ("MNS/SNM", ["Musée national suisse", "MNS", "Schweizerisches Nationalmuseum", "SNM"]),
    ("Pro Helvetia", ["Fondation suisse pour la culture Pro Helvetia",
                      "Schweizerische Kulturstiftung Pro Helvetia", "Pro Helvetia"]),
]


# ---------------------------------------------------------------------------
# 4) Federal councillors — canonical = the person's own name.
#    Council compositions: (start, end, {dept_code: councillor_name}),
#    copied from the legacy src/run5_prompts.py.
# ---------------------------------------------------------------------------
COUNCILLORS: list[str] = [
    "Joseph Deiss", "Ruth Dreifuss", "Ruth Metzler-Arnold", "Adolf Ogi",
    "Kaspar Villiger", "Pascal Couchepin", "Moritz Leuenberger", "Samuel Schmid",
    "Micheline Calmy-Rey", "Christoph Blocher", "Hans-Rudolf Merz", "Doris Leuthard",
    "Eveline Widmer-Schlumpf", "Ueli Maurer", "Didier Burkhalter", "Simonetta Sommaruga",
    "Johann Schneider-Ammann", "Alain Berset", "Guy Parmelin", "Ignazio Cassis",
    "Karin Keller-Sutter", "Viola Amherd", "Elisabeth Baume-Schneider", "Albert Rösti",
    "Beat Jans", "Martin Pfister",
]

_COUNCIL_COMPOSITIONS: list[tuple[date, date, dict[str, str]]] = [
    (date(2000, 1, 1), date(2000, 12, 31), {
        "DFAE/EDA": "Joseph Deiss", "DFI/EDI": "Ruth Dreifuss", "DFJP/EJPD": "Ruth Metzler-Arnold",
        "DDPS/VBS": "Adolf Ogi", "DFF/EFD": "Kaspar Villiger", "DFE/EVD": "Pascal Couchepin",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2001, 1, 1), date(2002, 12, 31), {
        "DFAE/EDA": "Joseph Deiss", "DFI/EDI": "Ruth Dreifuss", "DFJP/EJPD": "Ruth Metzler-Arnold",
        "DDPS/VBS": "Samuel Schmid", "DFF/EFD": "Kaspar Villiger", "DFE/EVD": "Pascal Couchepin",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2003, 1, 1), date(2003, 12, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Pascal Couchepin", "DFJP/EJPD": "Ruth Metzler-Arnold",
        "DDPS/VBS": "Samuel Schmid", "DFF/EFD": "Kaspar Villiger", "DFE/EVD": "Joseph Deiss",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2004, 1, 1), date(2006, 7, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Pascal Couchepin", "DFJP/EJPD": "Christoph Blocher",
        "DDPS/VBS": "Samuel Schmid", "DFF/EFD": "Hans-Rudolf Merz", "DFE/EVD": "Joseph Deiss",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2006, 8, 1), date(2007, 12, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Pascal Couchepin", "DFJP/EJPD": "Christoph Blocher",
        "DDPS/VBS": "Samuel Schmid", "DFF/EFD": "Hans-Rudolf Merz", "DFE/EVD": "Doris Leuthard",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2008, 1, 1), date(2008, 12, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Pascal Couchepin", "DFJP/EJPD": "Eveline Widmer-Schlumpf",
        "DDPS/VBS": "Samuel Schmid", "DFF/EFD": "Hans-Rudolf Merz", "DFE/EVD": "Doris Leuthard",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2009, 1, 1), date(2009, 10, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Pascal Couchepin", "DFJP/EJPD": "Eveline Widmer-Schlumpf",
        "DDPS/VBS": "Ueli Maurer", "DFF/EFD": "Hans-Rudolf Merz", "DFE/EVD": "Doris Leuthard",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2009, 11, 1), date(2010, 10, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Didier Burkhalter", "DFJP/EJPD": "Eveline Widmer-Schlumpf",
        "DDPS/VBS": "Ueli Maurer", "DFF/EFD": "Hans-Rudolf Merz", "DFE/EVD": "Doris Leuthard",
        "DETEC/UVEK": "Moritz Leuenberger",
    }),
    (date(2010, 11, 1), date(2011, 12, 31), {
        "DFAE/EDA": "Micheline Calmy-Rey", "DFI/EDI": "Didier Burkhalter", "DFJP/EJPD": "Simonetta Sommaruga",
        "DDPS/VBS": "Ueli Maurer", "DFF/EFD": "Eveline Widmer-Schlumpf", "DEFR/WBF": "Johann Schneider-Ammann",
        "DETEC/UVEK": "Doris Leuthard",
    }),
    (date(2012, 1, 1), date(2015, 12, 31), {
        "DFAE/EDA": "Didier Burkhalter", "DFI/EDI": "Alain Berset", "DFJP/EJPD": "Simonetta Sommaruga",
        "DDPS/VBS": "Ueli Maurer", "DFF/EFD": "Eveline Widmer-Schlumpf", "DEFR/WBF": "Johann Schneider-Ammann",
        "DETEC/UVEK": "Doris Leuthard",
    }),
    (date(2016, 1, 1), date(2017, 10, 31), {
        "DFAE/EDA": "Didier Burkhalter", "DFI/EDI": "Alain Berset", "DFJP/EJPD": "Simonetta Sommaruga",
        "DDPS/VBS": "Guy Parmelin", "DFF/EFD": "Ueli Maurer", "DEFR/WBF": "Johann Schneider-Ammann",
        "DETEC/UVEK": "Doris Leuthard",
    }),
    (date(2017, 11, 1), date(2018, 12, 31), {
        "DFAE/EDA": "Ignazio Cassis", "DFI/EDI": "Alain Berset", "DFJP/EJPD": "Simonetta Sommaruga",
        "DDPS/VBS": "Guy Parmelin", "DFF/EFD": "Ueli Maurer", "DEFR/WBF": "Johann Schneider-Ammann",
        "DETEC/UVEK": "Doris Leuthard",
    }),
    (date(2019, 1, 1), date(2022, 12, 31), {
        "DFAE/EDA": "Ignazio Cassis", "DFI/EDI": "Alain Berset", "DFJP/EJPD": "Karin Keller-Sutter",
        "DDPS/VBS": "Viola Amherd", "DFF/EFD": "Ueli Maurer", "DEFR/WBF": "Guy Parmelin",
        "DETEC/UVEK": "Simonetta Sommaruga",
    }),
    (date(2023, 1, 1), date(2023, 12, 31), {
        "DFAE/EDA": "Ignazio Cassis", "DFI/EDI": "Alain Berset", "DFJP/EJPD": "Elisabeth Baume-Schneider",
        "DDPS/VBS": "Viola Amherd", "DFF/EFD": "Karin Keller-Sutter", "DEFR/WBF": "Guy Parmelin",
        "DETEC/UVEK": "Albert Rösti",
    }),
    (date(2024, 1, 1), date(2025, 3, 31), {
        "DFAE/EDA": "Ignazio Cassis", "DFI/EDI": "Elisabeth Baume-Schneider", "DFJP/EJPD": "Beat Jans",
        "DDPS/VBS": "Viola Amherd", "DFF/EFD": "Karin Keller-Sutter", "DEFR/WBF": "Guy Parmelin",
        "DETEC/UVEK": "Albert Rösti",
    }),
    (date(2025, 4, 1), date(9999, 12, 31), {
        "DFAE/EDA": "Ignazio Cassis", "DFI/EDI": "Elisabeth Baume-Schneider", "DFJP/EJPD": "Beat Jans",
        "DDPS/VBS": "Martin Pfister", "DFF/EFD": "Karin Keller-Sutter", "DEFR/WBF": "Guy Parmelin",
        "DETEC/UVEK": "Albert Rösti",
    }),
]

# Pre-2010 "DFE/EVD" is the same department later renamed "DEFR/WBF".
_DEPT_KEY_ALIASES: dict[str, str] = {"DFE/EVD": "DEFR/WBF"}


def _coerce_date(val) -> Optional[date]:
    """Coerce any pubtime value (str, Timestamp, datetime, date, NaN) to date."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, pd.Timestamp):
        return val.date()
    try:
        return pd.to_datetime(str(val)).date()
    except Exception:
        return None


def get_council_for_date(pubtime) -> dict[str, str]:
    """Return {dept_code: councillor_name} in office at *pubtime* (canonical dept codes)."""
    d = _coerce_date(pubtime)
    comp = _COUNCIL_COMPOSITIONS[-1][2]
    if d is not None:
        for start, end, c in _COUNCIL_COMPOSITIONS:
            if start <= d <= end:
                comp = c
                break
        else:
            comp = _COUNCIL_COMPOSITIONS[0][2] if d < _COUNCIL_COMPOSITIONS[0][0] else _COUNCIL_COMPOSITIONS[-1][2]
    return {_DEPT_KEY_ALIASES.get(dept, dept): name for dept, name in comp.items()}


_COUNCILLOR_TERMS: dict[str, tuple[date, date]] = {}


def _build_councillor_terms() -> dict[str, tuple[date, date]]:
    """For each councillor, the (earliest start, latest end) across every
    composition period in which they headed any department. Federal
    councillors in this dataset never leave office and later return, so a
    single min/max span correctly captures continuous tenure even when they
    switch departments mid-term."""
    terms: dict[str, tuple[date, date]] = {}
    for start, end, comp in _COUNCIL_COMPOSITIONS:
        for name in comp.values():
            if name not in terms:
                terms[name] = (start, end)
            else:
                lo, hi = terms[name]
                terms[name] = (min(lo, start), max(hi, end))
    return terms


_COUNCILLOR_TERMS = _build_councillor_terms()


def get_councillor_term(name: str) -> Optional[tuple[date, date]]:
    """Return (start_date, end_date) the councillor was in office, or None if unknown."""
    return _COUNCILLOR_TERMS.get(name)


def is_councillor_in_office(name: str, pubtime) -> bool:
    """True if *name* was a sitting federal councillor on *pubtime*."""
    term = get_councillor_term(name)
    d = _coerce_date(pubtime)
    if term is None or d is None:
        return False
    start, end = term
    return start <= d <= end


# ---------------------------------------------------------------------------
# Language tagging for every alias (identical rules to the legacy
# scripts/tag_keywords.py): explicit overrides first, then an accent-based
# heuristic. "both" means the alias is checked regardless of article language.
# ---------------------------------------------------------------------------
_FR_ACCENT_CHARS = frozenset("éèêëàâùûôœçîïú")
_DE_UMLAUT_CHARS = frozenset("äöüÄÖÜß")

_TERM_LANG: dict[str, str] = {
    "VBS": "de", "DDPS": "fr", "EDA": "de", "DFAE": "fr", "UVEK": "de", "DETEC": "fr",
    "EJPD": "de", "DFJP": "fr", "EDI": "de", "DFI": "fr", "EFD": "de", "DFF": "fr",
    "WBF": "de", "DEFR": "fr",
    "GS-EDA": "de", "SG-DFAE": "fr", "DDIP": "fr", "DEZA": "de", "DDC": "fr",
    "GS-EDI": "de", "SG-DFI": "fr", "EBG": "de", "BFEG": "fr", "BAK": "de", "OFC": "fr",
    "AFS": "fr", "MeteoSchweiz": "de", "BAG": "de", "OFSP": "fr", "BLV": "de", "OSAV": "fr",
    "BFS": "de", "OFS": "fr", "BSV": "de", "OFAS": "fr",
    "GS-EJPD": "de", "SG-DFJP": "fr", "SEM": "both", "OFJ": "fr", "fedpol": "both",
    "Service SCPT": "fr",
    "GS-VBS": "de", "SG-DDPS": "fr", "BABS": "de", "OFPP": "fr", "armasuisse": "both",
    "swisstopo": "both", "BASPO": "de", "OFSPO": "fr", "SEPOS": "fr", "NDB": "de", "SRC": "fr",
    "OAC": "fr", "Schweizer Armee": "de", "BACS": "de", "OFCS": "fr",
    "GS-EFD": "de", "SG-DFF": "fr", "SIF": "de", "SFI": "fr", "EFV": "de", "AFF": "fr",
    "EPA": "de", "OFPER": "fr", "ESTV": "de", "AFC": "fr", "BAZG": "de", "OFDF": "fr",
    "OFIT": "fr", "BBL": "de", "OFCL": "fr",
    "GS-WBF": "de", "SG-DEFR": "fr", "SECO": "both", "SBFI": "de", "SEFRI": "fr",
    "BLW": "de", "OFAG": "fr", "OFAE": "fr", "BWO": "de", "OFL": "fr", "ZIVI": "de", "CIVI": "fr",
    "GS-UVEK": "de", "SG-DETEC": "fr", "BAV": "de", "OFT": "fr", "BAZL": "de", "OFAC": "fr",
    "BFE": "de", "OFEN": "fr", "ASTRA": "de", "OFROU": "fr", "BAKOM": "de", "OFCOM": "fr",
    "BAFU": "de", "OFEV": "fr", "ARE": "both",
    "ENSI": "de", "IFSN": "fr", "ESTI": "de", "SUST": "de", "SESE": "fr", "ElCom": "de",
    "EICom": "fr", "ComCom": "both", "UBI": "de", "AIEP": "fr", "PostCom": "both",
    "RailCom": "both", "PUE": "de", "SPR": "fr", "WEKO": "de", "COMCO": "fr",
    "ETH-Bereich": "de", "domaine des EPF": "fr", "EHB": "de", "HEFP": "fr",
    "Innosuisse": "both", "FINMA": "both", "EFK": "de", "CDF": "fr", "PUBLICA": "both",
    "IGE": "de", "IPI": "fr", "METAS": "both", "SIR": "de", "ISDC": "fr", "RAB": "de",
    "ASR": "fr", "ESBK": "de", "CFMJ": "fr", "ESchK": "de", "CAF": "fr", "NKVF": "de",
    "CNPT": "fr", "EKM": "de", "CFM": "fr", "Swissmedic": "both", "SNM": "de", "MNS": "fr",
    "Pro Helvetia": "both",
}


def _alias_language(alias: str) -> str:
    if alias in _TERM_LANG:
        return _TERM_LANG[alias]
    chars = set(alias)
    has_fr = bool(chars & _FR_ACCENT_CHARS)
    has_de = bool(chars & _DE_UMLAUT_CHARS)
    if has_fr and not has_de:
        return "fr"
    if has_de and not has_fr:
        return "de"
    return "both"


# ---------------------------------------------------------------------------
# Flatten everything into one matching table:
#   canonical -> (target_type, parent_dept, [(alias, pattern, lang), ...])
# ---------------------------------------------------------------------------
class _TargetEntry:
    __slots__ = ("canonical", "target_type", "parent_dept", "patterns")

    def __init__(self, canonical: str, target_type: str, parent_dept: Optional[str]):
        self.canonical = canonical
        self.target_type = target_type
        self.parent_dept = parent_dept
        self.patterns: list[tuple[str, re.Pattern, str]] = []  # (alias text, compiled pattern, lang)

    def add_alias(self, alias: str) -> None:
        pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        self.patterns.append((alias, pat, _alias_language(alias)))


_TARGET_ENTRIES: dict[str, _TargetEntry] = {}


def _register(canonical: str, target_type: str, parent_dept: Optional[str], aliases: list[str]) -> None:
    entry = _TargetEntry(canonical, target_type, parent_dept)
    for alias in aliases:
        entry.add_alias(alias)
    _TARGET_ENTRIES[canonical] = entry


def _build_registry() -> None:
    _TARGET_ENTRIES.clear()
    for alias, dept_code in DEPARTMENT_ALIASES.items():
        if dept_code not in _TARGET_ENTRIES:
            _register(dept_code, FEDERAL_DEPARTMENT, dept_code, [])
        _TARGET_ENTRIES[dept_code].add_alias(alias)

    for canonical, parent_dept, aliases in ADMIN_UNIT_GROUPS:
        _register(canonical, ADMINISTRATIVE_UNIT, parent_dept, aliases)

    for canonical, aliases in INDEPENDENT_AGENCY_GROUPS:
        _register(canonical, INDEPENDENT_AGENCY, None, aliases)

    for name in COUNCILLORS:
        _register(name, FEDERAL_COUNCILLOR, None, [name])


_build_registry()


def classify_target(canonical: str) -> dict:
    """Return {'target_type', 'parent_dept'} for a canonical target name."""
    entry = _TARGET_ENTRIES.get(canonical)
    if entry is None:
        return {"target_type": "Unknown", "parent_dept": None}
    return {"target_type": entry.target_type, "parent_dept": entry.parent_dept}


def match_targets_detailed(text: str, language: str) -> list[tuple[str, str]]:
    """Return [(canonical, matched_alias), ...] for every target mentioned in
    *text*, restricted to aliases valid for the article's *language*
    ('de'/'fr'). Each canonical target appears at most once. *matched_alias*
    is the literal alias actually found in the text — when several aliases
    of the same target matched (e.g. both "EDA" and its German full name),
    the longest (most spelled-out) one is reported, matching the legacy
    pipeline's own preference for the full name over the abbreviation."""
    if not isinstance(text, str) or not text:
        return []
    found: list[tuple[str, str]] = []
    for canonical, entry in _TARGET_ENTRIES.items():
        matched_aliases = [
            alias for alias, pat, lang in entry.patterns
            if lang in ("both", language) and pat.search(text)
        ]
        if matched_aliases:
            found.append((canonical, max(matched_aliases, key=len)))
    return found


def match_targets(text: str, language: str) -> list[str]:
    """Return just the canonical target names mentioned in *text* — see
    match_targets_detailed() for the (canonical, matched_alias) pairs."""
    return [canonical for canonical, _alias in match_targets_detailed(text, language)]


# ---------------------------------------------------------------------------
# Self-check: every alias declared in this module's groups must be unique
# (no alias assigned to two different canonical targets) and, combined,
# they must cover exactly what the legacy download_src.py lists contained.
# Run with: python -m src.target_taxonomy
# ---------------------------------------------------------------------------
def self_check(legacy_departments=None, legacy_admin_units=None, legacy_independent_agencies=None) -> None:
    # Rebuild alias->canonical directly from the source lists (rather than
    # trying to recover literal aliases from compiled regex patterns) and
    # check for internal consistency (no alias assigned twice) plus, when
    # the legacy lists are passed in, full coverage against them.
    alias_to_canonical: dict[str, str] = {}

    def add(alias: str, canonical: str):
        if alias in alias_to_canonical and alias_to_canonical[alias] != canonical:
            raise AssertionError(
                f"Alias {alias!r} assigned to both {alias_to_canonical[alias]!r} and {canonical!r}"
            )
        alias_to_canonical[alias] = canonical

    for alias, dept in DEPARTMENT_ALIASES.items():
        add(alias, dept)
    for canonical, _parent, aliases in ADMIN_UNIT_GROUPS:
        for a in aliases:
            add(a, canonical)
    for canonical, aliases in INDEPENDENT_AGENCY_GROUPS:
        for a in aliases:
            add(a, canonical)

    if legacy_departments is not None:
        missing = set(legacy_departments) - alias_to_canonical.keys()
        extra = {a for a in alias_to_canonical if alias_to_canonical[a] in DEPARTMENT_CODES} - set(legacy_departments)
        assert not missing, f"DEPARTMENTS aliases missing from taxonomy: {missing}"
        assert not extra, f"Taxonomy has department aliases not in legacy DEPARTMENTS: {extra}"

    if legacy_admin_units is not None:
        mine = {a for canonical, _p, aliases in ADMIN_UNIT_GROUPS for a in aliases}
        missing = set(legacy_admin_units) - mine
        extra = mine - set(legacy_admin_units)
        assert not missing, f"ADMIN_UNITS aliases missing from taxonomy: {missing}"
        assert not extra, f"Taxonomy has admin-unit aliases not in legacy ADMIN_UNITS: {extra}"

    if legacy_independent_agencies is not None:
        mine = {a for canonical, aliases in INDEPENDENT_AGENCY_GROUPS for a in aliases}
        missing = set(legacy_independent_agencies) - mine
        extra = mine - set(legacy_independent_agencies)
        assert not missing, f"INDEPENDENT_AGENCIES aliases missing from taxonomy: {missing}"
        assert not extra, f"Taxonomy has agency aliases not in legacy INDEPENDENT_AGENCIES: {extra}"

    print(f"[self_check] OK — {len(alias_to_canonical)} aliases, "
          f"{len(DEPARTMENT_CODES)} departments, {len(ADMIN_UNIT_GROUPS)} admin units, "
          f"{len(INDEPENDENT_AGENCY_GROUPS)} independent agencies, {len(COUNCILLORS)} councillors.")


if __name__ == "__main__":
    from src.download_src import DEPARTMENTS, ADMIN_UNITS, INDEPENDENT_AGENCIES
    self_check(
        legacy_departments=DEPARTMENTS,
        legacy_admin_units=ADMIN_UNITS,
        legacy_independent_agencies=INDEPENDENT_AGENCIES,
    )
