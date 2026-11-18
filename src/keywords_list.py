# keywords_list.py
"""
LISTE DES MOTS-CLÉS DE LA REQUÊTE SWISSDOX — source unique utilisée par le download.

  - FEDERAL_UNITS        : départements + unités fédérales (actuels et OLD),
                           copiés de SWISSDOX_REPO/src/keywords_list.py (nov. 2026),
                           SANS les mots-clés génériques (inutiles pour accountability).
  - INDEPENDENT_AGENCIES : agences indépendantes (actuelles et OLD), d'après
                           « Liste des départements fédéraux.csv », mêmes règles.
  - COUNCILLORS          : conseillers fédéraux 2000 – aujourd'hui.

Tout est écrit en toutes lettres, déclinaisons allemandes comprises : ce qui
figure ici est exactement ce qui est envoyé à Swissdox. Pour ajouter / retirer
un mot-clé, modifier ce fichier.

Swissdox respecte les majuscules et les accents : chaque graphie doit donc être
écrite ici (ex. "Office fédéral de l'énergie" ET "office fédéral de l'énergie").

Seule transformation automatique : chaque terme contenant une apostrophe est
aussi cherché avec l'autre apostrophe (' <-> ’).

Origine : « Liste des départements fédéraux.csv » (oct. 2026) + correction des
terminaisons allemandes par une collègue germanophone (nov. 2026). Le commentaire
au-dessus de chaque nom allemand rappelle la notation d'origine (ex. "Bundesamt.s.es").
Règle d'accord appliquée : le génitif du nom (-s/-es) n'apparaît qu'avec la forme
en -n de l'adjectif ("des Eidgenössischen Departements").
"""

# ════════════════════════════════════════════════════════════════════════════
# MOTS-CLÉS SPÉCIFIQUES : UNITÉS FÉDÉRALES (une entrée = une unité)
#   type / upper_level / en_* : métadonnées pour l'analyse (EN jamais cherché)
#   fr_names    : noms français ; chaque sous-liste = un nom avec TOUTES ses
#                 variantes (majuscule / minuscule initiale ; État / Etat / état),
#                 la 1re forme étant la forme de référence
#   de_names    : noms allemands ; chaque sous-liste = un nom avec TOUTES ses
#                 déclinaisons, la 1re forme étant la forme de référence
#   *_abbrev    : abréviation principale (sert de keyword_abbrev)
#   *_alt_abbrevs : abréviations alternatives
#   abbrev_case_variants : autres graphies d'une abréviation, rattachées à elle
#                 (ex. 'SECO' -> ['Seco'] ; 'Fedpol' -> ['fedpol', 'FEDPOL']).
#                 Règle : abréviation en majuscules de 4 lettres ou plus -> forme
#                 avec seule l'initiale en majuscule, sauf OFEN (« Ofen » = four),
#                 ZIVI (« Zivi » = un civiliste) et OFCOM (« Ofcom » = régulateur
#                 britannique). Les abréviations de 2-3 lettres restent en majuscules.
# ════════════════════════════════════════════════════════════════════════════
FEDERAL_UNITS = [
    # ──────────────── FDFA ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDFA',
        "en_name": 'Federal Department of Foreign Affairs', "en_abbrev": 'FDFA',
        "fr_names": [
            ['Département fédéral des affaires étrangères', 'département fédéral des affaires étrangères'],
        ],
        "fr_abbrev": 'DFAE', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s für auswärtige Angelegenheiten
            [
                'Eidgenössische Departement für auswärtige Angelegenheiten',
                'Eidgenössisches Departement für auswärtige Angelegenheiten',
                'Eidgenössischen Departement für auswärtige Angelegenheiten',
                'Eidgenössischen Departements für auswärtige Angelegenheiten',
            ],
        ],
        "de_abbrev": 'EDA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DFAE': ['Dfae']},
    },
    # ──────────────── FDHA ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDHA',
        "en_name": 'Federal Department of Home Affairs', "en_abbrev": 'FDHA',
        "fr_names": [
            ["Département fédéral de l'intérieur", "département fédéral de l'intérieur"],
        ],
        "fr_abbrev": 'DFI', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s des Innern
            [
                'Eidgenössische Departement des Innern',
                'Eidgenössisches Departement des Innern',
                'Eidgenössischen Departement des Innern',
                'Eidgenössischen Departements des Innern',
            ],
        ],
        "de_abbrev": 'EDI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── FDJP ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDJP',
        "en_name": 'Federal Department of Justice and Police', "en_abbrev": 'FDJP',
        "fr_names": [
            ['Département fédéral de justice et police', 'département fédéral de justice et police'],
        ],
        "fr_abbrev": 'DFJP', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Justiz- und Polizeidepartement.s
            [
                'Eidgenössische Justiz- und Polizeidepartement',
                'Eidgenössisches Justiz- und Polizeidepartement',
                'Eidgenössischen Justiz- und Polizeidepartement',
                'Eidgenössischen Justiz- und Polizeidepartements',
            ],
        ],
        "de_abbrev": 'EJPD', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DFJP': ['Dfjp'], 'EJPD': ['Ejpd']},
    },
    # ──────────────── DDPS ────────────────
    {
        "type": 'Federal Department', "upper_level": 'DDPS',
        "en_name": 'Federal Department of Defence, Civil Protection and Sport', "en_abbrev": 'DDPS',
        "fr_names": [
            ['Département fédéral de la défense, de la protection de la population et des sports', 'département fédéral de la défense, de la protection de la population et des sports'],
        ],
        "fr_abbrev": 'DDPS', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s für Verteidigung, Bevölkerungsschutz und Sport
            [
                'Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport',
                'Eidgenössisches Departement für Verteidigung, Bevölkerungsschutz und Sport',
                'Eidgenössischen Departement für Verteidigung, Bevölkerungsschutz und Sport',
                'Eidgenössischen Departements für Verteidigung, Bevölkerungsschutz und Sport',
            ],
        ],
        "de_abbrev": 'VBS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DDPS': ['Ddps']},
    },
    # ──────────────── FDF ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDF',
        "en_name": 'Federal Department of Finance', "en_abbrev": 'FDF',
        "fr_names": [
            ['Département fédéral des finances', 'département fédéral des finances'],
        ],
        "fr_abbrev": 'DFF', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Finanzdepartement.s
            [
                'Eidgenössische Finanzdepartement',
                'Eidgenössisches Finanzdepartement',
                'Eidgenössischen Finanzdepartement',
                'Eidgenössischen Finanzdepartements',
            ],
        ],
        "de_abbrev": 'EFD', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── EAER ────────────────
    {
        "type": 'Federal Department', "upper_level": 'EAER',
        "en_name": 'Federal Department of Economic Affairs, Education and Research', "en_abbrev": 'EAER',
        "fr_names": [
            ["Département fédéral de l'économie, de la formation et de la recherche", "département fédéral de l'économie, de la formation et de la recherche"],
            ['Département fédéral de l’économie', 'département fédéral de l’économie'],
        ],
        "fr_abbrev": 'DEFR', "fr_alt_abbrevs": ['DFE'],
        "de_names": [
            # Eidgenössische.s.n Departement.s für Wirtschaft, Bildung und Forschung
            [
                'Eidgenössische Departement für Wirtschaft, Bildung und Forschung',
                'Eidgenössisches Departement für Wirtschaft, Bildung und Forschung',
                'Eidgenössischen Departement für Wirtschaft, Bildung und Forschung',
                'Eidgenössischen Departements für Wirtschaft, Bildung und Forschung',
            ],
            # Eidgenössische.s.n Volkswirtschaftsdepartement.s
            [
                'Eidgenössische Volkswirtschaftsdepartement',
                'Eidgenössisches Volkswirtschaftsdepartement',
                'Eidgenössischen Volkswirtschaftsdepartement',
                'Eidgenössischen Volkswirtschaftsdepartements',
            ],
        ],
        "de_abbrev": 'WBF', "de_alt_abbrevs": ['EVD'],
        "abbrev_case_variants": {'DEFR': ['Defr']},
    },
    # ──────────────── DETEC ────────────────
    {
        "type": 'Federal Department', "upper_level": 'DETEC',
        "en_name": 'Federal Department of the Environment, Transport, Energy and Communications', "en_abbrev": 'DETEC',
        "fr_names": [
            ["Département fédéral de l'environnement, des transports, de l'énergie et de la communication", "département fédéral de l'environnement, des transports, de l'énergie et de la communication"],
        ],
        "fr_abbrev": 'DETEC', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s für Umwelt, Verkehr, Energie und Kommunikation
            [
                'Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation',
                'Eidgenössisches Departement für Umwelt, Verkehr, Energie und Kommunikation',
                'Eidgenössischen Departement für Umwelt, Verkehr, Energie und Kommunikation',
                'Eidgenössischen Departements für Umwelt, Verkehr, Energie und Kommunikation',
            ],
        ],
        "de_abbrev": 'UVEK', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DETEC': ['Detec'], 'UVEK': ['Uvek']},
    },
    # ──────────────── FDFA ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDFA',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DFAE', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EDA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'FDFA State Secretariat', "en_abbrev": 'FDFA-STS',
        "fr_names": [
            ['Secrétariat d’Etat du DFAE', 'secrétariat d’Etat du DFAE', 'Secrétariat d’État du DFAE', 'secrétariat d’État du DFAE', 'Secrétariat d’état du DFAE', 'secrétariat d’état du DFAE'],
            ['Secrétariat d’Etat', 'secrétariat d’Etat', 'Secrétariat d’État', 'secrétariat d’État', 'Secrétariat d’état', 'secrétariat d’état'],
        ],
        "fr_abbrev": 'SEE-DFAE', "fr_alt_abbrevs": ['SEE'],
        "de_names": [
            # Staatssekretariat.s EDA
            [
                'Staatssekretariat EDA',
                'Staatssekretariats EDA',
            ],
            # Staatssekretariat.s
            [
                'Staatssekretariat',
                'Staatssekretariats',
            ],
        ],
        "de_abbrev": 'STS-EDA', "de_alt_abbrevs": ['STS'],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Directorate of International Law', "en_abbrev": 'DIL',
        "fr_names": [
            ['Direction du droit international public', 'direction du droit international public'],
        ],
        "fr_abbrev": 'DDIP', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für Völkerrecht'],
        ],
        "de_abbrev": 'DV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DDIP': ['Ddip']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Swiss Agency for Development and Cooperation', "en_abbrev": 'SDC',
        "fr_names": [
            ['Direction du développement et de la coopération', 'direction du développement et de la coopération'],
        ],
        "fr_abbrev": 'DDC', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für Entwicklung und Zusammenarbeit'],
        ],
        "de_abbrev": 'DEZA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DEZA': ['Deza']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Directorate for Resources', "en_abbrev": 'DR',
        "fr_names": [
            ['Direction des ressources', 'direction des ressources'],
            ['Direction des ressources et du réseau extérieur', 'direction des ressources et du réseau extérieur'],
        ],
        "fr_abbrev": 'DR', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für Ressourcen'],
            ['Direktion für Ressourcen und Aussennetz'],
        ],
        "de_abbrev": 'DR', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Consular Directorate', "en_abbrev": 'CD',
        "fr_names": [
            ['Direction consulaire', 'direction consulaire'],
        ],
        "fr_abbrev": 'DC', "fr_alt_abbrevs": [],
        "de_names": [
            # Konsularische.n Direktion
            [
                'Konsularische Direktion',
                'Konsularischen Direktion',
            ],
        ],
        "de_abbrev": 'KD', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDFA',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Direction politique', 'direction politique'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Politische.n Direktion
            [
                'Politische Direktion',
                'Politischen Direktion',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDFA',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Direction des affaires européennes', 'direction des affaires européennes'],
        ],
        "fr_abbrev": 'DAE', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für europäische Angelegenheiten'],
        ],
        "de_abbrev": 'DEA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── FDHA ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDHA',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DFI', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EDI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office for Gender Equality', "en_abbrev": 'FOGE',
        "fr_names": [
            ['Bureau fédéral de l’égalité entre femmes et hommes', 'bureau fédéral de l’égalité entre femmes et hommes'],
        ],
        "fr_abbrev": 'BFEG', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Büro.s für die Gleichstellung von Frau und Mann
            [
                'Eidgenössische Büro für die Gleichstellung von Frau und Mann',
                'Eidgenössisches Büro für die Gleichstellung von Frau und Mann',
                'Eidgenössischen Büro für die Gleichstellung von Frau und Mann',
                'Eidgenössischen Büros für die Gleichstellung von Frau und Mann',
            ],
        ],
        "de_abbrev": 'EBG', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'BFEG': ['Bfeg']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office of Culture', "en_abbrev": 'FOC',
        "fr_names": [
            ['Office fédéral de la culture', 'office fédéral de la culture'],
        ],
        "fr_abbrev": 'OFC', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Kultur
            [
                'Bundesamt für Kultur',
                'Bundesamts für Kultur',
                'Bundesamtes für Kultur',
            ],
        ],
        "de_abbrev": 'BAK', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Archives', "en_abbrev": 'SFA',
        "fr_names": [
            ['Archives fédérales suisses', 'archives fédérales suisses'],
        ],
        "fr_abbrev": 'AFS', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.s.n Bundesarchiv.s
            [
                'Schweizerische Bundesarchiv',
                'Schweizerisches Bundesarchiv',
                'Schweizerischen Bundesarchiv',
                'Schweizerischen Bundesarchivs',
            ],
        ],
        "de_abbrev": 'BAR', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office of Meteorology and Climatology', "en_abbrev": 'MeteoSwiss',
        "fr_names": [
            ['Office fédéral de météorologie et de climatologie', 'office fédéral de météorologie et de climatologie'],
        ],
        "fr_abbrev": 'MétéoSuisse', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Meteorologie und Klimatologie
            [
                'Bundesamt für Meteorologie und Klimatologie',
                'Bundesamts für Meteorologie und Klimatologie',
                'Bundesamtes für Meteorologie und Klimatologie',
            ],
        ],
        "de_abbrev": 'MeteoSchweiz', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'MétéoSuisse': ['MÉTÉOSUISSE', 'Météosuisse', 'Météo-Suisse', 'MeteoSuisse'], 'MeteoSchweiz': ['METEOSCHWEIZ', 'Meteoschweiz', 'Meteo-Schweiz']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office of Public Health', "en_abbrev": 'FOPH',
        "fr_names": [
            ['Office fédéral de la santé publique', 'office fédéral de la santé publique'],
        ],
        "fr_abbrev": 'OFSP', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Gesundheit
            [
                'Bundesamt für Gesundheit',
                'Bundesamts für Gesundheit',
                'Bundesamtes für Gesundheit',
            ],
        ],
        "de_abbrev": 'BAG', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFSP': ['Ofsp']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Statistical Office', "en_abbrev": 'FSO',
        "fr_names": [
            ['Office fédéral de la statistique', 'office fédéral de la statistique'],
        ],
        "fr_abbrev": 'OFS', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Statistik
            [
                'Bundesamt für Statistik',
                'Bundesamts für Statistik',
                'Bundesamtes für Statistik',
            ],
        ],
        "de_abbrev": 'BFS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Social Insurance Office', "en_abbrev": 'FSIO',
        "fr_names": [
            ['Office fédéral des assurances sociales', 'office fédéral des assurances sociales'],
        ],
        "fr_abbrev": 'OFAS', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Sozialversicherungen
            [
                'Bundesamt für Sozialversicherungen',
                'Bundesamts für Sozialversicherungen',
                'Bundesamtes für Sozialversicherungen',
            ],
        ],
        "de_abbrev": 'BSV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFAS': ['Ofas']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Food Safety and Veterinary Office', "en_abbrev": 'FSVO',
        "fr_names": [
            ['Office fédéral de la sécurité alimentaire et des affaires vétérinaires', 'office fédéral de la sécurité alimentaire et des affaires vétérinaires'],
        ],
        "fr_abbrev": 'OSAV', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Lebensmittelsicherheit und Veterinärwesen
            [
                'Bundesamt für Lebensmittelsicherheit und Veterinärwesen',
                'Bundesamts für Lebensmittelsicherheit und Veterinärwesen',
                'Bundesamtes für Lebensmittelsicherheit und Veterinärwesen',
            ],
        ],
        "de_abbrev": 'BLV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OSAV': ['Osav']},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDHA',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Office fédéral de l’assurance militaire', 'office fédéral de l’assurance militaire'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Militärversicherung
            [
                'Bundesamt für Militärversicherung',
                'Bundesamts für Militärversicherung',
                'Bundesamtes für Militärversicherung',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDHA',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Groupement de la science et de la recherche', 'groupement de la science et de la recherche'],
            ['Secrétariat d’Etat à l’éducation et à la recherche', 'secrétariat d’Etat à l’éducation et à la recherche', 'Secrétariat d’État à l’éducation et à la recherche', 'secrétariat d’État à l’éducation et à la recherche', 'Secrétariat d’état à l’éducation et à la recherche', 'secrétariat d’état à l’éducation et à la recherche'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Gruppe für Wissenschaft und Forschung'],
            # Staatssekretariat.s für Bildung und Forschung
            [
                'Staatssekretariat für Bildung und Forschung',
                'Staatssekretariats für Bildung und Forschung',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── FDJP ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDJP',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DFJP', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EJPD', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'Federal office of Justice', "en_abbrev": 'FOJ',
        "fr_names": [
            ['Office fédéral de la justice', 'office fédéral de la justice'],
        ],
        "fr_abbrev": 'OFJ', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Justiz
            [
                'Bundesamt für Justiz',
                'Bundesamts für Justiz',
                'Bundesamtes für Justiz',
            ],
        ],
        "de_abbrev": 'BJ', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'Federal Office of Police', "en_abbrev": 'Fedpol',
        "fr_names": [
            ['Office fédéral de la police', 'office fédéral de la police'],
        ],
        "fr_abbrev": 'Fedpol', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Polizei
            [
                'Bundesamt für Polizei',
                'Bundesamts für Polizei',
                'Bundesamtes für Polizei',
            ],
        ],
        "de_abbrev": 'Fedpol', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'Fedpol': ['fedpol', 'FEDPOL', 'FedPol']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'State Secretariat for  Migration', "en_abbrev": 'SEM',
        "fr_names": [
            ["Secrétariat d'état aux migrations", "secrétariat d'état aux migrations", "Secrétariat d'État aux migrations", "secrétariat d'État aux migrations", "Secrétariat d'Etat aux migrations", "secrétariat d'Etat aux migrations"],
            ['Office fédéral des étrangers', 'office fédéral des étrangers'],
            ['Office fédéral des réfugiés', 'office fédéral des réfugiés'],
            ['Office fédéral des migrations', 'office fédéral des migrations'],
        ],
        "fr_abbrev": 'SEM', "fr_alt_abbrevs": ['ODM'],
        "de_names": [
            # Staatssekretariat.s für Migration
            [
                'Staatssekretariat für Migration',
                'Staatssekretariats für Migration',
            ],
            # Bundesamt.s.es für Ausländerfragen
            [
                'Bundesamt für Ausländerfragen',
                'Bundesamts für Ausländerfragen',
                'Bundesamtes für Ausländerfragen',
            ],
            # Bundesamt.s.es für Flüchtlinge
            [
                'Bundesamt für Flüchtlinge',
                'Bundesamts für Flüchtlinge',
                'Bundesamtes für Flüchtlinge',
            ],
            # Bundesamt.s.es für Migration
            [
                'Bundesamt für Migration',
                'Bundesamts für Migration',
                'Bundesamtes für Migration',
            ],
        ],
        "de_abbrev": 'SEM', "de_alt_abbrevs": ['BFM'],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDJP',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Office fédéral des assurances privées', 'office fédéral des assurances privées'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Privatversicherungen
            [
                'Bundesamt für Privatversicherungen',
                'Bundesamts für Privatversicherungen',
                'Bundesamtes für Privatversicherungen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDJP',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Office fédéral de métrologie', 'office fédéral de métrologie'],
        ],
        "fr_abbrev": 'METAS', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Institut.s für Metrologie
            [
                'Eidgenössische Institut für Metrologie',
                'Eidgenössisches Institut für Metrologie',
                'Eidgenössischen Institut für Metrologie',
                'Eidgenössischen Instituts für Metrologie',
            ],
            # Eidgenössische.s.n Amt.s.es für Messwesen
            [
                'Eidgenössische Amt für Messwesen',
                'Eidgenössisches Amt für Messwesen',
                'Eidgenössischen Amt für Messwesen',
                'Eidgenössischen Amts für Messwesen',
                'Eidgenössischen Amtes für Messwesen',
            ],
        ],
        "de_abbrev": 'METAS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'METAS': ['Metas']},
    },
    # ──────────────── DDPS ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-DDPS',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DDPS', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-VBS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'State Secretariat for Security Policy', "en_abbrev": 'SEPOS',
        "fr_names": [
            ["Secrétariat d'Etat à la politique de sécurité", "secrétariat d'Etat à la politique de sécurité", "Secrétariat d'État à la politique de sécurité", "secrétariat d'État à la politique de sécurité", "Secrétariat d'état à la politique de sécurité", "secrétariat d'état à la politique de sécurité"],
            ['Direction de la politique de sécurité', 'direction de la politique de sécurité'],
        ],
        "fr_abbrev": 'SEPOS', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für Sicherheitspolitik
            [
                'Staatssekretariat für Sicherheitspolitik',
                'Staatssekretariats für Sicherheitspolitik',
            ],
            ['Direktion für Sicherheitspolitik'],
        ],
        "de_abbrev": 'SEPOS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'SEPOS': ['Sepos']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Intelligence Service', "en_abbrev": 'FIS',
        "fr_names": [
            ['Service de renseignement de la Confédération', 'service de renseignement de la Confédération'],
            ['Direction du Renseignement stratégique', 'direction du Renseignement stratégique'],
        ],
        "fr_abbrev": 'SRC', "fr_alt_abbrevs": [],
        "de_names": [
            # Nachrichtendienst.es des Bundes
            [
                'Nachrichtendienst des Bundes',
                'Nachrichtendienstes des Bundes',
            ],
            # Direktion für strategische.n.r Nachrichtendienst
            [
                'Direktion für strategischen Nachrichtendienst',
            ],
        ],
        "de_abbrev": 'NDB', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Office of the Armed Force Attorney General', "en_abbrev": 'OAFAG',
        "fr_names": [
            ['Office de l’auditeur en chef', 'office de l’auditeur en chef'],
        ],
        "fr_abbrev": 'OAC', "fr_alt_abbrevs": [],
        "de_names": [
            # Oberauditorat.s
            [
                'Oberauditorat',
                'Oberauditorats',
            ],
        ],
        "de_abbrev": 'OA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Defence Group', "en_abbrev": '',
        "fr_names": [
            ['Groupement Défense', 'groupement Défense'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Gruppe Verteidigung'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office for Defence Procurement', "en_abbrev": 'armasuisse',
        "fr_names": [
            ['Office fédéral de l’armement', 'office fédéral de l’armement'],
            ['Groupement de l’armement', 'groupement de l’armement'],
            ['Groupement armasuisse', 'groupement armasuisse'],
        ],
        "fr_abbrev": 'armasuisse', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Rüstung
            [
                'Bundesamt für Rüstung',
                'Bundesamts für Rüstung',
                'Bundesamtes für Rüstung',
            ],
            ['Gruppe Rüstung'],
            ['Gruppe armasuisse'],
        ],
        "de_abbrev": 'armasuisse', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'armasuisse': ['Armasuisse', 'ARMASUISSE', 'ArmaSuisse']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office of Topography', "en_abbrev": 'swisstopo',
        "fr_names": [
            ['Office fédéral de topographie', 'office fédéral de topographie'],
        ],
        "fr_abbrev": 'swisstopo', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Landestopografie
            [
                'Bundesamt für Landestopografie',
                'Bundesamts für Landestopografie',
                'Bundesamtes für Landestopografie',
            ],
        ],
        "de_abbrev": 'swisstopo', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'swisstopo': ['Swisstopo', 'SWISSTOPO', 'SwissTopo']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office for Civil Protection', "en_abbrev": 'FOCP',
        "fr_names": [
            ['Office fédéral de la protection de la population', 'office fédéral de la protection de la population'],
            ['Office fédéral de la protection civile', 'office fédéral de la protection civile'],
        ],
        "fr_abbrev": 'OFPP', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Bevölkerungsschutz
            [
                'Bundesamt für Bevölkerungsschutz',
                'Bundesamts für Bevölkerungsschutz',
                'Bundesamtes für Bevölkerungsschutz',
            ],
            # Bundesamt.s.es für Zivilschutz
            [
                'Bundesamt für Zivilschutz',
                'Bundesamts für Zivilschutz',
                'Bundesamtes für Zivilschutz',
            ],
        ],
        "de_abbrev": 'BABS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFPP': ['Ofpp'], 'BABS': ['Babs']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office of Sport', "en_abbrev": 'FOSPO',
        "fr_names": [
            ['Office fédéral du sport', 'office fédéral du sport'],
        ],
        "fr_abbrev": 'OFSPO', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Sport
            [
                'Bundesamt für Sport',
                'Bundesamts für Sport',
                'Bundesamtes für Sport',
            ],
        ],
        "de_abbrev": 'BASPO', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFSPO': ['Ofspo'], 'BASPO': ['Baspo']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'National Cyber Security Centre', "en_abbrev": 'NCSC',
        "fr_names": [
            ['Office fédéral de la cybersécurité', 'office fédéral de la cybersécurité'],
        ],
        "fr_abbrev": 'OFCS', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Cybersicherheit
            [
                'Bundesamt für Cybersicherheit',
                'Bundesamts für Cybersicherheit',
                'Bundesamtes für Cybersicherheit',
            ],
        ],
        "de_abbrev": 'BACS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFCS': ['Ofcs'], 'BACS': ['Bacs']},
    },
    # Ajout accountability (nov. 2026) : absente du CSV et retirée de la liste
    # SWISSDOX, remise car l'armée est une cible fréquente (et critiquée) dans
    # les données GOLD. Pas d'abréviation.
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Swiss Armed Forces', "en_abbrev": '',
        "fr_names": [
            ['Armée suisse', 'armée suisse'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizer Armee (« Schweizer » invariable, nom féminin)
            ['Schweizer Armee'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DDPS',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Forces aériennes', 'forces aériennes'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Luftwaffe'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DDPS',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Forces terrestres', 'forces terrestres'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Heer.es
            [
                'Heer',
                'Heeres',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DDPS',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Etat-major général', 'etat-major général', 'État-major général', 'état-major général'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalstab.s.es
            [
                'Generalstab',
                'Generalstabs',
                'Generalstabes',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── FDF ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDF',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DFF', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EFD', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'State Secretariat for International Finance', "en_abbrev": 'SIF',
        "fr_names": [
            ["Secrétariat d'état aux questions financières internationales", "secrétariat d'état aux questions financières internationales", "Secrétariat d'État aux questions financières internationales", "secrétariat d'État aux questions financières internationales", "Secrétariat d'Etat aux questions financières internationales", "secrétariat d'Etat aux questions financières internationales"],
        ],
        "fr_abbrev": 'SFI', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für internationale Finanzfragen
            [
                'Staatssekretariat für internationale Finanzfragen',
                'Staatssekretariats für internationale Finanzfragen',
            ],
        ],
        "de_abbrev": 'SIF', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Finance Administration', "en_abbrev": 'FFA',
        "fr_names": [
            ['Administration fédérale des finances', 'administration fédérale des finances'],
        ],
        "fr_abbrev": 'AFF', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n.r Finanzverwaltung
            [
                'Eidgenössische Finanzverwaltung',
                'Eidgenössischen Finanzverwaltung',
            ],
        ],
        "de_abbrev": 'EFV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office of Personnel', "en_abbrev": 'FOPER',
        "fr_names": [
            ['Office fédéral du personnel', 'office fédéral du personnel'],
        ],
        "fr_abbrev": 'OFPER', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Personalamt.s.es
            [
                'Eidgenössische Personalamt',
                'Eidgenössisches Personalamt',
                'Eidgenössischen Personalamt',
                'Eidgenössischen Personalamts',
                'Eidgenössischen Personalamtes',
            ],
        ],
        "de_abbrev": 'EPA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFPER': ['Ofper']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Tax Administration', "en_abbrev": 'FTA',
        "fr_names": [
            ['Administration fédérale des contributions', 'administration fédérale des contributions'],
        ],
        "fr_abbrev": 'AFC', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n.r Steuerverwaltung
            [
                'Eidgenössische Steuerverwaltung',
                'Eidgenössischen Steuerverwaltung',
            ],
        ],
        "de_abbrev": 'ESTV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'ESTV': ['Estv']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office for Customs and Border Security', "en_abbrev": 'FOCBS',
        "fr_names": [
            ['Office fédéral de la douane et de la sécurité des frontières', 'office fédéral de la douane et de la sécurité des frontières'],
            ['Office fédéral des douanes', 'office fédéral des douanes'],
        ],
        "fr_abbrev": 'OFDF', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Zoll und Grenzsicherheit
            [
                'Bundesamt für Zoll und Grenzsicherheit',
                'Bundesamts für Zoll und Grenzsicherheit',
                'Bundesamtes für Zoll und Grenzsicherheit',
            ],
            # Eidgenössische.n.r Zollverwaltung
            [
                'Eidgenössische Zollverwaltung',
                'Eidgenössischen Zollverwaltung',
            ],
        ],
        "de_abbrev": 'BAZG', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFDF': ['Ofdf'], 'BAZG': ['Bazg']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office of Information Technology, Systems and Telecommunication', "en_abbrev": 'FOITT',
        "fr_names": [
            ['Office fédéral de l’informatique et de la télécommunication', 'office fédéral de l’informatique et de la télécommunication'],
            ['Office fédéral de l’informatique', 'office fédéral de l’informatique'],
        ],
        "fr_abbrev": 'OFIT', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Informatik und Telekommunikation
            [
                'Bundesamt für Informatik und Telekommunikation',
                'Bundesamts für Informatik und Telekommunikation',
                'Bundesamtes für Informatik und Telekommunikation',
            ],
            # Bundesamt.s.es für Informatik
            [
                'Bundesamt für Informatik',
                'Bundesamts für Informatik',
                'Bundesamtes für Informatik',
            ],
        ],
        "de_abbrev": 'BIT', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFIT': ['Ofit']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office for Buildings and Logistics', "en_abbrev": 'FOBL',
        "fr_names": [
            ['Office fédéral des constructions et de la logistique', 'office fédéral des constructions et de la logistique'],
        ],
        "fr_abbrev": 'OFCL', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Bauten und Logistik
            [
                'Bundesamt für Bauten und Logistik',
                'Bundesamts für Bauten und Logistik',
                'Bundesamtes für Bauten und Logistik',
            ],
        ],
        "de_abbrev": 'BBL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFCL': ['Ofcl']},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDF',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Caisse fédérale d’assurance', 'caisse fédérale d’assurance'],
            ['Office fédéral des assurances privées', 'office fédéral des assurances privées'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n.r Versicherungskasse
            [
                'Eidgenössische Versicherungskasse',
                'Eidgenössischen Versicherungskasse',
            ],
            # Bundesamt.s.es für Privatversicherungen
            [
                'Bundesamt für Privatversicherungen',
                'Bundesamts für Privatversicherungen',
                'Bundesamtes für Privatversicherungen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDF',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Unité de pilotage informatique de la Confédération', 'unité de pilotage informatique de la Confédération'],
        ],
        "fr_abbrev": 'UPIC', "fr_alt_abbrevs": [],
        "de_names": [
            # Informatiksteuerungsorgan.s des Bundes
            [
                'Informatiksteuerungsorgan des Bundes',
                'Informatiksteuerungsorgans des Bundes',
            ],
        ],
        "de_abbrev": 'ISB', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'UPIC': ['Upic']},
    },
    # ──────────────── EAER ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-EAER',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DEFR', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-WBF', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Office of the Price Supervisor', "en_abbrev": 'PRS',
        "fr_names": [
            ['Surveillance des prix', 'surveillance des prix'],
        ],
        "fr_abbrev": 'SPR', "fr_alt_abbrevs": [],
        "de_names": [
            ['Preisüberwachung'],
        ],
        "de_abbrev": 'PUE', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'State Secretariat for Economic Affairs', "en_abbrev": 'SECO',
        "fr_names": [
            ['Secrétariat d’Etat à l’économie', 'secrétariat d’Etat à l’économie', 'Secrétariat d’État à l’économie', 'secrétariat d’État à l’économie', 'Secrétariat d’état à l’économie', 'secrétariat d’état à l’économie'],
        ],
        "fr_abbrev": 'SECO', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für Wirtschaft
            [
                'Staatssekretariat für Wirtschaft',
                'Staatssekretariats für Wirtschaft',
            ],
        ],
        "de_abbrev": 'SECO', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'SECO': ['Seco']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'State Secretariat for Education, Research and Innovation', "en_abbrev": 'SERI',
        "fr_names": [
            ["Secrétariat d'Etat à la formation, à la recherche et à l’innovation", "secrétariat d'Etat à la formation, à la recherche et à l’innovation", "Secrétariat d'État à la formation, à la recherche et à l’innovation", "secrétariat d'État à la formation, à la recherche et à l’innovation", "Secrétariat d'état à la formation, à la recherche et à l’innovation", "secrétariat d'état à la formation, à la recherche et à l’innovation"],
            ['Office fédéral de la formation professionnelle et de la technologie', 'office fédéral de la formation professionnelle et de la technologie'],
        ],
        "fr_abbrev": 'SEFRI', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für Bildung, Forschung und Innovation
            [
                'Staatssekretariat für Bildung, Forschung und Innovation',
                'Staatssekretariats für Bildung, Forschung und Innovation',
            ],
            # Bundesamt.s.es für Berufsbildung und Technologie
            [
                'Bundesamt für Berufsbildung und Technologie',
                'Bundesamts für Berufsbildung und Technologie',
                'Bundesamtes für Berufsbildung und Technologie',
            ],
        ],
        "de_abbrev": 'SBFI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'SEFRI': ['Sefri'], 'SBFI': ['Sbfi']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for Agriculture', "en_abbrev": 'FOAG',
        "fr_names": [
            ['Office fédéral de l’agriculture', 'office fédéral de l’agriculture'],
        ],
        "fr_abbrev": 'OFAG', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Landwirtschaft
            [
                'Bundesamt für Landwirtschaft',
                'Bundesamts für Landwirtschaft',
                'Bundesamtes für Landwirtschaft',
            ],
        ],
        "de_abbrev": 'BLW', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFAG': ['Ofag']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for National Economic Supply', "en_abbrev": 'FONES',
        "fr_names": [
            ['Office fédéral pour l’approvisionnement économique du pays', 'office fédéral pour l’approvisionnement économique du pays'],
        ],
        "fr_abbrev": 'OFAE', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für wirtschaftliche Landesversorgung
            [
                'Bundesamt für wirtschaftliche Landesversorgung',
                'Bundesamts für wirtschaftliche Landesversorgung',
                'Bundesamtes für wirtschaftliche Landesversorgung',
            ],
        ],
        "de_abbrev": 'BWL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFAE': ['Ofae']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for Housing', "en_abbrev": 'FOH',
        "fr_names": [
            ['Office fédéral du logement', 'office fédéral du logement'],
        ],
        "fr_abbrev": 'OFL', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Wohnungswesen
            [
                'Bundesamt für Wohnungswesen',
                'Bundesamts für Wohnungswesen',
                'Bundesamtes für Wohnungswesen',
            ],
        ],
        "de_abbrev": 'BWO', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for Civilian Service', "en_abbrev": 'CIVI',
        "fr_names": [
            ['Office fédéral du service civil', 'office fédéral du service civil'],
        ],
        "fr_abbrev": 'CIVI', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Zivildienst
            [
                'Bundesamt für Zivildienst',
                'Bundesamts für Zivildienst',
                'Bundesamtes für Zivildienst',
            ],
        ],
        "de_abbrev": 'ZIVI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'CIVI': ['Civi']},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'EAER',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Office vétérinaire fédéral', 'office vétérinaire fédéral'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Veterinärwesen
            [
                'Bundesamt für Veterinärwesen',
                'Bundesamts für Veterinärwesen',
                'Bundesamtes für Veterinärwesen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── DETEC ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-DETEC',
        "fr_names": [
            ['Secrétariat général', 'secrétariat général'],
        ],
        "fr_abbrev": 'SG-DETEC', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-UVEK', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office of Transport', "en_abbrev": 'FOT',
        "fr_names": [
            ['Office fédéral des transports', 'office fédéral des transports'],
        ],
        "fr_abbrev": 'OFT', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Verkehr
            [
                'Bundesamt für Verkehr',
                'Bundesamts für Verkehr',
                'Bundesamtes für Verkehr',
            ],
        ],
        "de_abbrev": 'BAV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office of Civil Aviation', "en_abbrev": 'FOCA',
        "fr_names": [
            ['Office fédéral de l’aviation civile', 'office fédéral de l’aviation civile'],
        ],
        "fr_abbrev": 'OFAC', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Zivilluftfahrt
            [
                'Bundesamt für Zivilluftfahrt',
                'Bundesamts für Zivilluftfahrt',
                'Bundesamtes für Zivilluftfahrt',
            ],
        ],
        "de_abbrev": 'BAZL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFAC': ['Ofac'], 'BAZL': ['Bazl']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Swiss Federal Office of Energy', "en_abbrev": 'SFOE',
        "fr_names": [
            ['Office fédéral de l’énergie', 'office fédéral de l’énergie'],
        ],
        "fr_abbrev": 'OFEN', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Energie
            [
                'Bundesamt für Energie',
                'Bundesamts für Energie',
                'Bundesamtes für Energie',
            ],
        ],
        "de_abbrev": 'BFE', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Roads Office', "en_abbrev": 'FEDRO',
        "fr_names": [
            ['Office fédéral des routes', 'office fédéral des routes'],
        ],
        "fr_abbrev": 'OFROU', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Strassen
            [
                'Bundesamt für Strassen',
                'Bundesamts für Strassen',
                'Bundesamtes für Strassen',
            ],
        ],
        "de_abbrev": 'ASTRA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFROU': ['Ofrou'], 'ASTRA': ['Astra']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office of Communications', "en_abbrev": 'OFCOM',
        "fr_names": [
            ['Office fédéral de la communication', 'office fédéral de la communication'],
        ],
        "fr_abbrev": 'OFCOM', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Kommunikation
            [
                'Bundesamt für Kommunikation',
                'Bundesamts für Kommunikation',
                'Bundesamtes für Kommunikation',
            ],
        ],
        "de_abbrev": 'BAKOM', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'BAKOM': ['Bakom']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office for the Environment', "en_abbrev": 'FOEN',
        "fr_names": [
            ['Office fédéral de l’environnement', 'office fédéral de l’environnement'],
            ['Office fédéral de l’environnement, des forêts et du paysage', 'office fédéral de l’environnement, des forêts et du paysage'],
        ],
        "fr_abbrev": 'OFEV', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Umwelt
            [
                'Bundesamt für Umwelt',
                'Bundesamts für Umwelt',
                'Bundesamtes für Umwelt',
            ],
            # Bundesamt.s.es für Umwelt, Wald und Landschaft
            [
                'Bundesamt für Umwelt, Wald und Landschaft',
                'Bundesamts für Umwelt, Wald und Landschaft',
                'Bundesamtes für Umwelt, Wald und Landschaft',
            ],
        ],
        "de_abbrev": 'BAFU', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'OFEV': ['Ofev'], 'BAFU': ['Bafu']},
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office for Spatial Development', "en_abbrev": 'ARE',
        "fr_names": [
            ['Office fédéral du développement territorial', 'office fédéral du développement territorial'],
        ],
        "fr_abbrev": 'ARE', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Raumentwicklung
            [
                'Bundesamt für Raumentwicklung',
                'Bundesamts für Raumentwicklung',
                'Bundesamtes für Raumentwicklung',
            ],
        ],
        "de_abbrev": 'ARE', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DETEC',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Office fédéral des eaux et de la géologie', 'office fédéral des eaux et de la géologie'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Wasser und Geologie
            [
                'Bundesamt für Wasser und Geologie',
                'Bundesamts für Wasser und Geologie',
                'Bundesamtes für Wasser und Geologie',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
]


# ════════════════════════════════════════════════════════════════════════════
# MOTS-CLÉS SPÉCIFIQUES : AGENCES INDÉPENDANTES (une entrée = une ligne
# « Independent Agency » / « OLD Independent Agency » de
# « Liste des départements fédéraux.csv », nov. 2026)
#
# Même structure et mêmes règles que FEDERAL_UNITS ci-dessus :
#   - FR : chaque nom avec majuscule ET minuscule initiale ; É initial ->
#     É / é / E / e (comme « État-major général »).
#   - DE : déclinaisons écrites en toutes lettres, même règle d'accord que
#     pour les unités (validée par la collègue germanophone) :
#       adjectif + nom neutre   (Eidgenössische.s.n Institut.s)
#           -> -e / -es / -en, génitif seulement avec -en (« des Eidgenössischen Instituts »)
#       adjectif + nom féminin  (Eidgenössische.n Finanzkontrolle)
#           -> -e / -en
#       nom masculin / neutre seul (Paul-Scherrer-Institut.s, Dienst.es)
#           -> + génitif
#       nom féminin seul ou pluriel (Bundesanwaltschaft, Finanzkommissionen)
#           -> une seule forme
#     Le commentaire au-dessus de chaque nom allemand donne la notation
#     appliquée (dans le CSV, les agences n'avaient pas de notation).
#   - Abréviations : même règle de casse (majuscules de 4 lettres ou plus ->
#     forme avec seule l'initiale en majuscule ; noms de marque -> variantes
#     MAJUSCULES / CamelCase).
#   - Les établissements qui apparaissent deux fois dans le CSV (ancien
#     rattachement FDHA en « OLD », rattachement actuel EAER) ont deux
#     entrées, comme dans le CSV ; les termes identiques ne sont envoyés
#     qu'une fois à Swissdox.
# ════════════════════════════════════════════════════════════════════════════
INDEPENDENT_AGENCIES = [
    # ──────────────── FDHA ────────────────
    {
        "type": 'Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss National Museum', "en_abbrev": 'SNM',
        "fr_names": [
            ['Musée national suisse', 'musée national suisse'],
        ],
        "fr_abbrev": 'MNS', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.s.n Nationalmuseum.s
            [
                'Schweizerisches Nationalmuseum',
                'Schweizerische Nationalmuseum',
                'Schweizerischen Nationalmuseum',
                'Schweizerischen Nationalmuseums',
            ],
        ],
        "de_abbrev": 'SNM', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Arts Council Pro Helvetia', "en_abbrev": 'Pro Helvetia',
        "fr_names": [
            ['Fondation suisse pour la culture', 'fondation suisse pour la culture'],
        ],
        "fr_abbrev": 'Pro Helvetia', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizer Kulturstiftung (« Schweizer » invariable, nom féminin)
            ['Schweizer Kulturstiftung'],
        ],
        "de_abbrev": 'Pro Helvetia', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'Pro Helvetia': ['PRO HELVETIA']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Agency for Therapeutic Products', "en_abbrev": 'Swissmedic',
        "fr_names": [
            ['Institut suisse des produits thérapeutiques', 'institut suisse des produits thérapeutiques'],
        ],
        "fr_abbrev": 'Swissmedic', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.s.n Heilmittelinstitut.s
            [
                'Schweizerisches Heilmittelinstitut',
                'Schweizerische Heilmittelinstitut',
                'Schweizerischen Heilmittelinstitut',
                'Schweizerischen Heilmittelinstituts',
            ],
        ],
        "de_abbrev": 'Swissmedic', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'Swissmedic': ['SWISSMEDIC', 'swissmedic', 'SwissMedic']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Social Security Funds AHV/IV/EO', "en_abbrev": 'compenswiss',
        "fr_names": [
            ['Fonds de compensation AVS/AI/APG', 'fonds de compensation AVS/AI/APG'],
        ],
        "fr_abbrev": 'compenswiss', "fr_alt_abbrevs": [],
        "de_names": [
            # Ausgleichsfonds AHV/IV/EO (génitif identique)
            ['Ausgleichsfonds AHV/IV/EO'],
        ],
        "de_abbrev": 'Compenswiss', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'compenswiss': ['Compenswiss', 'COMPENSWISS', 'CompenSwiss']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Institute of Aquatic Science and Technology', "en_abbrev": 'EAWAG',
        "fr_names": [
            ['Institut fédéral pour l’aménagement, l’épuration et la protection des eaux', 'institut fédéral pour l’aménagement, l’épuration et la protection des eaux'],
        ],
        "fr_abbrev": 'IFAEPE', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Anstalt für Wasserversorgung, Abwasserreinigung und Gewässerschutz
            [
                'Eidgenössische Anstalt für Wasserversorgung, Abwasserreinigung und Gewässerschutz',
                'Eidgenössischen Anstalt für Wasserversorgung, Abwasserreinigung und Gewässerschutz',
            ],
        ],
        "de_abbrev": 'EAWAG', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'IFAEPE': ['Ifaepe'], 'EAWAG': ['Eawag']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Institute for Forest, Snow and Landscape Research', "en_abbrev": 'WSL',
        "fr_names": [
            ['Institut fédéral de recherches sur la forêt, la neige et le paysage', 'institut fédéral de recherches sur la forêt, la neige et le paysage'],
        ],
        "fr_abbrev": 'FNP', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Forschungsanstalt für Wald, Schnee und Landschaft
            [
                'Eidgenössische Forschungsanstalt für Wald, Schnee und Landschaft',
                'Eidgenössischen Forschungsanstalt für Wald, Schnee und Landschaft',
            ],
        ],
        "de_abbrev": 'WSL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Laboratories for Materials Science and Technology', "en_abbrev": 'Empa',
        "fr_names": [
            ['Laboratoire fédéral d’essai des matériaux et de recherche', 'laboratoire fédéral d’essai des matériaux et de recherche'],
        ],
        "fr_abbrev": 'LFEM', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Materialprüfungs- und Forschungsanstalt
            [
                'Eidgenössische Materialprüfungs- und Forschungsanstalt',
                'Eidgenössischen Materialprüfungs- und Forschungsanstalt',
            ],
        ],
        "de_abbrev": 'Empa', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'LFEM': ['Lfem'], 'Empa': ['EMPA']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Paul Scherrer Institute', "en_abbrev": 'PSI',
        "fr_names": [
            ['Institut Paul Scherrer', 'institut Paul Scherrer'],
        ],
        "fr_abbrev": 'IPS', "fr_alt_abbrevs": [],
        "de_names": [
            # Paul-Scherrer-Institut.s
            [
                'Paul-Scherrer-Institut',
                'Paul-Scherrer-Instituts',
            ],
        ],
        "de_abbrev": 'PSI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Institute of Technology Zurich', "en_abbrev": 'ETHZ',
        "fr_names": [
            ['École polytechnique fédérale de Zurich', 'école polytechnique fédérale de Zurich', 'Ecole polytechnique fédérale de Zurich', 'ecole polytechnique fédérale de Zurich'],
            ['Écoles polytechniques fédérales', 'écoles polytechniques fédérales', 'Ecoles polytechniques fédérales', 'ecoles polytechniques fédérales'],
        ],
        "fr_abbrev": 'EPFZ', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Technische.n Hochschulen Zürich (pluriel, tel quel dans le CSV)
            [
                'Eidgenössische Technische Hochschulen Zürich',
                'Eidgenössischen Technischen Hochschulen Zürich',
            ],
            # Eidgenössische.n Technische.n Hochschulen (pluriel)
            [
                'Eidgenössische Technische Hochschulen',
                'Eidgenössischen Technischen Hochschulen',
            ],
        ],
        "de_abbrev": 'ETHZ', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'EPFZ': ['Epfz'], 'ETHZ': ['Ethz']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Institute of Technology Lausanne', "en_abbrev": 'EPFL',
        "fr_names": [
            ['École polytechnique fédérale de Lausanne', 'école polytechnique fédérale de Lausanne', 'Ecole polytechnique fédérale de Lausanne', 'ecole polytechnique fédérale de Lausanne'],
            ['Écoles polytechniques fédérales', 'écoles polytechniques fédérales', 'Ecoles polytechniques fédérales', 'ecoles polytechniques fédérales'],
        ],
        "fr_abbrev": 'EPFL', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Technische.n Hochschulen Lausanne (pluriel, tel quel dans le CSV)
            [
                'Eidgenössische Technische Hochschulen Lausanne',
                'Eidgenössischen Technischen Hochschulen Lausanne',
            ],
            # Eidgenössische.n Technische.n Hochschulen (pluriel)
            [
                'Eidgenössische Technische Hochschulen',
                'Eidgenössischen Technischen Hochschulen',
            ],
        ],
        "de_abbrev": 'ETHL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'EPFL': ['Epfl'], 'ETHL': ['Ethl']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDHA',
        "en_name": 'Domain of the Swiss Federal Institutes of Technology', "en_abbrev": 'ETH Domain',
        "fr_names": [
            ['Domaine des écoles polytechniques fédérales', 'domaine des écoles polytechniques fédérales', 'Domaine des Écoles polytechniques fédérales', 'domaine des Écoles polytechniques fédérales', 'Domaine des Ecoles polytechniques fédérales', 'domaine des Ecoles polytechniques fédérales'],
            ['Conseil des écoles polytechniques fédérales', 'conseil des écoles polytechniques fédérales', 'Conseil des Écoles polytechniques fédérales', 'conseil des Écoles polytechniques fédérales', 'Conseil des Ecoles polytechniques fédérales', 'conseil des Ecoles polytechniques fédérales'],
        ],
        "fr_abbrev": 'Domaine des EPF', "fr_alt_abbrevs": [],
        "de_names": [
            # Bereich.s.es der Eidgenössischen Technischen Hochschulen
            [
                'Bereich der Eidgenössischen Technischen Hochschulen',
                'Bereichs der Eidgenössischen Technischen Hochschulen',
                'Bereiches der Eidgenössischen Technischen Hochschulen',
            ],
            # Rat.s.es der Eidgenössischen Technischen Hochschulen
            [
                'Rat der Eidgenössischen Technischen Hochschulen',
                'Rats der Eidgenössischen Technischen Hochschulen',
                'Rates der Eidgenössischen Technischen Hochschulen',
            ],
        ],
        "de_abbrev": 'ETH-Bereich', "de_alt_abbrevs": [],
        # 'ETH-Bereich' est un nom composé : on ajoute aussi son génitif
        "abbrev_case_variants": {'Domaine des EPF': ['domaine des EPF'], 'ETH-Bereich': ['ETH-Bereichs', 'ETH-Bereiches']},
    },
    # ──────────────── FDJP ────────────────
    {
        "type": 'Independent Agency', "upper_level": 'FDJP',
        "en_name": 'Post and Telecommunications Surveillance Service', "en_abbrev": 'PTSS',
        "fr_names": [
            ['Service de Surveillance de la correspondance par poste et télécommunication', 'service de Surveillance de la correspondance par poste et télécommunication', 'Service de surveillance de la correspondance par poste et télécommunication', 'service de surveillance de la correspondance par poste et télécommunication'],
        ],
        "fr_abbrev": 'SCPT', "fr_alt_abbrevs": [],
        "de_names": [
            # Dienst.es Überwachung Post- und Fernmeldeverkehr
            [
                'Dienst Überwachung Post- und Fernmeldeverkehr',
                'Dienstes Überwachung Post- und Fernmeldeverkehr',
            ],
        ],
        "de_abbrev": 'ÜPF', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'SCPT': ['Scpt']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDJP',
        "en_name": 'National Commission for the Prevention of Torture', "en_abbrev": 'NCPT',
        "fr_names": [
            ['Commission nationale de prévention de la torture', 'commission nationale de prévention de la torture'],
            ['Commission de prévention de la torture', 'commission de prévention de la torture'],
        ],
        "fr_abbrev": 'CNPT', "fr_alt_abbrevs": [],
        "de_names": [
            # Nationale.n Kommission zur Verhütung von Folter
            [
                'Nationale Kommission zur Verhütung von Folter',
                'Nationalen Kommission zur Verhütung von Folter',
            ],
            ['Kommission zur Verhütung von Folter'],
        ],
        "de_abbrev": 'NKVF', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'CNPT': ['Cnpt'], 'NKVF': ['Nkvf']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDJP',
        "en_name": 'Swiss Institute of Comparative Law', "en_abbrev": 'SICL',
        "fr_names": [
            ['Institut suisse de droit comparé', 'institut suisse de droit comparé'],
        ],
        "fr_abbrev": 'ISDC', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.s.n Institut.s für Rechtsvergleichung
            [
                'Schweizerisches Institut für Rechtsvergleichung',
                'Schweizerische Institut für Rechtsvergleichung',
                'Schweizerischen Institut für Rechtsvergleichung',
                'Schweizerischen Instituts für Rechtsvergleichung',
            ],
        ],
        "de_abbrev": 'SIR', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'ISDC': ['Isdc']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDJP',
        "en_name": 'Swiss Federal Institute of Intellectual Property', "en_abbrev": 'IPI',
        "fr_names": [
            ['Institut fédéral de la propriété intellectuelle', 'institut fédéral de la propriété intellectuelle'],
        ],
        "fr_abbrev": 'IPI', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Institut.s für Geistiges Eigentum
            [
                'Eidgenössisches Institut für Geistiges Eigentum',
                'Eidgenössische Institut für Geistiges Eigentum',
                'Eidgenössischen Institut für Geistiges Eigentum',
                'Eidgenössischen Instituts für Geistiges Eigentum',
            ],
        ],
        "de_abbrev": 'IGE', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDJP',
        "en_name": 'Federal Audit Supervisory Authority', "en_abbrev": 'FAOA',
        "fr_names": [
            ['Autorité fédérale de surveillance en matière de révision', 'autorité fédérale de surveillance en matière de révision'],
        ],
        "fr_abbrev": 'ASR', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Revisionsaufsichtsbehörde
            [
                'Eidgenössische Revisionsaufsichtsbehörde',
                'Eidgenössischen Revisionsaufsichtsbehörde',
            ],
        ],
        "de_abbrev": 'RAB', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDJP',
        "en_name": 'Federal Institute of Metrology', "en_abbrev": 'METAS',
        "fr_names": [
            ['Institut fédéral de métrologie', 'institut fédéral de métrologie'],
        ],
        "fr_abbrev": 'METAS', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Institut.s für Metrologie
            [
                'Eidgenössisches Institut für Metrologie',
                'Eidgenössische Institut für Metrologie',
                'Eidgenössischen Institut für Metrologie',
                'Eidgenössischen Instituts für Metrologie',
            ],
        ],
        "de_abbrev": 'METAS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'METAS': ['Metas']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDJP',
        "en_name": 'Office of the Attorney General of Switzerland', "en_abbrev": 'OAG',
        "fr_names": [
            ['Ministère public de la Confédération', 'ministère public de la Confédération'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Bundesanwaltschaft'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── DDPS ────────────────
    {
        "type": 'Independent Agency', "upper_level": 'DDPS',
        "en_name": 'Independent Oversight Authority for Intelligence Activities', "en_abbrev": 'OA-IA',
        "fr_names": [
            ['Autorité de surveillance indépendante des activités de renseignement', 'autorité de surveillance indépendante des activités de renseignement'],
        ],
        "fr_abbrev": 'AS-Rens', "fr_alt_abbrevs": [],
        "de_names": [
            # Unabhängige.n Aufsichtsbehörde über die nachrichtendienstlichen Tätigkeiten
            [
                'Unabhängige Aufsichtsbehörde über die nachrichtendienstlichen Tätigkeiten',
                'Unabhängigen Aufsichtsbehörde über die nachrichtendienstlichen Tätigkeiten',
            ],
        ],
        "de_abbrev": 'AB-ND', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'AS-Rens': ['AS-RENS']},
    },
    # ──────────────── FDF ────────────────
    {
        "type": 'Independent Agency', "upper_level": 'FDF',
        "en_name": 'Swiss Federal Audit Office', "en_abbrev": 'SFAO',
        "fr_names": [
            ['Contrôle fédéral des finances', 'contrôle fédéral des finances'],
        ],
        "fr_abbrev": 'CDF', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Finanzkontrolle
            [
                'Eidgenössische Finanzkontrolle',
                'Eidgenössischen Finanzkontrolle',
            ],
        ],
        "de_abbrev": 'EFK', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDF',
        "en_name": 'Swiss Federal Financial Market Supervisory Authority', "en_abbrev": 'FINMA',
        "fr_names": [
            ['Autorité fédérale de surveillance des marchés financiers', 'autorité fédérale de surveillance des marchés financiers'],
        ],
        "fr_abbrev": 'FINMA', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Finanzmarktaufsicht
            [
                'Eidgenössische Finanzmarktaufsicht',
                'Eidgenössischen Finanzmarktaufsicht',
            ],
        ],
        "de_abbrev": 'FINMA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'FINMA': ['Finma']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'FDF',
        "en_name": 'Swiss Federal Pension Fund', "en_abbrev": 'PUBLICA',
        "fr_names": [
            ['Caisse fédérale de pensions', 'caisse fédérale de pensions'],
        ],
        "fr_abbrev": 'PUBLICA', "fr_alt_abbrevs": [],
        "de_names": [
            ['Pensionskasse des Bundes'],
        ],
        "de_abbrev": 'PUBLICA', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'PUBLICA': ['Publica']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDF',
        "en_name": 'Swiss Alcohol Board', "en_abbrev": 'SAB',
        "fr_names": [
            ['Régie fédérale des alcools', 'régie fédérale des alcools'],
        ],
        "fr_abbrev": 'RFA', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Alkoholverwaltung
            [
                'Eidgenössische Alkoholverwaltung',
                'Eidgenössischen Alkoholverwaltung',
            ],
        ],
        "de_abbrev": 'EAV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDF',
        "en_name": 'Swiss Federal Banking Commission', "en_abbrev": 'SFBC',
        "fr_names": [
            ['Commission fédérale des banques', 'commission fédérale des banques'],
        ],
        "fr_abbrev": 'CFB', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Bankenkommission
            [
                'Eidgenössische Bankenkommission',
                'Eidgenössischen Bankenkommission',
            ],
        ],
        "de_abbrev": 'EBK', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDF',
        "en_name": 'Finance Committees', "en_abbrev": 'FC',
        "fr_names": [
            ['Commissions des finances', 'commissions des finances'],
        ],
        "fr_abbrev": 'Cdf', "fr_alt_abbrevs": [],
        "de_names": [
            ['Finanzkommissionen'],
        ],
        "de_abbrev": 'FK', "de_alt_abbrevs": [],
        # pas de variante 'CDF' : c'est l'abréviation du Contrôle fédéral des finances
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'FDF',
        "en_name": 'Finance Delegation', "en_abbrev": 'FINDEL',
        "fr_names": [
            # CSV : « La délégation des finances » ; l'article est retiré pour
            # que « la Délégation des finances », « de la délégation ... » matchent
            ['Délégation des finances', 'délégation des finances'],
        ],
        "fr_abbrev": 'DélFin', "fr_alt_abbrevs": [],
        "de_names": [
            ['Finanzdelegation'],
        ],
        "de_abbrev": 'FINDEL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'DélFin': ['DelFin', 'Délfin', 'DELFIN'], 'FINDEL': ['Findel']},
    },
    # ──────────────── EAER ────────────────
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Domain of the Swiss Federal Institutes of Technology', "en_abbrev": 'ETH Domain',
        "fr_names": [
            ['Domaine des écoles polytechniques fédérales', 'domaine des écoles polytechniques fédérales', 'Domaine des Écoles polytechniques fédérales', 'domaine des Écoles polytechniques fédérales', 'Domaine des Ecoles polytechniques fédérales', 'domaine des Ecoles polytechniques fédérales'],
        ],
        "fr_abbrev": 'Domaine des EPF', "fr_alt_abbrevs": [],
        "de_names": [
            # Bereich.s.es der Eidgenössischen Technischen Hochschulen
            [
                'Bereich der Eidgenössischen Technischen Hochschulen',
                'Bereichs der Eidgenössischen Technischen Hochschulen',
                'Bereiches der Eidgenössischen Technischen Hochschulen',
            ],
        ],
        "de_abbrev": 'ETH-Bereich', "de_alt_abbrevs": [],
        # 'ETH-Bereich' est un nom composé : on ajoute aussi son génitif
        "abbrev_case_variants": {'Domaine des EPF': ['domaine des EPF'], 'ETH-Bereich': ['ETH-Bereichs', 'ETH-Bereiches']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Switzerland Tourism', "en_abbrev": 'ST',
        "fr_names": [
            ['Suisse Tourisme'],
        ],
        "fr_abbrev": 'ST', "fr_alt_abbrevs": [],
        "de_names": [
            ['Schweiz Tourismus'],
        ],
        "de_abbrev": 'ST', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Association for Hotel Credit', "en_abbrev": 'SGH',
        "fr_names": [
            ['Société suisse de crédit hôtelier', 'société suisse de crédit hôtelier'],
        ],
        "fr_abbrev": 'SCH', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.n Gesellschaft für Hotelkredit
            [
                'Schweizerische Gesellschaft für Hotelkredit',
                'Schweizerischen Gesellschaft für Hotelkredit',
            ],
        ],
        "de_abbrev": 'SGH', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Export Risk Insurance', "en_abbrev": 'SERV',
        "fr_names": [
            ['Assurance suisse contre les risques à l’exportation', 'assurance suisse contre les risques à l’exportation'],
        ],
        "fr_abbrev": 'ASRE', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.n Exportrisikoversicherung
            [
                'Schweizerische Exportrisikoversicherung',
                'Schweizerischen Exportrisikoversicherung',
            ],
        ],
        "de_abbrev": 'SERV', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'ASRE': ['Asre'], 'SERV': ['Serv']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Federal University for Vocational Education and Training', "en_abbrev": 'SFUVET',
        "fr_names": [
            ['Haute école fédérale en formation professionnelle', 'haute école fédérale en formation professionnelle'],
            ['Institut fédéral des hautes études en formation professionnelle', 'institut fédéral des hautes études en formation professionnelle'],
        ],
        "fr_abbrev": 'HEFP', "fr_alt_abbrevs": ['IFFP'],
        "de_names": [
            # Eidgenössische.n Hochschule für Berufsbildung
            [
                'Eidgenössische Hochschule für Berufsbildung',
                'Eidgenössischen Hochschule für Berufsbildung',
            ],
            # Eidgenössische.s.n Hochschulinstitut.s für Berufsbildung
            [
                'Eidgenössisches Hochschulinstitut für Berufsbildung',
                'Eidgenössische Hochschulinstitut für Berufsbildung',
                'Eidgenössischen Hochschulinstitut für Berufsbildung',
                'Eidgenössischen Hochschulinstituts für Berufsbildung',
            ],
        ],
        "de_abbrev": 'EHB', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'HEFP': ['Hefp'], 'IFFP': ['Iffp']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Federal Institute of Technology Zurich', "en_abbrev": 'ETHZ',
        "fr_names": [
            ['École polytechnique fédérale de Zurich', 'école polytechnique fédérale de Zurich', 'Ecole polytechnique fédérale de Zurich', 'ecole polytechnique fédérale de Zurich'],
        ],
        "fr_abbrev": 'EPFZ', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Technische.n Hochschule Zürich
            [
                'Eidgenössische Technische Hochschule Zürich',
                'Eidgenössischen Technischen Hochschule Zürich',
            ],
        ],
        "de_abbrev": 'ETHZ', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'EPFZ': ['Epfz'], 'ETHZ': ['Ethz']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Federal Institute of Technology Lausanne', "en_abbrev": 'EPFL',
        "fr_names": [
            ['École polytechnique fédérale de Lausanne', 'école polytechnique fédérale de Lausanne', 'Ecole polytechnique fédérale de Lausanne', 'ecole polytechnique fédérale de Lausanne'],
        ],
        "fr_abbrev": 'EPFL', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Technische.n Hochschule Lausanne
            [
                'Eidgenössische Technische Hochschule Lausanne',
                'Eidgenössischen Technischen Hochschule Lausanne',
            ],
        ],
        "de_abbrev": 'ETHL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'EPFL': ['Epfl'], 'ETHL': ['Ethl']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Paul Scherrer Institute', "en_abbrev": 'PSI',
        "fr_names": [
            ['Institut Paul Scherrer', 'institut Paul Scherrer'],
        ],
        "fr_abbrev": 'IPS', "fr_alt_abbrevs": [],
        "de_names": [
            # Paul-Scherrer-Institut.s
            [
                'Paul-Scherrer-Institut',
                'Paul-Scherrer-Instituts',
            ],
        ],
        "de_abbrev": 'PSI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Federal Institute for Forest, Snow and Landscape Research', "en_abbrev": 'WSL',
        "fr_names": [
            ['Institut fédéral de recherches sur la forêt, la neige et le paysage', 'institut fédéral de recherches sur la forêt, la neige et le paysage'],
        ],
        "fr_abbrev": 'FNP', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Forschungsanstalt für Wald, Schnee und Landschaft
            [
                'Eidgenössische Forschungsanstalt für Wald, Schnee und Landschaft',
                'Eidgenössischen Forschungsanstalt für Wald, Schnee und Landschaft',
            ],
        ],
        "de_abbrev": 'WSL', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Federal Laboratories for Materials Science and Technology', "en_abbrev": 'Empa',
        "fr_names": [
            ['Laboratoire fédéral d’essai des matériaux et de recherche', 'laboratoire fédéral d’essai des matériaux et de recherche'],
        ],
        "fr_abbrev": 'LFEM', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Materialprüfungs- und Forschungsanstalt
            [
                'Eidgenössische Materialprüfungs- und Forschungsanstalt',
                'Eidgenössischen Materialprüfungs- und Forschungsanstalt',
            ],
        ],
        "de_abbrev": 'Empa', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'LFEM': ['Lfem'], 'Empa': ['EMPA']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Federal Institute of Aquatic Science and Technology', "en_abbrev": 'EAWAG',
        "fr_names": [
            ['Institut fédéral pour l’aménagement, l’épuration et la protection des eaux', 'institut fédéral pour l’aménagement, l’épuration et la protection des eaux'],
        ],
        "fr_abbrev": 'IFAEPE', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Anstalt für Wasserversorgung, Abwasserreinigung und Gewässerschutz
            [
                'Eidgenössische Anstalt für Wasserversorgung, Abwasserreinigung und Gewässerschutz',
                'Eidgenössischen Anstalt für Wasserversorgung, Abwasserreinigung und Gewässerschutz',
            ],
        ],
        "de_abbrev": 'EAWAG', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'IFAEPE': ['Ifaepe'], 'EAWAG': ['Eawag']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Innovation Agency', "en_abbrev": 'Innosuisse',
        "fr_names": [
            ['Agence suisse pour l’encouragement de l’innovation', 'agence suisse pour l’encouragement de l’innovation'],
        ],
        "fr_abbrev": 'Innosuisse', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.n Agentur für Innovationsförderung
            [
                'Schweizerische Agentur für Innovationsförderung',
                'Schweizerischen Agentur für Innovationsförderung',
            ],
        ],
        "de_abbrev": 'Innosuisse', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'Innosuisse': ['INNOSUISSE', 'innosuisse', 'InnoSuisse']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'EAER',
        "en_name": 'Swiss Investment Fund for Emerging Markets', "en_abbrev": 'SIFEM AG',
        "fr_names": [
            ['Swiss Investment Fund for Emerging Markets'],
        ],
        "fr_abbrev": 'SIFEM SA', "fr_alt_abbrevs": [],
        "de_names": [
            ['Swiss Investment Fund for Emerging Markets'],
        ],
        "de_abbrev": 'SIFEM AG', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'SIFEM SA': ['Sifem SA'], 'SIFEM AG': ['Sifem AG']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'EAER',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Commission de la concurrence', 'commission de la concurrence'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Wettbewerbskommission'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    # ──────────────── DETEC ────────────────
    {
        "type": 'Independent Agency', "upper_level": 'DETEC',
        "en_name": 'Swiss Federal Nuclear Safety Inspectorate', "en_abbrev": 'ENSI',
        "fr_names": [
            ['Inspection fédérale de la sécurité nucléaire', 'inspection fédérale de la sécurité nucléaire'],
        ],
        "fr_abbrev": 'IFSN', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Nuklearsicherheitsinspektorat.s
            [
                'Eidgenössisches Nuklearsicherheitsinspektorat',
                'Eidgenössische Nuklearsicherheitsinspektorat',
                'Eidgenössischen Nuklearsicherheitsinspektorat',
                'Eidgenössischen Nuklearsicherheitsinspektorats',
            ],
        ],
        "de_abbrev": 'ENSI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'IFSN': ['Ifsn'], 'ENSI': ['Ensi']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'DETEC',
        "en_name": 'Waste Disposal Fund for Nuclear Installations', "en_abbrev": 'STENFO',
        "fr_names": [
            ['Fonds de désaffectation et Fonds de gestion pour les installations nucléaires', 'fonds de désaffectation et Fonds de gestion pour les installations nucléaires', 'Fonds de désaffectation et fonds de gestion pour les installations nucléaires', 'fonds de désaffectation et fonds de gestion pour les installations nucléaires'],
        ],
        "fr_abbrev": 'STENFO', "fr_alt_abbrevs": [],
        "de_names": [
            # Stilllegungs- und Entsorgungsfonds für Kernanlagen (génitif identique)
            ['Stilllegungs- und Entsorgungsfonds für Kernanlagen'],
        ],
        "de_abbrev": 'STENFO', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'STENFO': ['Stenfo']},
    },
    {
        "type": 'Independent Agency', "upper_level": 'DETEC',
        "en_name": 'Swiss Capacity Allocation Body', "en_abbrev": 'TVS',
        "fr_names": [
            ['Service suisse d’attribution des sillons', 'service suisse d’attribution des sillons'],
            # CSV : colonne « Alternative Abbr FR »
            ['Service d’attribution des sillons', 'service d’attribution des sillons'],
        ],
        "fr_abbrev": 'SAS', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.n Trassenvergabestelle
            [
                'Schweizerische Trassenvergabestelle',
                'Schweizerischen Trassenvergabestelle',
            ],
            # CSV : colonne « Alternative Abbr DE »
            ['Trassenvergabestelle'],
        ],
        "de_abbrev": 'TVS', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'DETEC',
        "en_name": 'Aircraft Accident Investigation Bureau', "en_abbrev": 'AAIB',
        "fr_names": [
            # CSV : « Bureau d’enquête sur les accidents d’aviation et Bureau
            # d’enquête sur les accidents ferroviaires » -> séparé en deux noms,
            # le nom combiné n'apparaissant presque jamais tel quel dans la presse
            ['Bureau d’enquête sur les accidents d’aviation', 'bureau d’enquête sur les accidents d’aviation'],
            ['Bureau d’enquête sur les accidents ferroviaires', 'bureau d’enquête sur les accidents ferroviaires'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Büro.s für Flugunfalluntersuchungen
            [
                'Büro für Flugunfalluntersuchungen',
                'Büros für Flugunfalluntersuchungen',
            ],
            # Büro.s für Eisenbahnunfalluntersuchungen
            [
                'Büro für Eisenbahnunfalluntersuchungen',
                'Büros für Eisenbahnunfalluntersuchungen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'DETEC',
        "en_name": 'Independent Complaints Authority for Radio and Television', "en_abbrev": 'ICA',
        "fr_names": [
            ['Autorité indépendante d’examen des plaintes en matière de radio-télévision', 'autorité indépendante d’examen des plaintes en matière de radio-télévision'],
        ],
        "fr_abbrev": 'AIEP', "fr_alt_abbrevs": [],
        "de_names": [
            # Unabhängige.n Beschwerdeinstanz für Radio und Fernsehen
            [
                'Unabhängige Beschwerdeinstanz für Radio und Fernsehen',
                'Unabhängigen Beschwerdeinstanz für Radio und Fernsehen',
            ],
        ],
        "de_abbrev": 'UBI', "de_alt_abbrevs": [],
        "abbrev_case_variants": {'AIEP': ['Aiep']},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'DETEC',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Commission fédérale sur les accidents d’aviation', 'commission fédérale sur les accidents d’aviation'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Flugunfallkommission
            [
                'Eidgenössische Flugunfallkommission',
                'Eidgenössischen Flugunfallkommission',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'DETEC',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Commission fédérale de la communication', 'commission fédérale de la communication'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n Kommunikationskommission
            [
                'Eidgenössische Kommunikationskommission',
                'Eidgenössischen Kommunikationskommission',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
    {
        "type": 'OLD Independent Agency', "upper_level": 'DETEC',
        "en_name": '', "en_abbrev": '',
        "fr_names": [
            ['Commission d’arbitrage dans le domaine des chemins de fer', 'commission d’arbitrage dans le domaine des chemins de fer'],
        ],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Schiedskommission im Eisenbahnverkehr'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
        "abbrev_case_variants": {},
    },
]


# ════════════════════════════════════════════════════════════════════════════
# CONSEILLERS FÉDÉRAUX (2000 – aujourd'hui)
# Chaque sous-liste = un nom avec toutes ses formes, la 1re étant la forme
# de référence (identique à src/target_taxonomy.py).
#   - Génitif allemand en -s (« Alain Bersets Departement »). Pas de génitif
#     pour les noms finissant en -s / -ss / -z (Deiss, Dreifuss, Merz, Cassis,
#     Jans) : il s'écrit avec une apostrophe (« Cassis’ »), donc la forme de
#     base suffit.
#   - Seul nom à diacritique : Rösti -> aussi la graphie sans tréma « Roesti ».
# ════════════════════════════════════════════════════════════════════════════
COUNCILLORS = [
    ['Joseph Deiss'],
    ['Ruth Dreifuss'],
    ['Ruth Metzler-Arnold', 'Ruth Metzler-Arnolds'],
    ['Adolf Ogi', 'Adolf Ogis'],
    ['Kaspar Villiger', 'Kaspar Villigers'],
    ['Pascal Couchepin', 'Pascal Couchepins'],
    ['Moritz Leuenberger', 'Moritz Leuenbergers'],
    ['Samuel Schmid', 'Samuel Schmids'],
    ['Micheline Calmy-Rey', 'Micheline Calmy-Reys'],
    ['Christoph Blocher', 'Christoph Blochers'],
    ['Hans-Rudolf Merz'],
    ['Doris Leuthard', 'Doris Leuthards'],
    ['Eveline Widmer-Schlumpf', 'Eveline Widmer-Schlumpfs'],
    ['Ueli Maurer', 'Ueli Maurers'],
    ['Didier Burkhalter', 'Didier Burkhalters'],
    ['Simonetta Sommaruga', 'Simonetta Sommarugas'],
    ['Johann Schneider-Ammann', 'Johann Schneider-Ammanns'],
    ['Alain Berset', 'Alain Bersets'],
    ['Guy Parmelin', 'Guy Parmelins'],
    ['Ignazio Cassis'],
    ['Karin Keller-Sutter', 'Karin Keller-Sutters'],
    ['Viola Amherd', 'Viola Amherds'],
    ['Elisabeth Baume-Schneider', 'Elisabeth Baume-Schneiders'],
    ['Albert Rösti', 'Albert Röstis', 'Albert Roesti', 'Albert Roestis'],
    ['Beat Jans'],
    ['Martin Pfister', 'Martin Pfisters'],
]
