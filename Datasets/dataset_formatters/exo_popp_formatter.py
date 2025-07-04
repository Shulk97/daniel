# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(PARENT_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(PARENT_DIR))))

def get_ne_order(ne_tag):
    ne_order = {
        "Administratif": 1,
        "Mari": 1,
        "Epouse": 1,
        "Témoin": 1,
        "Enfant": 1,
        "Père": 1.1,
        "Mère": 1.1,
        "Ex-epoux": 1.1,
        "Naissance": 2,
        "Résidence": 2,
        "Décès": 2,
        "Divorce": 2,
        "Majeur": 2,
        "Sexe": 2,
        "Registre": 2,
        "Contrat de mariage": 2,
        "Oubli en marge": 2,
        "Lien": 2,
        "Age": 2,
        "Divorce 1ere noce": 2,
        "Legitime": 2,
        "Appel aux armes": 2,
        "Divorce (Transcription)": 2.1,
        "Divorce (Mention)": 2.1,
        "Date": 3,
        "Adresse": 3,
        "Profession": 3,
        "Prenom": 3,
        "Nom": 3,
        "Veuf": 3,
        "Consentant": 3,
        "Non consentant": 3,
        "Absent": 3,
        "Présent": 3,
        "Disparu": 3,
        "Non dénommé": 3,
        "Tribunal": 3,
        "Consentement (présent et consentant)": 3,
        "Consentement (décès)": 3,
        "Consentement (absent mais consentant)": 3,
        "Consentement (disparu)": 3,
        "Consentement (non dénommé)": 3,
        "Pays": 4,
        "Département": 4,
        "Ville": 4,
        "Numéro voie": 4,
        "Type voie": 4,
        "Nom voie": 4,
        "Année": 4,
        "Mois": 4,
        "Jour": 4,
        "Heure": 4,
        "Minute": 4,
    }

    return ne_order[ne_tag]


non_person_tokens = {
    "Naissance": "🏥",
    "Résidence": "🏠",
    "Age": "⌛",
    "Profession": "🔧",
    "Prenom": "💬",
    "Nom": "🗨",
    "Pays": "🏳",
    "Département": "🗺",
    "Ville": "🌇",
    "Numéro voie": "🔟",
    "Type voie": "🛣",
    "Nom voie": "🔠",
    "Année": "🗓",
    "Mois": "📅",
    "Jour": "🌞",
    "Heure": "⏰",
    "Minute": "🕑",
    "Décès": "🪦",
    "Veuf": "😢",
    "Disparu": "🔎",
}

ne_name_to_token = {
    "Administratif": "📖",
    "Mari": "👨",
    "Epouse": "👰",
    "Témoin": "🥸",
    "Père": "👴",
    "Mère": "👵",
    "Ex-epoux": "👹",
    "Naissance": "🏥",
    "Résidence": "🏠",
    "Age": "⌛",
    "Profession": "🔧",
    "Prenom": "💬",
    "Nom": "🗨",
    "Pays": "🏳",
    "Département": "🗺",
    "Ville": "🌇",
    "Numéro voie": "🔟",
    "Type voie": "🛣",
    "Nom voie": "🔠",
    "Année": "🗓",
    "Mois": "📅",
    "Jour": "🌞",
    "Heure": "⏰",
    "Minute": "🕑",
    "Décès": "🪦",
    "Veuf": "😢",
    "Disparu": "🔎",
    "Divorce": "",
    "Majeur": "",
    "Sexe": "",
    "Registre": "",
    "Contrat de mariage": "",
    "Oubli en marge": "",
    "Lien": "",
    "Divorce 1ere noce": "",
    "Legitime": "",
    "Appel aux armes": "",
    "Divorce (Transcription)": "",
    "Divorce (Mention)": "",
    "Date": "",
    "Adresse": "",
    "Consentant": "",
    "Non consentant": "",
    "Absent": "",
    "Présent": "",
    "Non dénommé": "",
    "Tribunal": "",
    "Consentement (présent et consentant)": "",
    "Consentement (décès)": "",
    "Consentement (absent mais consentant)": "",
    "Consentement (disparu)": "",
    "Consentement (non dénommé)": "",
}

ne_name_to_end_token = {
    "Administratif": "📕",
    "Mari": "👦",
    "Epouse": "👧",
    "Témoin": "🧐",
    "Père": "🎩",
    "Mère": "👒",
    "Ex-epoux": "😡",
    "Naissance": "👶",
    "Résidence": "🏡",
    "Age": "⏳",
    "Profession": "🪛",
    "Prenom": "🗯",
    "Nom": "💭",
    "Pays": "🌍",
    "Département": "📌",
    "Ville": "🌉",
    "Numéro voie": "🔢",
    "Type voie": "🛤",
    "Nom voie": "🔡",
    "Année": "🎉",
    "Mois": "📆",
    "Jour": "🌝",
    "Heure": "⌚",
    "Minute": "🕘",
    "Décès": "⚰",
    "Veuf": "😭",
    "Disparu": "🔍",
    "Divorce": "",
    "Majeur": "",
    "Sexe": "",
    "Registre": "",
    "Contrat de mariage": "",
    "Oubli en marge": "",
    "Lien": "",
    "Divorce 1ere noce": "",
    "Legitime": "",
    "Appel aux armes": "",
    "Divorce (Transcription)": "",
    "Divorce (Mention)": "",
    "Date": "",
    "Adresse": "",
    "Consentant": "",
    "Non consentant": "",
    "Absent": "",
    "Présent": "",
    "Non dénommé": "",
    "Tribunal": "",
    "Consentement (présent et consentant)": "",
    "Consentement (décès)": "",
    "Consentement (absent mais consentant)": "",
    "Consentement (disparu)": "",
    "Consentement (non dénommé)": "",
}
# '📖👨👰🥸👴👵👹🏥🏠⌛🔧💬🗨🏳🗺🌇🔟🛣🔠🗓📅🌞⏰🕑🪦😢🔎'
ne_name_to_token_no_hierarchy = {
    "Jour mariage": "📖🌞",
    "Mois mariage": "📖📅",
    "Année mariage": "📖🗓",
    "Heure mariage": "📖⏰",
    "Minute mariage": "📖🕑",
    "Prénom ajdoint": "📖💬",
    "Nom ajdoint": "📖🗨",
    "Ville administrative": "📖🌇",
    "H Prenom": "👨💬",
    "H Nom": "👨🗨",
    "H Profession": "👨🔧",
    "H Ville naissance": "👨🏥🌇",
    "H Departement naissance": "👨🏥🗺",
    "H Pays naissance": "👨🏥🏳",
    "H Jour naissance": "👨🏥🌞",
    "H Mois naissance": "👨🏥📅",
    "H Année naissance": "👨🏥🗓",
    "H Age": "👨⌛",
    "H Ville residence": "👨🏠🌇",
    "H Departement residence": "👨🏠🗺",
    "H Pays residence": "👨🏠🏳",
    "H Numero voie residence": "👨🏠🔟",
    "H Type voie residence": "👨🏠🛣",
    "H Nom voie residence": "👨🏠🔠",
    "H_Pere Prenom": "👨👴💬",
    "H_Pere Nom": "👨👴🗨",
    "H_Pere Profession": "👨👴🔧",
    "H_Pere Ville residence": "👨👴🏠🌇",
    "H_Pere Departement residence": "👨👴🏠🗺",
    "H_Pere Pays residence": "👨👴🏠🏳",
    "H_Pere Numero voie residence": "👨👴🏠🔟",
    "H_Pere Type voie residence": "👨👴🏠🛣",
    "H_Pere Nom voie residence": "👨👴🏠🔠",
    "H_Mere Prenom": "👨👵🗨",
    "H_Mere Nom": "👨👵🏳",
    "H_Mere Profession": "👨👵🔧",
    "H_Mere Ville residence": "👨👵🌇",
    "H_Mere Departement residence": "👨👵🗺",
    "H_Mere Pays residence": "👨👵🏠🏳",
    "H_Mere Numero voie residence": "👨👵🔟",
    "H_Mere Type voie residence": "👨👵🛣",
    "H_Mere Nom voie residence": "👨👵🔠",
    "H Veuf": "👨😢",
    "H Prenom Ex-epoux": "👨💬",
    "H Nom Ex-epoux": "👨🗨",
    "F Prenom": "👰💬",
    "F Nom": "👰🗨",
    "F Profession": "👰🔧",
    "F Ville naissance": "👰🏥🌇",
    "F Departement naissance": "👰🏥🗺",
    "F Pays naissance": "👰🏥🏳",
    "F Jour naissance": "👰🏥🌞",
    "F Mois naissance": "👰🏥📅",
    "F Année naissance": "👰🏥🗓",
    "F Age": "👰⌛",
    "F Ville residence": "👰🏠🌇",
    "F Departement residence": "👰🏠🗺",
    "F Numero voie residence": "👰🏠🔟",
    "F Type voie residence": "👰🏠🛣",
    "F Nom voie residence": "👰🏠🔠",
    "F_Pere Prenom": "👰👴💬",
    "F_Pere Nom": "👰👴🗨",
    "F_Pere Profession": "👰👴🔧",
    "F_Pere Ville residence": "👰👴🏠🌇",
    "F_Pere Departement residence": "👰👴🏠🗺",
    "F_Pere Numero voie residence": "👰👴🏠🔟",
    "F_Pere Type voie residence": "👰👴🏠🛣",
    "F_Pere Nom voie residence": "👰👴🏠🔠",
    "F_Mere Prenom": "👰👵💬",
    "F_Mere Nom": "👰👵🗨",
    "F_Mere Profession": "👰👵🔧",
    "F_Mere Ville residence": "👰👵🏠🌇",
    "F_Mere Departement residence": "👰👵🏠🗺",
    "F_Mere Numero voie residence": "👰👵🏠🔟",
    "F_Mere Type voie residence": "👰👵🏠🛣",
    "F_Mere Nom voie residence": "👰👵🏠🌇",
    "F Veuf": "👰😢",
    "F Prenom Ex-epoux": "👰👹💬",
    "F Nom Ex-epoux": "👰👹🗨",
    "T Prenom": "🥸💬",
    "T Nom": "🥸🗨",
    "T Profession": "🥸🔧",
    "T Age": "🥸⌛",
    "T Numero voie": "🥸🔟",
    "T Type voie": "🥸🛣",
    "T Nom voie": "🥸🔠",
    "T Ville": "🥸🌇",
    "T Departement": "🥸🗺",
    "Deces": "🪦",
}
hierch_emojis_to_unique_emojis = {
    '📖🌞': '🁴',
    '📖📅': '🂂',
    '📖🗓': '🃄',
    '📖⏰': '🆉',
    '📖🕑': '🆀',
    '📖💬': '🃑',
    '📖🗨': '🃌',
    '📖🌇': '🐢',
    '👨💬': '🄑',
    '👨🗨': '🄶',
    '👨🔧': '🄰',
    '👨🏥🌇': '🚑',
    '👨🏥🗺': '🗺',
    '👨🏥🏳': '🏳',
    '👨🏥🌞': '🚀',
    '👨🏥📅': '🔗',
    '👨🏥🗓': '🚲',
    '👨⌛': '⌛',
    '👨🏠🌇': '🛪',
    '👨🏠🗺': '🏓',
    '👨🏠🏳': '📗',
    '👨🏠🔟': '🔟',
    '👨🏠🛣': '🛣',
    '👨🏠🔠': '🔠',
    '👨👴💬': '🍭',
    '👨👴🗨': '🖔',
    '👨👴🔧': '🐮',
    '👨👴🏠🌇': '🎐',
    '👨👴🏠🗺': '🖅',
    '👨👴🏠🏳': '🚔',
    '👨👴🏠🔟': '🌥',
    '👨👴🏠🛣': '🏰',
    '👨👴🏠🔠': '🍂',
    '👨👵💬': '💯',
    '👨👵🗨': '🐓',
    '👨👵🔧': '🕈',
    '👨👵🏠🌇': '🏶',
    '👨👵🏠🗺': '🐃',
    '👨👵🏠🏳': '😠',
    '👨👵🏠🔟': '🕎',
    '👨👵🏠🛣': '🕍',
    '👨👵🏠🔠': '🍽',
    '👨👵🗨🪦': '🍲',
    '👨👴🗨🪦': '🌲',
    '👨👵👴🗨🪦': '🌈',
    '👨👵💬🪦': '🄻',
    '👨👴💬🪦': '🄾',
    '👨👵🗨🔎': '🐱',
    '👨👴🗨🔎': '🐑',
    '👨👵👴🗨🔎': '🐕',
    '👨👵💬🔎': '🐁',
    '👨👴💬🔎': '🐋',
    '👨👵👴🔧': '😍',
    '👨👵👴🏠🔟': '📺',
    '👨👵👴🏠🌇': '🌴',
    '👨👵👴🏠🛣': '🠱',
    '👨👵👴🏠🔠': '🠔',
    '👨👵👴🏠🗺': '🍎',
    '👨👵👴🏠🏳': '🔒',
    '👨💬😢': '👋',
    '👨🗨😢': '🍆',
    '👰💬': '🖱',
    '👰🗨': '🖻',
    '👰🔧': '🌌',
    '👰🏥🌇': '🎤',
    '👰🏥🗺': '🍞',
    '👰🏥🏳': '👘',
    '👰🏥🌞': '🗴',
    '👰🏥📅': '🚾',
    '👰🏥🗓': '🍗',
    '👰⌛': '😚',
    '👰🏠🌇': '📎',
    '👰🏠🗺': '👼',
    '👰🏠🔟': '🖶',
    '👰🏠🛣': '🎁',
    '👰🏠🔠': '🌐',
    '👰👴💬': '🍜',
    '👰👴🗨': '💈',
    '👰👴🔧': '🙈',
    '👰👴🏠🌇': '😶',
    '👰👴🏠🗺': '🚳',
    '👰👴🏠🏳': '🧣',
    '👰👴🏠🔟': '🐡',
    '👰👴🏠🛣': '🙃',
    '👰👴🏠🔠': '🌂',
    '👰👵💬': '🎸',
    '👰👵🗨': '🔂',
    '👰👵🔧': '🎠',
    '👰👵🏠🌇': '🌠',
    '👰👵🏠🗺': '🗘',
    '👰👵🏠🔟': '🔑',
    '👰👵🏠🛣': '🕮',
    '👰👵🏠🔠': '🏉',
    '👰👵🏠🏳': '👔',
    '👰😢': '🚌',
    '👰💬😢': '🥑',
    '👰🗨😢': '🥦',
    '👰👵🗨🪦': '🔱',
    '👰👴🗨🪦': '💦',
    '👰👵👴🗨🪦': '🎊',
    '👰👵💬🪦': '🅆',
    '👰👴💬🪦': '🅍',
    '👰👵🗨🔎': '💚',
    '👰👴🗨🔎': '💛',
    '👰👵👴🗨🔎': '💞',
    '👰👵💬🔎': '🐙',
    '👰👴💬🔎': '🐅',
    '👰👵👴🔧': '🙀',
    '👰👵👴🏠🌇': '🛌',
    '👰👵👴🏠🏳': '💣',
    '👰👵👴🏠🔟': '🔥',
    '👰👵👴🏠🛣': '🠳',
    '👰👵👴🏠🔠': '🠖',
    '👰👹💬': '🖦',
    '👰👹🗨': '🎼',
    '👰👵👹🗨':'🎼',
    '👰🗺':'👠',
    '👨👰🗨':'🍁',
    '🥸💬': '🎺',
    '🥸🗨': '🍒',
    '🥸🔧': '🎜',
    '🥸⌛': '🖘',
    '🥸🏠🔟': '🔍',
    '🥸🏠🛣': '🖢',
    '🥸🏠🔠': '🏁',
    '🥸🏠🌇': '😔',
    '🥸🌇': '😔',
    '🥸🏠🗺': '🖺',
    '🥸🗺': '🖺',
    '🌞': '🌞',
    '📅': '📅',
    '🗓': '🗓',
    '🌇': '🌇',
    '🔧': '🔧',
    '💬': '💬',
    '🗨': '🗨',
    '🗺': '🗺',
    '⏰': '⏰',
    '🕑': '🕑',
    '🌍':'🌍',
    '👰👵💬😢':'🇦',
    '👰👵🗨😢':'🇧',
    '👰👴🪦':'🇨',
    '👰👵🪦':'🇩',
    '👰👵👴🪦':'🇪',
    '👰👵😢':'🇬',
    '👰👴😢':'🇭',
    '👨👹💬':'🇮',
    '👨👹🗨':'🇯',
    '👨👵💬😢':'🇰',
    '👨👵🗨😢':'🇱',
    '👨👵🪦':'🇲',
    '👨👴🪦':'🇳',
    '👨👴👵🪦':'🇴',
    '👨😢':'🇵',
    '👨👵😢':'🇶',
    '👨👴😢':'🇷',
    '👰👴🔎':'🇸',
    '👰👴💬😢':'🇹',
    '👰👴🗨😢':'🇺',
    '👰👵🏠🗺👴':'🇻',
    '👰👴🔎':'🇼',
    '👰🏥🛣':'🇽',
    '🏠🔟':'🇾',
    '🏠🛣': '🇿',
    '🏠🔠': '💢',
    '👨👴💬😢':'👖',
    '👨👹💬🪦':'⚓',
    '👰👹💬🪦':'🧣',
    '👰👴💬😢':'🧤',
    '👰🏥🔟':'💲',
    '👰🏥🔠':'👕',
    '👰👵👹💬':'😀',
    '💬😢':'😁',
    '🗨👹':'😎'
}
ii={}
for k, v in hierch_emojis_to_unique_emojis.items():
    ii[''.join(sorted([s for s in k]))] = v

hierch_emojis_to_unique_emojis=ii

auto_corrections_charset = {
    "   ": " ",
    " ": " ",
    "«": '"',
    "¨": "*",
    "—": "-",
    "_": "-",
    "‘": '"',
    "…": "...",
    " = ": " ",
    "Î": "I",
    "$": "",
    "+": "",
    "Â": "A",
    "Ç": "C",
    "È": "E",
    "É": "E",
    "Ê": "E",
    "Ë": "E",
    "Ö": "O",
    "Ü": "U",
}

# Layout begin-token to end-token
SEM_MATCHING_TOKENS = {
    "ⓟ": "Ⓟ",  # page
    "ⓜ": "Ⓜ",  # marriage license (can include the tokens ⓝ, ⓑ and ⓘ)
    "ⓝ": "Ⓝ",  # margin-names
    "ⓑ": "Ⓑ",  # paragraph (body of a marriage license)
    "ⓘ": "Ⓘ",  # annotation, marge infos
    "ⓢ": "Ⓢ",  # signature (currently not used)
}

EXO_POPP_MULTI_MATCHING_TOKENS = {
    'ⓗ': 'Ⓗ', # paragraph ExoPOPP
    'ⓙ': 'Ⓙ',# margin-names M-POPP
    "ⓘ": "Ⓘ",# margin-infos ExoPOPP
    'ⓜ': 'Ⓜ',# marriage act ExoPOPP
    'ⓚ': 'Ⓚ',# page ExoPOPP
}

rpl_dict = {
    "📖": "admin",  # admin
    "👨": "husband",  # husband
    "👰": "bride",  # bride
    "🥸": "witness",  # witness
    "👴": "father",  # father
    "👵": "mother",  # mother
    "💔": "ex-husband",  # ex-husband
    "🏥": "birth",  # birth
    "🏠": "residence",  # residence
    "⌛": "age",  # age
    "🔧": "job",  # job/profession
    "💬": "first-name",  # first name
    "🗨": "family-name",  # family name
    "🏳": "country",  # country
    "🗺": "department",  # department
    "🌇": "city",  # city
    "🔟": "street-number",  # street number
    "🛣": "street-type",  # street type
    "🔠": "street-name",  # street name
    "🗓": "year",  # year
    "📅": "month",  # month
    "🌞": "day",  # day
    "⏰": "hour",  # hour
    "🕑": "minute",  # minute
}

MATCHING_NAMED_ENTITY_TOKENS = {
    "📖": "📖",  # admin
    "👨": "👨",  # husband
    "👰": "👰",  # bride
    "🥸": "🥸",  # witness
    "👴": "👴",  # father
    "👵": "👵",  # mother
    "👹": "👹",  # ex-husband
    "🏥": "🏥",  # birth
    "🏠": "🏠",  # residence
    "⌛": "⌛",  # age
    "🔧": "🔧",  # job/profession
    "💬": "💬",  # first name
    "🗨": "🗨",  # family name
    "🏳": "🏳",  # country
    "🗺": "🗺",  # department
    "🌇": "🌇",  # city
    "🔟": "🔟",  # street number
    "🛣": "🛣",  # street type
    "🔠": "🔠",  # street name
    "🗓": "🗓",  # year
    "📅": "📅",  # month
    "🌞": "🌞",  # day
    "⏰": "⏰",  # hour
    "🕑": "🕑",  # minute
    "🪦": "🪦",  # death
    "😢": "😢",  # widow/widower
    "🔎": "🔎",  # missing
}

MATCHING_NAMED_ENTITY_TOKENS_BEGIN_END = {
    "📖": "📕",  # admin
    "👨": "👦",  # husband
    "👰": "👧",  # bride
    "🥸": "🧐",  # witness
    "👴": "🎩",  # father
    "👵": "👒",  # mother
    "👹": "😡",  # ex-husband
    "🏥": "👶",  # birth
    "🏠": "🏡",  # residence
    "⌛": "⏳",  # age
    "🔧": "🪛",  # job/profession
    "💬": "🗯",  # first name
    "🗨": "💭",  # family name
    "🏳": "🌍",  # country
    "🗺": "📌",  # department
    "🌇": "🌉",  # city
    "🔟": "🔢",  # street number
    "🛣": "🛤",  # street type
    "🔠": "🔡",  # street name
    "🗓": "🎉",  # year
    "📅": "📆",  # month
    "🌞": "🌝",  # day
    "⏰": "⌚",  # hour
    "🕑": "🕘",  # minute
    "🪦": "⚰",  # death
    "😢": "😭",  # widow/widower
    "🔎": "🔍",  # missing
}


MATCHING_NAMED_ENTITY_TOKENS_NO_HIERARCHY = {
    '🁴': '🁴',
    '🂂': '🂂',
    '🃄': '🃄',
    '🆉': '🆉',
    '🆀': '🆀',
    '🃑': '🃑',
    '🃌': '🃌',
    '🐢': '🐢',
    '🄑': '🄑',
    '🄶': '🄶',
    '🄰': '🄰',
    '🚑': '🚑',
    '🗺': '🗺',
    '🏳': '🏳',
    '🚀': '🚀',
    '🔗': '🔗',
    '🚲': '🚲',
    '⌛': '⌛',
    '🛪': '🛪',
    '🏓': '🏓',
    '📗': '📗',
    '🔟': '🔟',
    '🛣': '🛣',
    '🔠': '🔠',
    '🍭': '🍭',
    '🖔': '🖔',
    '🐮': '🐮',
    '🎐': '🎐',
    '🖅': '🖅',
    '🚔': '🚔',
    '🌥': '🌥',
    '🏰': '🏰',
    '🍂': '🍂',
    '💯': '💯',
    '🐓': '🐓',
    '🕈': '🕈',
    '🏶': '🏶',
    '🐃': '🐃',
    '😠': '😠',
    '🕎': '🕎',
    '🕍': '🕍',
    '🍽': '🍽',
    '🍲': '🍲',
    '🌲': '🌲',
    '🌈': '🌈',
    '🄻': '🄻',
    '🄾': '🄾',
    '🐱': '🐱',
    '🐑': '🐑',
    '🐕': '🐕',
    '🐁': '🐁',
    '🐋': '🐋',
    '😍': '😍',
    '📺': '📺',
    '🌴': '🌴',
    '🠱': '🠱',
    '🠔': '🠔',
    '🍎': '🍎',
    '🔒': '🔒',
    '👋': '👋',
    '🍆': '🍆',
    '🖱': '🖱',
    '🖻': '🖻',
    '🌌': '🌌',
    '🎤': '🎤',
    '🍞': '🍞',
    '👘': '👘',
    '🗴': '🗴',
    '🚾': '🚾',
    '🍗': '🍗',
    '😚': '😚',
    '📎': '📎',
    '👼': '👼',
    '🖶': '🖶',
    '🎁': '🎁',
    '🌐': '🌐',
    '🍜': '🍜',
    '💈': '💈',
    '🙈': '🙈',
    '😶': '😶',
    '🚳': '🚳',
    '🧣': '🧣',
    '🐡': '🐡',
    '🙃': '🙃',
    '🌂': '🌂',
    '🎸': '🎸',
    '🔂': '🔂',
    '🎠': '🎠',
    '🌠': '🌠',
    '🗘': '🗘',
    '🔑': '🔑',
    '🕮': '🕮',
    '🏉': '🏉',
    '👔': '👔',
    '🚌': '🚌',
    '🥑': '🥑',
    '🥦': '🥦',
    '🔱': '🔱',
    '💦': '💦',
    '🎊': '🎊',
    '🅆': '🅆',
    '🅍': '🅍',
    '💚': '💚',
    '💛': '💛',
    '💞': '💞',
    '🐙': '🐙',
    '🐅': '🐅',
    '🙀': '🙀',
    '🛌': '🛌',
    '💣': '💣',
    '🔥': '🔥',
    '🠳': '🠳',
    '🠖': '🠖',
    '🖦': '🖦',
    '🎼': '🎼',
    '👠': '👠',
    '🍁': '🍁',
    '🎺': '🎺',
    '🍒': '🍒',
    '🎜': '🎜',
    '🖘': '🖘',
    '🔍': '🔍',
    '🖢': '🖢',
    '🏁': '🏁',
    '😔': '😔',
    '🖺': '🖺',
    '🌞': '🌞',
    '📅': '📅',
    '🗓': '🗓',
    '🌇': '🌇',
    '🔧': '🔧',
    '💬': '💬',
    '🗨': '🗨',
    '⏰': '⏰',
    '🕑': '🕑',
    '🌍': '🌍'
}
MATCHING_NAMED_ENTITY_TOKENS_NO_HIERARCHY = {
    '🁴': '🁴',
    '🂂': '🂂',
    '🃄': '🃄',
    '🆉': '🆉',
    '🆀': '🆀',
    '🃑': '🃑',
    '🃌': '🃌',
    '🐢': '🐢',
    '🄑': '🄑',
    '🄶': '🄶',
    '🄰': '🄰',
    '🚑': '🚑',
    '��': '��',
    '🏳': '🏳',
    '🚀': '🚀',
    '🔗': '🔗',
    '🚲': '🚲',
    '⌛': '⌛',
    '🛪': '��',
    '🏓': '🏓',
    '📗': '📗',
    '🔟': '🔟',
    '🛣': '🛣',
    '🔠': '🔠',
    '🍭': '🍭',
    '🖔': '🖔',
    '🐮': '🐮',
    '🎐': '🎐',
    '🖅': '🖅',
    '🚔': '🚔',
    '🌥': '🌥',
    '🏰': '🏰',
    '🍂': '🍂',
    '💯': '💯',
    '🕈': '🕈',
    '🏶': '🏶',
    '🐃': '🐃',
    '😠': '😠',
    '🕎': '🕎',
    '🕍': '🕍',
    '🍽': '🍽',
    '🍲': '🍲',
    '🌲': '��',
    '🌈': '🌈',
    '🄻': '🄻',
    '🄾': '🄾',
    '🐱': '🐱',
    '🐑': '🐑',
    '🐕': '🐕',
    '🐁': '🐁',
    '🐋': '🐋',
    '😍': '😍',
    '📺': '📺',
    '🌴': '🌴',
    '🠱': '🠱',
    '🠔': '🠔',
    '🍎': '🍎',
    '🔒': '🔒',
    '🍆': '🍆',
    '🖱': '🖱',
    '🖻': '🖻',
    '🌌': '🌌',
    '🎤': '🎤',
    '🍞': '🍞',
    '👘': '👘',
    '🗴': '🗴',
    '🚾': '��',
    '🍗': '🍗',
    '😚': '😚',
    '📎': '📎',
    '👼': '👼',
    '🖶': '🖶',
    '🎁': '🎁',
    '🌐': '🌐',
    '🍜': '🍜',
    '💈': '💈',
    '🙈': '🙈',
    '😶': '😶',
    '🚳': '🚳',
    '🧣': '🧣',
    '🐡': '🐡',
    '🙃': '🙃',
    '🎸': '🎸',
    '🔂': '🔂',
    '🎠': '🎠',
    '🌠': '🌠',
    '🗘': '🗘',
    '🔑': '🔑',
    '🕮': '🕮',
    '🏉': '🏉',
    '👔': '��',
    '🚌': '🚌',
    '🥑': '🥑',
    '🥦': '🥦',
    '🔱': '🔱',
    '💦': '💦',
    '🎊': '🎊',
    '🅆': '🅆',
    '🅍': '🅍',
    '💚': '💚',
    '💛': '💛',
    '💞': '💞',
    '🐙': '🐙',
    '🐅': '🐅',
    '🙀': '🙀',
    '🛌': '🛌',
    '🔥': '🔥',
    '🠳': '🠳',
    '🠖': '🠖',
    '🖦': '🖦',
    '🎼': '🎼',
    '👠': '👠',
    '🍁': '🍁',
    '🎺': '🎺',
    '🍒': '🍒',
    '🎜': '🎜',
    '🖘': '🖘',
    '🔍': '🔍',
    '🖢': '🖢',
    '🏁': '🏁',
    '😔': '😔',
    '🖺': '🖺',
    '🌞': '🌞',
    '📅': '📅',
    '🗓': '🗓',
    '🌇': '🌇',
    '🔧': '🔧',
    '💬': '💬',
    '🗨': '🗨',
    '⏰': '⏰',
    '🕑': '🕑',
    '🌍': '🌍',
    '🇦': '🇦',
    '🇧': '🇧',
    '🇨': '🇨',
    '🇩': '🇩',
    '🇪': '🇪',
    '🇬': '🇬',
    '🇭': '🇭',
    '🇮': '🇮',
    '🇯': '🇯',
    '🇰': '🇰',
    '🇱': '🇱',
    '🇲': '🇲',
    '🇳': '🇳',
    '🇴': '🇴',
    '🇵': '🇵',
    '🇶': '🇶',
    '🇷': '🇷',
    '🇼': '🇼',
    '🧤': '🧤',
    '🇺': '🇺',
    '🇻': '🇻',
    '🇽': '🇽',
    '🇾': '🇾',
    '🇿': '🇿',
    '💢': '💢',
    '👕': '👕',
    '⚓': '⚓',
    '👖': '👖',
    '💲': '💲',
    '🧣': '🧣',
    '😁': '😁',
    '😎': '😎'
}

MATCHING_NAMED_ENTITY_TOKENS_NO_HIERARCHY = {'🁴': '🁴', '🂂': '🂂', '🃄': '🃄', '🆉': '🆉', '🆀': '🆀', '🃑': '🃑', '🃌': '🃌', '🐢': '🐢', '🄑': '🄑', '🄶': '🄶', '🄰': '🄰', '🚑': '🚑', '🗺': '🗺', '🏳': '🏳', '🚀': '🚀', '🔗': '🔗', '🚲': '🚲', '⌛': '⌛', '🛪': '🛪', '🏓': '🏓', '📗': '📗', '🔟': '🔟', '🛣': '🛣', '🔠': '🔠', '🍭': '🍭', '🖔': '🖔', '🐮': '🐮', '🎐': '🎐', '🖅': '🖅', '🚔': '🚔', '🌥': '🌥', '🏰': '🏰', '🍂': '🍂', '💯': '💯', '🐓': '🐓', '🕈': '🕈', '🏶': '🏶', '🐃': '🐃', '😠': '😠', '🕎': '🕎', '🕍': '🕍', '🍽': '🍽', '🍲': '🍲', '🌲': '🌲', '🌈': '🌈', '🄻': '🄻', '🄾': '🄾', '🐱': '🐱', '🐑': '🐑', '🐕': '🐕', '🐁': '🐁', '🐋': '🐋', '😍': '😍', '📺': '📺', '🌴': '🌴', '🠱': '🠱', '🠔': '🠔', '🍎': '🍎', '🔒': '🔒', '👋': '👋', '🍆': '🍆', '🖱': '🖱', '🖻': '🖻', '🌌': '🌌', '🎤': '🎤', '🍞': '🍞', '👘': '👘', '🗴': '🗴', '🚾': '🚾', '🍗': '🍗', '😚': '😚', '📎': '📎', '👼': '👼', '🖶': '🖶', '🎁': '🎁', '🌐': '🌐', '🍜': '🍜', '💈': '💈', '🙈': '🙈', '😶': '😶', '🚳': '🚳', '🧣': '🧣', '🐡': '🐡', '🙃': '🙃', '🌂': '🌂', '🎸': '🎸', '🔂': '🔂', '🎠': '🎠', '🌠': '🌠', '🗘': '🗘', '🔑': '🔑', '🕮': '🕮', '🏉': '🏉', '👔': '👔', '🚌': '🚌', '🥑': '🥑', '🥦': '🥦', '🔱': '🔱', '💦': '💦', '🎊': '🎊', '🅆': '🅆', '🅍': '🅍', '💚': '💚', '💛': '💛', '💞': '💞', '🐙': '🐙', '🐅': '🐅', '🙀': '🙀', '🛌': '🛌', '💣': '💣', '🔥': '🔥', '🠳': '🠳', '🠖': '🠖', '🖦': '🖦', '🎼': '🎼', '👠': '👠', '🍁': '🍁', '🎺': '🎺', '🍒': '🍒', '🎜': '🎜', '🖘': '🖘', '🔍': '🔍', '🖢': '🖢', '🏁': '🏁', '😔': '😔', '🖺': '🖺', '🌞': '🌞', '📅': '📅', '🗓': '🗓', '🌇': '🌇', '🔧': '🔧', '💬': '💬', '🗨': '🗨', '⏰': '⏰', '🕑': '🕑', '🌍': '🌍', '🇦': '🇦', '🇧': '🇧', '🇨': '🇨', '🇩': '🇩', '🇪': '🇪', '🇬': '🇬', '🇭': '🇭', '🇮': '🇮', '🇯': '🇯', '🇰': '🇰', '🇱': '🇱', '🇲': '🇲', '🇳': '🇳', '🇴': '🇴', '🇵': '🇵', '🇶': '🇶', '🇷': '🇷', '🇼': '🇼', '🧤': '🧤', '🇺': '🇺', '🇻': '🇻', '🇽': '🇽', '🇾': '🇾', '🇿': '🇿', '💢': '💢', '👕': '👕', '⚓': '⚓', '👖': '👖', '💲': '💲', '🧣': '🧣', '😀': '😀', '😁': '😁', '😎': '😎'}

EVERY_NE_TOKENS = {**MATCHING_NAMED_ENTITY_TOKENS_BEGIN_END, **MATCHING_NAMED_ENTITY_TOKENS_NO_HIERARCHY}

UNWANTED_TOKENS = {'👨'}

type_to_token = {"paragraphe": ("ⓑ", "Ⓑ"), "marge_noms": ("ⓝ", "Ⓝ"), "marge_info": ("ⓘ", "Ⓘ")}


def get_charset(labels_dict):
    charset = set()
    for split_name, split_dict in labels_dict.items():
        for page_name, page_dict in split_dict.items():
            for content_id, content_dict in page_dict["contents"].items():
                for region_id, region_dict in content_dict.items():
                    charset = charset.union(set(region_dict["label"]))

    return charset
