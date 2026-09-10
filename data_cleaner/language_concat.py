# -*- coding: utf-8 -*-
"""language_concat.py

Consolidates, cleans, balances, and shuffles African language datasets into a unified
multilingual corpus file (languages.txt) for tokenizer training:

Pipeline:
1. Data Sanitization:
   - Removes URLs, web links, handles (@user), and hashtags (#tag)
   - Cleans spam, promotional CTAs, platform campaign tags, and CamelCase compounds
   - Strips PII (emails, phone numbers, IP addresses, credit cards, national IDs)
   - Removes placeholders & tags ("image not found", "irl not found", HTML/XML)
   - Removes standalone/isolated numbers while preserving Arabizi words (e.g. '3amer', '7abibi')
     and words with numbers (e.g. 'COVID-19')
   - Standardizes and collapses spaces

2. Multilingual Sentence Delimitation:
   - Segments cleaned text into discrete sentences delimited by punctuation marks across all languages:
     * Latin / African Latin: . ! ? …
     * Ethiopic (Amharic, Tigrinya, Ge'ez, Tigre): ። (U+1362), ፧ (U+1367), ፠ (U+1368)
     * Arabic script (Maghrebi, Arabic, Mauritanian, Chadian): ؟ (U+061F), ۔ (U+06D4)
     * Keeps terminal punctuation and trailing closing quotes/brackets attached to each sentence
     * Words inside each individual sentence are preserved in their original order

3. Balanced Sampling & Global Shuffling:
   - Identifies the minimum sentence count found across all valid dataset files (e.g. 20,000)
   - Randomly samples exactly that minimum count of sentences from each dataset file
   - Globally shuffles the sentences across all languages (words inside sentences are preserved)
   - Writes the resulting balanced corpus into languages.txt

Usage:
    python language_concat.py
    python language_concat.py --input-dir . --output-dir . --output-file languages.txt
"""

import html
import os
import random
import re
import sys
import unicodedata
import argparse
from collections import defaultdict

# ============================================================================
# 1. CENTRALIZED CONFIGURATION & 10 LINGUISTIC ROOT FAMILIES
# ============================================================================

ENCODING_UTF8 = "utf-8"
ERROR_HANDLER_REPLACE = "replace"
FILE_MODE_READ = "r"
FILE_MODE_WRITE = "w"
TXT_EXTENSION = ".txt"

# 10 Target Linguistic Root Family Filenames
FAMILY_BANTOU = "bantou.txt"
FAMILY_SEMETIQUE = "semetique.txt"
FAMILY_MAGHREBI = "maghrebi.txt"
FAMILY_TCHADIQUE = "tchadique.txt"
FAMILY_COUCHITIQUE = "couchitique.txt"
FAMILY_KWA = "kwa.txt"
FAMILY_MANDE = "mande.txt"
FAMILY_SAHARIEN = "saharien.txt"
FAMILY_PIDGINS = "pidgins.txt"
FAMILY_KHOISAN = "khoisan.txt"

# Comprehensive Keyword and Pattern Rules mapping dataset names / aliases / ISO codes to families
FAMILY_PATTERNS: dict[str, list[str]] = {
    FAMILY_BANTOU: [
        # Swahili / Kiswahili
        "kiswahili", "swahili", "shiswahili", "swh", "sw", "swh_latn", "swhlatn",
        # Shona
        "chshona", "shona", "chishona", "chi_shona", "sna", "sn", "sna_latn", "snalatn",
        # isiXhosa / Xhosa
        "isixhosa", "xhosa", "isi_xhosa", "xho", "xh", "xho_latn", "xholatn", "xosa",
        # isiZulu / Zulu
        "isizulu", "zulu", "isi_zulu", "zoulou", "zul", "zu", "zul_latn", "zullatn",
        # Lingala
        "lingala", "ngala", "lin", "ln", "lin_latn", "linlatn",
        # Luganda
        "luganda", "ganda", "oluganda", "lug", "lg", "lug_latn", "luglatn",
        # Kinyarwanda / Rwanda
        "kinyarwanda", "rwanda", "ikinyarwanda", "kin", "rw", "kin_latn", "kinlatn",
        # Rundi / Kirundi
        "rundi", "kirundi", "ikirundi", "run", "rn", "run_latn", "runlatn",
        # Xitsonga / Tsonga / Xithonga
        "xitsonga", "tsonga", "xithonga", "thonga", "chitsonga", "tso", "ts", "tso_latn", "tsolatn",
        # Additional Bantu varieties & dialects
        "sotho", "sesotho", "sot", "st", "nso", "nso_latn", "nsolatn",
        "tswana", "setswana", "tsn", "tn", "tsn_latn", "tsnlatn",
        "venda", "tshivenda", "ven", "ve",
        "chewa", "chichewa", "nyanja", "chinyanja", "nya", "nya_latn", "nyalatn",
        "bemba", "cibemba", "bem",
        "kikuyu", "gikuyu", "kik", "kik_latn", "kiklatn",
        "lozi", "ndebele", "isindebele", "nbl",
        "swati", "siswati", "ssw", "ssw_latn", "sswlatn",
        "kongo", "kikongo", "kon", "kon_latn", "konlatn",
        "umbundu", "umb", "umb_latn", "umblatn",
        "kimbundu", "kmb", "kmb_latn", "kmblatn",
        "tumbuka", "tum", "tum_latn", "tumlatn",
        "bantu", "bantou",
    ],
    FAMILY_SEMETIQUE: [
        # Amharic
        "amharic", "amharique", "amarinya", "amhara", "amh", "am", "amh_ethi", "amhethi",
        # Tigrinya
        "tigrinya", "tigrigna", "tir", "ti", "tir_ethi", "tirethi",
        # Ge'ez
        "geez", "ge'ez", "ge_ez", "giiz", "gez",
        # Tigre
        "tigre", "tigré", "tigrayit", "x-tigre", "tig",
        # Additional Semitic varieties
        "gurage", "sebatbet", "harari", "silt'e", "silte", "argobba", "inor",
        "semetique", "semitique", "semitic",
    ],
    FAMILY_MAGHREBI: [
        # Moroccan Arabic / Darija
        "morocco", "moroccan", "morocan", "maroc", "marocain", "darija", "moroccan_arabic", "moroccan_darija", "ary", "ary_arab", "aryarab",
        # Algerian Arabic
        "algeria", "algerian", "algerie", "algerien", "algerian_arabic", "algerian_darija", "arq", "arq_arab", "arqarab",
        # Tunisian Arabic
        "tunisia", "tunisian", "tunisie", "tunisien", "tunisian_arabic", "aeb", "aeb_arab", "aebarab",
        # Egyptian Arabic / Masri
        "egypt", "egyptian", "egypte", "egyptien", "masri", "egyptian_arabic", "arz", "arz_arab", "arzarab",
        # Libyan Arabic
        "libya", "libyan", "libye", "libyen", "libyan_arabic", "ayl",
        # Sudanese Arabic
        "sudan", "sudanese", "soudan", "soudanais", "sudanese_arabic", "apd",
        # Mauritanian Arabic / Hassaniya
        "mauritania", "mauritanian", "mauritanie", "mauritanien", "hassaniya", "hassaniyya", "mey",
        # Chadian Arabic
        "chad", "chadian", "tchad", "shu",
        # Standard Arabic & General
        "modern_standard_arabic", "standard_arabic", "modernstandardarabic", "standardarabic", "msa", "fusha", "arb", "arb_arab", "arbarab",
        "arabic", "arabe", "ara", "ar",
        "arabizi", "arabzi", "north_african_arabizi", "northafricanarabizi", "darija_arabizi", "araby",
        "maghribi", "maghrebi",
    ],
    FAMILY_TCHADIQUE: [
        # Hausa
        "hausa", "haoussa", "hawsa", "hau", "ha", "hau_latn", "haulatn",
        # Bura
        "bura", "bura_pabir", "burapabir", "pabir", "bwr",
        # Mafa
        "mafa", "mofa", "matakam", "maf",
        # Additional Chadic varieties
        "kotoko", "marghi", "margi", "tera", "bade", "ngizim", "karekare", "mandara", "mnd",
        "tchadique", "chadic",
    ],
    FAMILY_COUCHITIQUE: [
        # Oromo
        "oromo", "afaan_oromo", "afan_oromo", "afaanoromo", "afanoromo", "oromoo", "orm", "om", "gaz", "gaz_latn", "gazlatn",
        # Somali
        "somali", "soomaali", "somalia", "som", "so", "som_latn", "somlatn",
        # Afar
        "afar", "qafar", "afar_af", "afaraf", "aar", "aa", "aar_latn", "aarlatn",
        # Sidama
        "sidama", "sidaama", "sidaamu", "sid",
        # Additional Cushitic varieties
        "beja", "bedawiyet", "bej", "hadiyya", "hdy", "kambaata", "ktb", "agaw", "bilen", "byn",
        "alaaba", "gawwada", "konso", "rendille", "rel", "iraqw", "irk", "burji", "daasanach",
        "couchitique", "cushitic",
    ],
    FAMILY_KWA: [
        # Yoruba
        "yoruba", "yorùbá", "yorouba", "èkìtì", "ekiti", "yor", "yo", "yor_latn", "yorlatn",
        # Igbo
        "igbo", "ibo", "asusu_igbo", "asusuigbo", "ibo_latn", "ibolatn", "ig",
        # Akan / Twi / Fante
        "akan", "twi", "asante", "fante", "asante_twi", "asantetwi", "fanti", "akuapem", "aka", "ak", "aka_latn", "akalatn", "twi_latn", "twilatn",
        # Ewe
        "ewe", "èwè", "evegbé", "ewe_latn", "ewelatn", "ee",
        # Fon
        "fon", "fongbe", "fon_gbe", "fon_latn", "fonlatn",
        # Baoulé
        "baoule", "baoulé", "bawule", "bci",
        # Edo
        "edo", "bini", "bin",
        # Tiv
        "tiv", "tiv_latn", "tivlatn",
        # Additional Kwa varieties
        "ga", "gaa", "dangme", "adja", "aja", "anyin", "any", "nzema", "nzi", "ebrie", "abron",
        "kwa",
    ],
    FAMILY_MANDE: [
        # Wolof
        "wolof", "ouolof", "wol", "wo", "wol_latn", "wollatn",
        # Bambara
        "bambara", "bamanankan", "bamana", "bam", "bm", "bam_latn", "bamlatn",
        # Fula / Fulfulde / Pulaar / Peul
        "fula", "fulfulde", "pulaar", "pular", "peul", "fulani", "fulbe", "ful", "ff", "fuv", "fuv_latn", "fuvlatn", "fuc",
        # Mandinka
        "mandinka", "mandingo", "manding", "mnk",
        # Dyula
        "dyula", "dioula", "jula", "dyu", "dyu_latn", "dyulatn",
        # Serer
        "serer", "seereer", "sereer", "srr",
        # Additional Mande varieties
        "soninke", "snk", "mende", "men", "malinke", "maninka", "mlq", "susu", "sousson", "sus", "kpelle", "kpe", "vai",
        "mande", "mandé",
    ],
    FAMILY_SAHARIEN: [
        # Dholuo / Luo
        "dholuo", "luo", "luo_latn", "luolatn",
        # Dinka
        "dinka", "thuongjang", "thuɔŋjäŋ", "jieng", "din", "din_latn", "dinlatn",
        # Nuer
        "nuer", "thok_naath", "thoknaath", "naath", "nus", "nus_latn", "nuslatn",
        # Zarma / Djerma / Songhai
        "zarma", "djerma", "zerma", "songhai", "sonrai", "koyraboro", "dje", "dje_latn", "djelatn", "son",
        # Kanuri
        "kanuri", "kanouri", "bornu", "yerwa", "knc", "knc_latn", "knclatn", "knc_arab", "kncarab", "kr",
        # Maasai
        "maasai", "masai", "maa", "mas", "mas_latn", "maslatn",
        # Additional Saharan & Nilo-Saharan varieties
        "tubu", "toubou", "teda", "daza", "fur", "fvr", "shilluk", "shk", "bari", "acholi", "ach",
        "teso", "kalenjin", "turkana", "nubian", "nobiin", "dongolawi",
        "saharien", "saharan",
    ],
    FAMILY_PIDGINS: [
        # Nigerian Pidgin / Naija
        "nigerian_pidgin", "nigerianpidgin", "naija", "pidgin_nigerian", "pidginnigerian", "pcm", "pcm_latn", "pcmlatn",
        # Afrikaans
        "afrikaans", "afr", "af", "afr_latn", "afrlatn",
        # Cape Verdean Creole
        "cape_verdean", "capeverdean", "cape_verdean_creole", "capeverdeancreole", "kriolu", "crioulo", "kabuverdianu", "caboverdiano", "kea", "kea_latn", "kealatn",
        # Cameroonian Pidgin / Kamtok
        "cameroonian_pidgin", "cameroonianpidgin", "cameroon_pidgin", "cameroonpidgin", "kamtok", "camtok", "wes",
        # Mauritian Creole
        "mauritian_creole", "mauritiancreole", "morisyen", "morisien", "kreol_morisien", "kreolmorisien", "mfe",
        # Additional Pidgins & Creoles
        "krio", "sango", "sag", "sag_latn", "saglatn", "seychellois_creole", "seychelloiscreole", "seselwa", "crs", "patois", "reunion_creole", "reunioncreole",
        "pidgins", "creoles", "pidgin",
    ],
    FAMILY_KHOISAN: [
        # Nama / Khoekhoe
        "nama", "khoekhoe", "khoekhoegowab", "damara", "naq", "naq_latn", "naqlatn",
        # Juǀʼhoan
        "juǀʼhoan", "ju/hoan", "ju_hoan", "juhoan", "ju'hoan", "zhu", "kung", "!kung", "xung", "ktz",
        # !Xóõ
        "!xóõ", "!xoo", "!xoon", "xoo", "xoon", "taa", "nmn",
        # Additional Khoisan varieties
        "san", "khoisan", "khoïsan", "hadza", "hts", "sandawe", "sad", "khwe", "shua", "tswa", "naro",
    ],
}

# Target Output Filenames & Configuration
OUTPUT_LANGUAGES_FILE = "languages.txt"
DEFAULT_RANDOM_SEED = 42

# Output filenames to ignore during dataset discovery to prevent recursive processing
IGNORED_OUTPUT_FILES = {
    OUTPUT_LANGUAGES_FILE,
    "tokenizer.json",
}

# Output family filenames to ignore during discovery to prevent recursive processing
GENERATED_FAMILY_FILES = set(FAMILY_PATTERNS.keys()) | {
    "semetique.txt", "semitique.txt", "maghrebi.txt", "bantou.txt", "tchadique.txt",
    "couchitique.txt", "kwa.txt", "mande.txt", "saharien.txt", "pidgins.txt", "khoisan.txt",
    "bantu.txt", "arabic.txt", "pidgin.txt",
}

# Regex to strip common dataset name suffixes for clean matching
DATASET_CLEAN_REGEX = re.compile(
    r"(_normalized|_norm)?(_dataset|_cln|_cleaned|_clean|_train|_eval|_test|_corpus|_text|_webtext|_mix|_raw|_data|_final|_v\d+|\b[0-9]+\b)?$",
    flags=re.IGNORECASE,
)

# ============================================================================
# 1B. MULTILINGUAL SENTENCE DELIMITERS (PUNCTUATION MARKS ACROSS ALL LANGUAGES)
# ============================================================================
# Delimiters:
# - Latin / General: . ! ? … (\u2026)
# - Ethiopic (Amharic, Tigrinya, Ge'ez, Tigre): ። (\u1362), ፧ (\u1367), ፠ (\u1368)
# - Arabic script (Maghrebi, Arabic, Mauritanian, Chadian): ؟ (\u061F), ۔ (\u06D4)
# Closing quotes/brackets attached to sentence: " ' » « ) ] ” ’ ›
PUNCTUATION_TERMINATORS = r"[.!?…\u1362\u1367\u1368\u061F\u06D4]"
CLOSING_PUNCTUATION = r"""[\"')\]»”’›«]*"""
SENTENCE_SPLIT_REGEX = re.compile(rf"({PUNCTUATION_TERMINATORS}+{CLOSING_PUNCTUATION})")

TERMINAL_PUNCTUATION_CHARS = (
    ".", "!", "?", "…", "\u1362", "\u1367", "\u1368", "\u061F", "\u06D4",
    '"', "'", "»", "«", "”", "’", "›", ")", "]"
)


# ============================================================================
# 2. COMPREHENSIVE TEXT CLEANING REGEX ENGINE (ALL LANGUAGES & SPAM FRAMINGS)
# ============================================================================

# 1. URLs & Web Links
URL_CLEAN_REGEX = re.compile(
    r"\b(?:https?://|www\.|ftp://|t\.co/|bit\.ly/|goo\.gl/|tinyurl\.com/|wa\.me/|t\.me/|chat\.whatsapp\.com/)\S+",
    re.IGNORECASE,
)

# 2. Handles & Mentions
USER_HANDLE_REGEX = re.compile(r"@\w+", re.UNICODE)

# 3. Hashtags (#tag or #MultiWordTag)
HASHTAG_REGEX = re.compile(r"#[^\s#]+", re.UNICODE)

# 4. Media Tags, Placeholders, and Not Found Markers
TAGS_AND_NOT_FOUND_REGEX = re.compile(
    r"(?:"
    r"!\[[^\]]*\](?:\([^\)]*\))?"
    r"|\[\s*(?:image|photo|pic|picture|video|audio|file|link|url|http|https|table|fig|figure|illustration|embedded\s+content)[\w\s:./_-]*\]"
    r"|<[^>]+>"
    r"|\[[^\]]*\]\((?:https?://|www\.)\S+\)"
    r"|\[(?:watch|photos?|video|breaking|advert|sponsored|exclusive|caption)\]"
    r"|\b(?:image|photo|pic|picture|video|audio|file|link|url|irl|document|fig|figure|illustration)\s+(?:not\s+found|unavailable|missing|error|deleted|introuvable)\b"
    r"|\birl\s+not\s+found\b"
    r"|\bimage\s+not\s+found\b"
    r"|\burl\s+not\s+found\b"
    r")",
    re.IGNORECASE,
)

# 5. Personally Identifiable Information (PII)
# 5a. Emails
EMAIL_PII_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# 5b. IPv4 Addresses
IPV4_PII_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)

# 5c. Phone Numbers (international & local standard phone formats: +123..., (123) 456-7890, etc.)
PHONE_PII_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
)

# 5d. Credit Card / National ID Number Patterns (16-digit or 4x4 digit groups)
CREDIT_CARD_PII_REGEX = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

# 6. Comprehensive Multilingual Spam Framings (Promotional, CTA, Boilerplate & Scams)
MULTILINGUAL_SPAM_REGEX = re.compile(
    r"(?:"
    # --- English Promotional & CTA ---
    r"\b(?:subscribe|subscribers?|follow\s+us(?:\s+on)?|follow\s+me|like\s+and\s+subscribe|share\s+this(?:\s+post)?|join\s+our\s+channel)\b"
    r"|\b(?:click\s+here|link\s+in\s+bio|read\s+more(?:\s+at)?|visit\s+our\s+website|tap\s+the\s+link|link\s+in\s+comments?)\b"
    r"|\b(?:dm\s+for\s+(?:promo|adverts?|inquiries|details|prices?)|buy\s+now|order\s+now|free\s+delivery|limited\s+offer|discount\s+code|promo\s+code|call\s+now)\b"
    r"|\b(?:breaking\s+news\s*:|exclusive\s*:|watch\s+till\s+the\s+end|you\s+won'?t\s+believe|trending\s+now|viral\s+video|photo\s+credit\s*:|video\s+credit\s*:|courtesy\s+of)\b"
    r"|\b(?:all\s+rights\s+reserved|terms\s+and\s+conditions|privacy\s+policy|terms\s+of\s+(?:service|use)|cookie\s+policy|copyright\s+©?\s*\d{0,4})\b"
    r"|\b(?:congratulations\s+you\s+won|claim\s+your\s+prize|giveaways?|free\s+bitcoin|crypto\s+investment|forex\s+trading|fixed\s+matches?|100%\s*win|easy\s+money|lottery\s+winner|cash\s+prize)\b"
    r"|\b(?:daily\s+quotes|quote\s+of\s+the\s+day|random\s+quotes|random\s+thoughts|translated\s+quotes|any\s+quote\s+in\s+\w+)\b"
    # --- French Promotional & CTA ---
    r"|\b(?:abonnez[- ]vous|abonne[- ]toi|suivez[- ]nous|suivez[- ]moi|partagez(?:\s+cette\s+vid[eé]o|\s+au\s+maximum|\s+la\s+vid[eé]o)?|aimez\s+et\s+partagez|laissez\s+un\s+commentaire)\b"
    r"|\b(?:cliquez\s+ici|clique\s+ici|lien\s+(?:dans\s+la\s+bio|en\s+bio|en\s+commentaire)|consultez\s+notre\s+site|lire\s+la\s+suite)\b"
    r"|\b(?:contactez[- ]nous\s+sur\s+whatsapp|rejoignez\s+(?:notre\s+canal\s+telegram|le\s+groupe)|num[eé]ro\s+whatsapp|infoline|pour\s+commander|livraison\s+(?:disponible|gratuite)|prix\s+en\s+dm|promo\s+exclusive|offre\s+limit[eé]e)\b"
    r"|\b(?:source\s*:|cr[eé]dit\s+photo\s*:|droits\s+r[eé]serv[eé]s|tous\s+droits\s+r[eé]serv[eé]s|politique\s+de\s+confidentialit[eé]|actualit[eé]\s+en\s+direct|derni[eè]re\s+minute|vid[eé]o\s+virale|regardez\s+jusqu'[aà]\s+la\s+fin|urgent\s*:|exclusif\s*:)\b"
    r"|\b(?:gagnez\s+de\s+l'argent|investissez\s+maintenant|crypto[- ]monnaie|paris\s+sportifs|match\s+truqu[eé])\b"
    # --- Arabic (MSA & Maghrebi / Darija) ---
    r"|(?:اشترك(?:وا)?\s+في\s+القناة|اشترك\s+الآن|اشتراك|فعل\s+الجرس|تابعنا|تابعونا|تابعوني|تابعوا|شارك\s+الفيديو|بارطاجي)"
    r"|(?:اضغط\s+هنا|الرابط\s+في\s+البايو|الرابط\s+في\s+الوصف|الرابط\s+في\s+أول\s+تعليق)"
    r"|(?:للطلب\s+عبر\s+الواتساب|للطلب|تواصل\s+معنا\s+على\s+الواتساب|تواصل\s+معنا|راسلنا|انضم\s+الى\s+القناة|انضموا\s+لمجموعتنا|رقم\s+الواتساب|توصيل\s+مجاني|الدفع\s+عند\s+الاستلام|عرض\s+محدود|سارعوا\s+بالطلب)"
    r"|(?:عاجل\s*:|حصريا\s*:|شاهد\s+الفيديو|شاهد\s+قبل\s+الحذف|لا\s+يفوتك|بالفيديو\s*:|صور\s+وفيديو|منقول\s+للأمانة|جميع\s+الحقوق\s+محفوظة)"
    r"|(?:ربح\s+المال|استثمار|تداول|مراهنات|مبروك\s+ربحت\s+معنا)"
    # --- Arabizi ---
    r"|\b(?:aboni|aboniw|partagi|partagiw|cliqui\s+hna|lien\s+f\s+bio|lien\s+f\s+lwasf|chouf\s+lvideo|chouf\s+hna|marhba\s+bikom|pour\s+commander|livraison\s+gratuite|contactewna\s+f\s+whatsapp|direct\s+whatsapp|rejoignez\s+telegram)\b"
    # --- Swahili ---
    r"|\b(?:bonyeza\s+hapa|bofya\s+hapa|jiunge\s+nasi|tufuate|fuatilia|tazama\s+video|tazama\s+hapa|kiungo\s+kwenye\s+bio|link\s+kwenye\s+bio|share\s+na\s+marafiki|habari\s+za\s+hivi\s+punde|habari\s+zilizotufikia|soma\s+zaidi|chanzo\s+cha\s+habari|haki\s+zote\s+zimehifadhiwa|wasiliana\s+nasi\s+kwa\s+whatsapp|piga\s+simu\s+namba|kwa\s+mawasiliano|agiza\s+sasa|ofa\s+maalum|mikopo\s+ya\s+haraka)\b"
    # --- Hausa ---
    r"|\b(?:danna\s+nan|ku\s+biyo\s+mu|ku\s+danna\s+subscribe|kuyi\s+subscribing|kuyi\s+sharing|shafin\s+yanar\s+gizo|domin\s+karin\s+bayani|kalli\s+bidiyon\s+nan|kalli\s+yadda|sabon\s+labari|labaran\s+yanzu|tuntube\s+mu\s+a\s+whatsapp|tallata\s+haja|kudi\s+cikin\s+sauki)\b"
    # --- Yoruba ---
    r"|\b(?:e\s+te\s+ibi|e\s+t[eẹ]le\s+wa|[eẹ]\s+wo\s+fidio\s+yii|alaye\s+l[eẹ]kunr[eẹ]r[eẹ]|ipolowo\s+[oọ]ja|fun\s+ipolowo|pe\s+wa\s+lori\s+whatsapp|alaye\s+siwaju\s+sii|e\s+pin\s+fun\s+awon\s+ore)\b"
    # --- Igbo ---
    r"|\b(?:p[iị]a\s+ebe\s+a|soro\s+any[iị]|kiri\s+vidiyo\s+a|mgbasa\s+ozi|kp[oọ]t[uụ]r[uụ]\s+any[iị]\s+na\s+whatsapp|maka\s+ozi\s+nd[iị]\s+[oọ]z[oọ]|kesaa\s+nd[iị]\s+[oọ]z[oọ])\b"
    # --- Amharic ---
    r"|(?:ሰብስክራይብ\s+ያድርጉ|ይከተሉን|እዚህ\s+ይጫኑ|ሊንኩን\s+ይጫኑ|ቪዲዮውን\s+ይመልከቱ|ሼር\s+ያድርጉ|ማስታወቂያ|ለበለጠ\s+መረጃ|በዋትስአፕ\s+ያግኙን|ቴሌግራም\s+ቻናል|ሰበር\s+ዜና|ሁሉም\s+መብቱ\s+የተጠበቀ\s+ነው)"
    # --- Somali / Oromo ---
    r"|\b(?:nu\s+hordofaa|asi\s+tuqi|vidiyoo\s+kana\s+ilaalaa|odeeffannoo\s+dabalataaf|beeksisa|ku\s+soo\s+biir|riix\s+halkan|nagala\s+soo\s+xiriir\s+whatsapp)\b"
    # --- Pidgin ---
    r"|\b(?:make\s+una\s+follow|follow\s+us|drop\s+your\s+number|join\s+our\s+telegram|dm\s+for\s+promo|dm\s+for\s+adverts?|no\s+dull\s+yourself|whatsapp\s+us\s+on)\b"
    # --- Disguised chat / invite links ---
    r"|\b(?:wa\.me|t\.me|chat\.whatsapp\.com|tinyurl\.com)/\S+"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# 7. Repeated Character Floods and Punctuation Floods
REPEATED_CHAR_FLOOD_REGEX = re.compile(r"\b\w*([a-zA-Z])\1{4,}\w*\b")
PUNCTUATION_FLOOD_REGEX = re.compile(r"[!?.]{4,}|[#*=_~-]{4,}")

# 8. Generalized CamelCase Campaign Compounds (e.g. 'TranslatedQuotes', 'March25WorldSultansDay')
CAMELCASE_SPAM_REGEX = re.compile(
    r"\b(?:[A-Z][a-z0-9]+){2,}\b"
)

# 9. Platform Keywords & Campaign Roots
PLATFORM_SPAM_REGEX = re.compile(
    r"\b(?:"
    r"instagram|facebook|twitter|tiktok|youtube|telegram|snapchat|whatsapp|reddit|linkedin|pinterest"
    r"|yorubatv|yorubaweddings|translatedquotes|anyquoteinyoruba|randomquotes|randomthoughts"
    r"|march25worldsultansday|justiceforhacaaluuhundeessa|oromiyaan_biyya|quoteinyoruba"
    r"|officialpage|officialchannel"
    r")\b",
    re.IGNORECASE,
)

# 10. Whitespace Normalization
WHITESPACE_COLLAPSE_REGEX = re.compile(r"[ \t\f\v]+")
NEWLINE_REGEX = re.compile(r"[\r\n]+")


# ============================================================================
# 3. TEXT SANITIZATION ENGINE
# ============================================================================

def is_arabizi_or_alphanumeric_word(token: str) -> bool:
    """
    Determines whether a token is a mixed alphanumeric word or Arabizi term.

    Identifies tokens containing both alphabetical characters (Latin, Arabic,
    Ethiopic, etc.) and numerical digits, such as Arabizi expressions
    ('3amer', 'm3amer', '7abibi', 'sal7a', '9ahwa') or terms ('COVID-19').

    Args:
        token: Surface string token to analyze.

    Returns:
        True if the token contains at least one alphabetic character and at least
        one numerical digit; False otherwise.
    """
    has_alpha = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    return has_alpha and has_digit


def filter_standalone_numbers_and_spam_words(text: str) -> str:
    """Filters noisy standalone numbers, campaign hashtags, and character floods.

    Iterates token-by-token over a space-delimited string:
    - Removes purely numeric tokens (e.g., '2022', '100-19', '12/34', '00', '3').
    - Preserves Arabizi and valid alphanumeric terms (e.g., '3amer', 'COVID-19').
    - Removes repeated character floods (e.g., 'sooooo', 'hhhhhhh').
    - Removes CamelCase marketing campaign compounds (e.g., 'March25WorldSultansDay').
    - Filters recognized social media platform keywords.

    Args:
        text: Space-delimited string of tokens.

    Returns:
        Cleaned string with noise tokens removed.
    """
    words = text.split(" ")
    cleaned_words = []

    for w in words:
        if not w:
            continue

        # Strip surrounding basic, Ethiopic, and Arabic punctuation for classification
        stripped = w.strip(""".,;:!?"'«»()[]{}<>~`@#$%^&*+=|\\/\u1360\u1361\u1362\u1363\u1364\u1365\u1366\u1367\u1368\u060C\u061B\u061F\u06D4""")

        if not stripped:
            continue

        # Purely numeric or numeric+punctuation without letters -> REMOVE
        if not any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped):
            continue

        # Repeated character flood (e.g. 'soooooo', 'hhhhhh')
        if REPEATED_CHAR_FLOOD_REGEX.fullmatch(stripped):
            continue

        # CamelCase campaign detection on tokens of length >= 8 with >= 2 uppercase transitions
        if len(stripped) >= 8 and sum(1 for c in stripped if c.isupper()) >= 2:
            if CAMELCASE_SPAM_REGEX.fullmatch(stripped):
                continue

        # Keyword spam / platform check
        if PLATFORM_SPAM_REGEX.fullmatch(stripped):
            continue

        cleaned_words.append(w)

    return " ".join(cleaned_words)


def clean_raw_dataset_text(text: str) -> str:
    """Applies the complete sanitization pipeline to a raw corpus line.

    Execution Stages:
    1. Unescapes HTML entities (&amp; -> &, etc.).
    2. Converts all line breaks (\\r\\n, \\r, \\n) into single spaces.
    3. Removes tags, placeholders, HTML/XML elements, and 'not found' markers.
    4. Removes URLs, web links, and disguised shortlinks.
    5. Strips multilingual promotional boilerplate and spam call-to-actions.
    6. Removes social media handles (@user) and hashtags (#tag).
    7. Removes PII (emails, phone numbers, IP addresses, credit cards).
    8. Filters standalone numbers and campaign spam while preserving Arabizi words.
    9. Collapses whitespace runs into single ASCII spaces.

    Args:
        text: Raw line of text from an input dataset.

    Returns:
        Sanitized and space-collapsed text string.
    """

    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""

    # 1. HTML Unescape
    text = html.unescape(text)

    # 2. Replace return to lines (\r\n, \r, \n) with a space
    text = NEWLINE_REGEX.sub(" ", text)

    # 3. Remove Tags, Placeholders, HTML/XML markup, and "not found" tags
    text = TAGS_AND_NOT_FOUND_REGEX.sub(" ", text)

    # 4. Remove URLs and links
    text = URL_CLEAN_REGEX.sub(" ", text)

    # 5. Remove Multilingual Spam Framings
    text = MULTILINGUAL_SPAM_REGEX.sub(" ", text)
    text = PUNCTUATION_FLOOD_REGEX.sub(" ", text)

    # 6. Remove Handles and Hashtags
    text = USER_HANDLE_REGEX.sub(" ", text)
    text = HASHTAG_REGEX.sub(" ", text)

    # 7. Remove PII: Emails, IPs, Credit Cards, Phone Numbers
    text = EMAIL_PII_REGEX.sub(" ", text)
    text = IPV4_PII_REGEX.sub(" ", text)
    text = CREDIT_CARD_PII_REGEX.sub(" ", text)
    text = PHONE_PII_REGEX.sub(" ", text)

    # 8. Word-level filtering: Remove isolated numbers & campaign spam; preserve Arabizi & normal words
    text = filter_standalone_numbers_and_spam_words(text)

    # 9. Normalize whitespace to single spaces
    text = WHITESPACE_COLLAPSE_REGEX.sub(" ", text)
    return text.strip()


# ============================================================================
# 4. UNIVERSAL CLASSIFICATION & DISCOVERY ENGINE (10 LINGUISTIC FAMILIES)
# ============================================================================

def normalize_filename_key(filename: str) -> str:
    """Extracts a normalized, accent-free, alphanumeric key from a filename.

    Used for robust matching across varied naming schemes, aliases, and diacritics.
    Example: 'Yorùbá_normalized_dataset.txt' -> 'yoruba'
             'isiXhosa_dataset.txt' -> 'isixhosa'
             'Juǀʼhoan_clean.txt' -> 'juhoan'

    Args:
        filename: Relative or absolute path of the target file.

    Returns:
        Lowercased alphanumeric key string.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    cleaned = DATASET_CLEAN_REGEX.sub("", base_name).lower()
    # Normalize unicode diacritics
    normalized = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    # Clean non-alphanumeric characters
    cleaned_key = re.sub(r"[^a-z0-9]", "", normalized)
    return cleaned_key if cleaned_key else base_name.lower()


def classify_file(filename: str) -> str | None:
    """Classifies a dataset file into one of the 10 African linguistic root families.

    Applies a three-tier matching heuristic:
    - Tier 1: Token-based exact matching against family alias sets.
    - Tier 2: Substring matching against the collapsed filename string.
    - Tier 3: Substring matching against the raw lowercased filename.

    Args:
        filename: Path or basename of the candidate dataset file.

    Returns:
        Target family filename (e.g., 'bantou.txt', 'maghrebi.txt') or None if unclassified.
    """

    base_name = os.path.basename(filename)
    if base_name.lower() in GENERATED_FAMILY_FILES:
        return None

    # Step 1: Extract normalized stem tokens from filename
    # e.g. "akan_1.txt" -> ["akan", "1"]
    # e.g. "egypt_Arabic.txt" -> ["egypt", "arabic"]
    # e.g. "bambara_text.txt" -> ["bambara", "text"]
    raw_name_no_ext = os.path.splitext(base_name)[0]
    normalized_ascii = unicodedata.normalize("NFKD", raw_name_no_ext).encode("ascii", "ignore").decode("ascii").lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", normalized_ascii) if t]
    collapsed_key = re.sub(r"[^a-z0-9]", "", normalized_ascii)

    # Words that are purely metadata noise and should not be used as language identifiers
    METADATA_NOISE = {
        "normalized", "norm", "dataset", "corpus", "text", "webtext", "mix",
        "cln", "clean", "cleaned", "train", "test", "eval", "data", "raw",
        "v1", "v2", "v3", "v4", "v5", "final", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "txt", "latn", "ethi", "arab"
    }
    meaningful_tokens = [t for t in tokens if t not in METADATA_NOISE]

    # Tier 1: Token-based exact matching against all family patterns
    for family_file, patterns in FAMILY_PATTERNS.items():
        pattern_set = {
            re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", p.lower()).encode("ascii", "ignore").decode("ascii"))
            for p in patterns
        }
        for token in meaningful_tokens:
            if token in pattern_set:
                return family_file

    # Tier 2: Substring matching against collapsed filename
    for family_file, patterns in FAMILY_PATTERNS.items():
        for p in patterns:
            p_clean = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", p.lower()).encode("ascii", "ignore").decode("ascii"))
            if p_clean and len(p_clean) >= 3 and p_clean in collapsed_key:
                return family_file

    # Tier 3: Substring matching against raw lowercase filename
    raw_lower = base_name.lower()
    for family_file, patterns in FAMILY_PATTERNS.items():
        for p in patterns:
            if len(p) >= 3 and p.lower() in raw_lower:
                return family_file

    return None


def discover_and_group_files(input_dir: str) -> dict[str, list[str]]:
    """Discovers .txt files in a directory and groups them by root family.

    Args:
        input_dir: Directory containing individual language dataset files.

    Returns:
        Mapping from family filename (e.g. 'bantou.txt') to list of absolute file paths.
    """

    grouped_files: dict[str, list[str]] = defaultdict(list)
    unmatched_files = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(TXT_EXTENSION):
            continue

        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            continue

        family = classify_file(fname)
        if family is not None:
            grouped_files[family].append(fpath)
        else:
            if fname.lower() not in GENERATED_FAMILY_FILES:
                unmatched_files.append(fpath)

    if unmatched_files:
        print("\n[NOTE] Unmatched / Unclassified dataset files:")
        for uf in unmatched_files:
            print(f"  - {os.path.basename(uf)}")

    return dict(grouped_files)


def discover_dataset_files(input_dir: str) -> list[str]:
    """Discovers all language dataset .txt files for corpus consolidation.

    Discovery Priority:
    1. Identifies individual language dataset files matching family patterns.
    2. Fallback: Identifies pre-generated family corpus files if individual files are absent.
    3. Fallback: Selects any remaining non-ignored text files in the directory.
    Excludes output artifacts such as 'languages.txt' and 'tokenizer.json'.

    Args:
        input_dir: Directory path to scan.

    Returns:
        List of absolute file paths to discovered dataset files.
    """
    if not os.path.isdir(input_dir):
        return []

    all_txt_files = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(TXT_EXTENSION):
            continue
        if fname.lower() in IGNORED_OUTPUT_FILES:
            continue
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath):
            all_txt_files.append(fpath)

    # 1. Identify individual language dataset files matching classify_file
    individual_files = [f for f in all_txt_files if classify_file(os.path.basename(f)) is not None]
    if individual_files:
        return individual_files

    # 2. Fallback: Identify family corpus files if individual language files are not present
    family_files = [
        f for f in all_txt_files
        if os.path.basename(f).lower() in GENERATED_FAMILY_FILES
    ]
    if family_files:
        return family_files

    # 3. Fallback: Any remaining non-ignored text files in directory
    return all_txt_files


# ============================================================================
# 5. MULTILINGUAL SENTENCE EXTRACTION & BALANCED SHUFFLING ENGINE
# ============================================================================

def extract_sentences_from_cleaned_text(cleaned_text: str) -> list[str]:
    """Segments cleaned text into discrete sentences delimited by punctuation.

    Supports punctuation across Latin, Ethiopic, and Arabic scripts. Preserves
    terminal punctuation and closing quotes with each sentence. Replaces unpunctuated
    line breaks with '.' to treat individual lines as distinct sentences.

    Args:
        cleaned_text: Pre-sanitized text string.

    Returns:
        List of isolated sentence strings containing valid alphanumeric content.
    """

    if not cleaned_text or not cleaned_text.strip():
        return []

    # Handle internal line breaks: if a line break is not preceded by terminal punctuation, replace it with '.'
    if "\n" in cleaned_text or "\r" in cleaned_text:
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        normalized = []
        for line in lines:
            if not line.endswith(TERMINAL_PUNCTUATION_CHARS):
                normalized.append(line + ".")
            else:
                normalized.append(line)
        cleaned_text = " ".join(normalized)

    parts = SENTENCE_SPLIT_REGEX.split(cleaned_text)
    sentences = []
    num_complete = (len(parts) - 1) // 2

    for i in range(num_complete):
        s_text = parts[2 * i].strip()
        s_delim = parts[2 * i + 1].strip()
        if s_text:
            sentence = f"{s_text}{s_delim}"
            if any(c.isalnum() for c in sentence):
                sentences.append(sentence)

    # Handle remaining tail if any
    tail = parts[-1].strip()
    if tail and any(c.isalnum() for c in tail):
        if not tail.endswith(TERMINAL_PUNCTUATION_CHARS):
            tail += "."
        sentences.append(tail)

    return sentences


def extract_sentences_from_file(file_path: str) -> list[str]:
    """
    Reads, sanitizes, and extracts discrete sentences from a dataset file.

    Line breaks lacking terminal punctuation are treated as sentence boundaries
    by appending a '.' to prevent sentences from improperly merging across lines.

    Args:
        file_path: Path to the dataset text file.

    Returns:
        List of clean, segmented sentences.
    """
    sentences = []

    with open(file_path, FILE_MODE_READ, encoding=ENCODING_UTF8, errors=ERROR_HANDLER_REPLACE) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue

            cleaned_line = clean_raw_dataset_text(stripped)
            if not cleaned_line:
                continue

            # If the line has no terminal punctuation, treat the line break as a sentence boundary
            if not cleaned_line.endswith(TERMINAL_PUNCTUATION_CHARS):
                cleaned_line += "."

            line_sents = extract_sentences_from_cleaned_text(cleaned_line)
            sentences.extend(line_sents)

    return sentences


def build_balanced_shuffled_corpus(
    dataset_paths: list[str],
    output_path: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """Builds a balanced, globally shuffled multilingual corpus file.

    Execution Workflow:
    1. Extracts clean sentences from each source dataset file.
    2. Identifies the minimum sentence count available across all valid files.
    3. Randomly samples that exact minimum count from each dataset to guarantee balance.
    4. Globally shuffles the combined sentence collection across all languages.
    5. Writes the final corpus to output_path (languages.txt).

    Args:
        dataset_paths: List of file paths to process.
        output_path: Destination path for the output languages.txt file.
        seed: Random seed for deterministic sampling and shuffling.

    Returns:
        Dictionary containing corpus statistics (words, characters, sentences, file size).
    """

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 80)
    print("  EXTRACTING & CLEANING SENTENCES FROM LANGUAGE DATASETS")
    print("=" * 80)

    file_sentences: dict[str, list[str]] = {}
    for fpath in dataset_paths:
        fname = os.path.basename(fpath)
        sents = extract_sentences_from_file(fpath)
        if sents:
            file_sentences[fpath] = sents
            print(f"  + {fname:<40} : {len(sents):>10,d} sentences extracted")
        else:
            print(f"  [WARN] {fname:<38} : 0 valid sentences extracted (skipped)")

    if not file_sentences:
        print("[ERROR] No valid sentences found in any dataset file.")
        return {}

    # Find minimum sentence count across all files
    min_sentences = min(len(sents) for sents in file_sentences.values())

    print("\n" + "-" * 80)
    print(f"  BALANCING: Minimum sentences found across all files = {min_sentences:,}")
    print(f"  Sampling {min_sentences:,} random sentences per dataset file (seed={seed})")
    print("-" * 80)

    # Deterministic sampling with seed
    rng = random.Random(seed)
    sampled_sentences: list[str] = []
    stats_per_file = {}

    for fpath, sents in file_sentences.items():
        fname = os.path.basename(fpath)
        if len(sents) == min_sentences:
            chosen = list(sents)
        else:
            chosen = rng.sample(sents, min_sentences)

        sampled_sentences.extend(chosen)
        stats_per_file[fname] = {
            "total_sentences": len(sents),
            "sampled_sentences": len(chosen),
        }
        print(f"  * {fname:<40} : sampled {len(chosen):>10,d} / {len(sents):>10,d} sentences")

    # Global sentence shuffle across all languages (words inside each sentence remain unchanged)
    print(f"\n[SHUFFLE] Globally shuffling {len(sampled_sentences):,} sentences across all languages...")
    rng.shuffle(sampled_sentences)

    # Write to languages.txt
    print(f"[WRITE] Writing shuffled sentences to: {output_path}")
    total_words = 0
    total_chars = 0

    with open(output_path, FILE_MODE_WRITE, encoding=ENCODING_UTF8) as out_f:
        for sentence in sampled_sentences:
            out_f.write(sentence + "\n")
            total_words += len(sentence.split())
            total_chars += len(sentence) + 1

    out_size_bytes = os.path.getsize(output_path)

    summary = {
        "output_path": output_path,
        "num_datasets": len(file_sentences),
        "min_sentences": min_sentences,
        "total_sentences": len(sampled_sentences),
        "total_words": total_words,
        "total_chars": total_chars,
        "size_bytes": out_size_bytes,
        "per_file": stats_per_file,
    }

    print("\n" + "=" * 80)
    print(f"  SUMMARY: BALANCED & SHUFFLED MULTILINGUAL CORPUS GENERATED")
    print("=" * 80)
    print(f"  Output File              : {os.path.basename(output_path)}")
    print(f"  Datasets Included        : {len(file_sentences):,}")
    print(f"  Sentences per Dataset    : {min_sentences:,}")
    print(f"  Total Shuffled Sentences : {len(sampled_sentences):,}")
    print(f"  Total Words              : {total_words:,}")
    print(f"  Total Characters         : {total_chars:,}")
    print(f"  File Size                : {out_size_bytes / (1024 * 1024):.2f} MB")
    print(f"  Full Path                : {output_path}")
    print("=" * 80 + "\n")

    return summary

def concatenate_and_clean_family_files(
    grouped_files: dict[str, list[str]],
    output_dir: str,
) -> dict[str, dict]:
    """Consolidates individual language datasets into family-level corpus files.

    Cleans all input lines, replaces line breaks with spaces, and writes
    the concatenated content out as continuous paragraphs for each linguistic family.

    Args:
        grouped_files: Mapping from family filename to list of source file paths.
        output_dir: Destination directory for the generated family files.

    Returns:
        Dictionary of summary statistics per linguistic family file.
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_stats = {}

    all_families_in_order = [
        FAMILY_BANTOU,
        FAMILY_SEMETIQUE,
        FAMILY_MAGHREBI,
        FAMILY_TCHADIQUE,
        FAMILY_COUCHITIQUE,
        FAMILY_KWA,
        FAMILY_MANDE,
        FAMILY_SAHARIEN,
        FAMILY_PIDGINS,
        FAMILY_KHOISAN,
    ]

    print("=" * 80)
    print("  CONSOLIDATING & CLEANING AFRICAN LANGUAGE DATASETS (10 LINGUISTIC FAMILIES)")
    print("=" * 80)

    for family_file in all_families_in_order:
        source_paths = grouped_files.get(family_file, [])
        out_path = os.path.join(output_dir, family_file)

        if not source_paths:
            print(f"\n[SKIP] {family_file}: No source files found.")
            continue

        print(f"\nProcessing [{family_file}] from {len(source_paths)} source file(s):")
        total_chunks = 0
        total_words = 0
        total_chars = 0
        is_first_chunk = True

        with open(out_path, FILE_MODE_WRITE, encoding=ENCODING_UTF8) as out_f:
            for src_path in source_paths:
                file_chunks = 0
                file_words = 0
                file_chars = 0
                src_name = os.path.basename(src_path)

                with open(src_path, FILE_MODE_READ, encoding=ENCODING_UTF8, errors=ERROR_HANDLER_REPLACE) as in_f:
                    for raw_line in in_f:
                        cleaned = clean_raw_dataset_text(raw_line)
                        if cleaned:
                            w_count = len(cleaned.split())
                            if is_first_chunk:
                                out_f.write(cleaned)
                                is_first_chunk = False
                                file_chars += len(cleaned)
                            else:
                                out_f.write(" " + cleaned)
                                file_chars += len(cleaned) + 1
                            file_chunks += 1
                            file_words += w_count

                total_chunks += file_chunks
                total_words += file_words
                total_chars += file_chars
                print(f"  + {src_name:<38} : {file_words:>8,d} words ({file_chars:>12,d} chars)")

        out_size_bytes = os.path.getsize(out_path)
        summary_stats[family_file] = {
            "source_count": len(source_paths),
            "chunks": total_chunks,
            "words": total_words,
            "chars": total_chars,
            "size_bytes": out_size_bytes,
            "out_path": out_path,
        }

        print(f"  --> Saved {family_file} as single continuous paragraph: {total_words:,} words ({total_chars:,} chars, {out_size_bytes / (1024 * 1024):.2f} MB)")

    return summary_stats


def print_summary_report(summary_stats: dict[str, dict]):
    """
    Prints an aligned summary table of the generated linguistic family files.

    Args:
        summary_stats: Dictionary containing statistics per linguistic family file.
    """
    print("\n" + "=" * 80)
    print(f"{'Linguistic Family File':<24} {'Sources':>8} {'Total Words':>14} {'Total Chars':>16} {'Size (MB)':>12}")
    print("-" * 80)

    grand_words = 0
    grand_chars = 0
    grand_bytes = 0

    for family_file, stats in summary_stats.items():
        grand_words += stats["words"]
        grand_chars += stats["chars"]
        grand_bytes += stats["size_bytes"]
        size_mb = stats["size_bytes"] / (1024 * 1024)
        print(f"{family_file:<24} {stats['source_count']:>8,d} {stats['words']:>14,d} {stats['chars']:>16,d} {size_mb:>12.2f}")

    print("-" * 80)
    print(f"{'TOTAL':<24} {'':>8} {grand_words:>14,d} {grand_chars:>16,d} {grand_bytes / (1024 * 1024):>12.2f}")
    print("=" * 80)


# ============================================================================
# 6. MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """CLI entry point for consolidating and balancing African language corpora."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

    parser = argparse.ArgumentParser(
        description="Clean, balance, and shuffle African language datasets into a unified languages.txt corpus."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=script_dir,
        help=f"Directory containing source dataset .txt files (default: {script_dir})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=script_dir,
        help=f"Directory to save output files (default: {script_dir})"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=OUTPUT_LANGUAGES_FILE,
        help=f"Filename for balanced shuffled corpus (default: {OUTPUT_LANGUAGES_FILE})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for sentence sampling and shuffling (default: {DEFAULT_RANDOM_SEED})"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    output_path = os.path.join(output_dir, args.output_file)

    print(f"[INFO] Input directory : {input_dir}")
    print(f"[INFO] Output file     : {output_path}")

    dataset_files = discover_dataset_files(input_dir)
    if not dataset_files:
        print("[ERROR] No matching language dataset files found in input directory.")
        sys.exit(1)

    print(f"[INFO] Discovered {len(dataset_files)} dataset file(s):")
    for df in dataset_files:
        print(f"  - {os.path.basename(df)}")

    build_balanced_shuffled_corpus(dataset_files, output_path, seed=args.seed)


if __name__ == "__main__":
    main()
