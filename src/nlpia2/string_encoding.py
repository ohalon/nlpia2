"""

>> dfs = pd.read_html('https://docs.python.org/3/library/codecs.html#standard-encodings')
>> df = dfs[4]
>> df.to_json('python_standard_string_encodings.json', orient='records', indent=4)

replace "None" with "None" and prefix with "ENCODINGS = "
"""

ENCODINGS = [
    {
        "Codec": "ascii",
        "Aliases": "646, us-ascii",
        "Languages": "English"
    },
    {
        "Codec": "big5",
        "Aliases": "big5-tw, csbig5",
        "Languages": "Traditional Chinese"
    },
    {
        "Codec": "big5hkscs",
        "Aliases": "big5-hkscs, hkscs",
        "Languages": "Traditional Chinese"
    },
    {
        "Codec": "cp037",
        "Aliases": "IBM037, IBM039",
        "Languages": "English"
    },
    {
        "Codec": "cp273",
        "Aliases": "273, IBM273, csIBM273",
        "Languages": "German New in version 3.4."
    },
    {
        "Codec": "cp424",
        "Aliases": "EBCDIC-CP-HE, IBM424",
        "Languages": "Hebrew"
    },
    {
        "Codec": "cp437",
        "Aliases": "437, IBM437",
        "Languages": "English"
    },
    {
        "Codec": "cp500",
        "Aliases": "EBCDIC-CP-BE, EBCDIC-CP-CH, IBM500",
        "Languages": "Western Europe"
    },
    {
        "Codec": "cp720",
        "Aliases": None,
        "Languages": "Arabic"
    },
    {
        "Codec": "cp737",
        "Aliases": None,
        "Languages": "Greek"
    },
    {
        "Codec": "cp775",
        "Aliases": "IBM775",
        "Languages": "Baltic languages"
    },
    {
        "Codec": "cp850",
        "Aliases": "850, IBM850",
        "Languages": "Western Europe"
    },
    {
        "Codec": "cp852",
        "Aliases": "852, IBM852",
        "Languages": "Central and Eastern Europe"
    },
    {
        "Codec": "cp855",
        "Aliases": "855, IBM855",
        "Languages": "Bulgarian, Byelorussian, Macedonian, Russian, Serbian"
    },
    {
        "Codec": "cp856",
        "Aliases": None,
        "Languages": "Hebrew"
    },
    {
        "Codec": "cp857",
        "Aliases": "857, IBM857",
        "Languages": "Turkish"
    },
    {
        "Codec": "cp858",
        "Aliases": "858, IBM858",
        "Languages": "Western Europe"
    },
    {
        "Codec": "cp860",
        "Aliases": "860, IBM860",
        "Languages": "Portuguese"
    },
    {
        "Codec": "cp861",
        "Aliases": "861, CP-IS, IBM861",
        "Languages": "Icelandic"
    },
    {
        "Codec": "cp862",
        "Aliases": "862, IBM862",
        "Languages": "Hebrew"
    },
    {
        "Codec": "cp863",
        "Aliases": "863, IBM863",
        "Languages": "Canadian"
    },
    {
        "Codec": "cp864",
        "Aliases": "IBM864",
        "Languages": "Arabic"
    },
    {
        "Codec": "cp865",
        "Aliases": "865, IBM865",
        "Languages": "Danish, Norwegian"
    },
    {
        "Codec": "cp866",
        "Aliases": "866, IBM866",
        "Languages": "Russian"
    },
    {
        "Codec": "cp869",
        "Aliases": "869, CP-GR, IBM869",
        "Languages": "Greek"
    },
    {
        "Codec": "cp874",
        "Aliases": None,
        "Languages": "Thai"
    },
    {
        "Codec": "cp875",
        "Aliases": None,
        "Languages": "Greek"
    },
    {
        "Codec": "cp932",
        "Aliases": "932, ms932, mskanji, ms-kanji",
        "Languages": "Japanese"
    },
    {
        "Codec": "cp949",
        "Aliases": "949, ms949, uhc",
        "Languages": "Korean"
    },
    {
        "Codec": "cp950",
        "Aliases": "950, ms950",
        "Languages": "Traditional Chinese"
    },
    {
        "Codec": "cp1006",
        "Aliases": None,
        "Languages": "Urdu"
    },
    {
        "Codec": "cp1026",
        "Aliases": "ibm1026",
        "Languages": "Turkish"
    },
    {
        "Codec": "cp1125",
        "Aliases": "1125, ibm1125, cp866u, ruscii",
        "Languages": "Ukrainian New in version 3.4."
    },
    {
        "Codec": "cp1140",
        "Aliases": "ibm1140",
        "Languages": "Western Europe"
    },
    {
        "Codec": "cp1250",
        "Aliases": "windows-1250",
        "Languages": "Central and Eastern Europe"
    },
    {
        "Codec": "cp1251",
        "Aliases": "windows-1251",
        "Languages": "Bulgarian, Byelorussian, Macedonian, Russian, Serbian"
    },
    {
        "Codec": "cp1252",
        "Aliases": "windows-1252",
        "Languages": "Western Europe"
    },
    {
        "Codec": "cp1253",
        "Aliases": "windows-1253",
        "Languages": "Greek"
    },
    {
        "Codec": "cp1254",
        "Aliases": "windows-1254",
        "Languages": "Turkish"
    },
    {
        "Codec": "cp1255",
        "Aliases": "windows-1255",
        "Languages": "Hebrew"
    },
    {
        "Codec": "cp1256",
        "Aliases": "windows-1256",
        "Languages": "Arabic"
    },
    {
        "Codec": "cp1257",
        "Aliases": "windows-1257",
        "Languages": "Baltic languages"
    },
    {
        "Codec": "cp1258",
        "Aliases": "windows-1258",
        "Languages": "Vietnamese"
    },
    {
        "Codec": "euc_jp",
        "Aliases": "eucjp, ujis, u-jis",
        "Languages": "Japanese"
    },
    {
        "Codec": "euc_jis_2004",
        "Aliases": "jisx0213, eucjis2004",
        "Languages": "Japanese"
    },
    {
        "Codec": "euc_jisx0213",
        "Aliases": "eucjisx0213",
        "Languages": "Japanese"
    },
    {
        "Codec": "euc_kr",
        "Aliases": "euckr, korean, ksc5601, ks_c-5601, ks_c-5601-1987, ksx1001, ks_x-1001",
        "Languages": "Korean"
    },
    {
        "Codec": "gb2312",
        "Aliases": "chinese, csiso58gb231280, euc-cn, euccn, eucgb2312-cn, gb2312-1980, gb2312-80, iso-ir-58",
        "Languages": "Simplified Chinese"
    },
    {
        "Codec": "gbk",
        "Aliases": "936, cp936, ms936",
        "Languages": "Unified Chinese"
    },
    {
        "Codec": "gb18030",
        "Aliases": "gb18030-2000",
        "Languages": "Unified Chinese"
    },
    {
        "Codec": "hz",
        "Aliases": "hzgb, hz-gb, hz-gb-2312",
        "Languages": "Simplified Chinese"
    },
    {
        "Codec": "iso2022_jp",
        "Aliases": "csiso2022jp, iso2022jp, iso-2022-jp",
        "Languages": "Japanese"
    },
    {
        "Codec": "iso2022_jp_1",
        "Aliases": "iso2022jp-1, iso-2022-jp-1",
        "Languages": "Japanese"
    },
    {
        "Codec": "iso2022_jp_2",
        "Aliases": "iso2022jp-2, iso-2022-jp-2",
        "Languages": "Japanese, Korean, Simplified Chinese, Western Europe, Greek"
    },
    {
        "Codec": "iso2022_jp_2004",
        "Aliases": "iso2022jp-2004, iso-2022-jp-2004",
        "Languages": "Japanese"
    },
    {
        "Codec": "iso2022_jp_3",
        "Aliases": "iso2022jp-3, iso-2022-jp-3",
        "Languages": "Japanese"
    },
    {
        "Codec": "iso2022_jp_ext",
        "Aliases": "iso2022jp-ext, iso-2022-jp-ext",
        "Languages": "Japanese"
    },
    {
        "Codec": "iso2022_kr",
        "Aliases": "csiso2022kr, iso2022kr, iso-2022-kr",
        "Languages": "Korean"
    },
    {
        "Codec": "latin_1",
        "Aliases": "iso-8859-1, iso8859-1, 8859, cp819, latin, latin1, L1",
        "Languages": "Western Europe"
    },
    {
        "Codec": "iso8859_2",
        "Aliases": "iso-8859-2, latin2, L2",
        "Languages": "Central and Eastern Europe"
    },
    {
        "Codec": "iso8859_3",
        "Aliases": "iso-8859-3, latin3, L3",
        "Languages": "Esperanto, Maltese"
    },
    {
        "Codec": "iso8859_4",
        "Aliases": "iso-8859-4, latin4, L4",
        "Languages": "Baltic languages"
    },
    {
        "Codec": "iso8859_5",
        "Aliases": "iso-8859-5, cyrillic",
        "Languages": "Bulgarian, Byelorussian, Macedonian, Russian, Serbian"
    },
    {
        "Codec": "iso8859_6",
        "Aliases": "iso-8859-6, arabic",
        "Languages": "Arabic"
    },
    {
        "Codec": "iso8859_7",
        "Aliases": "iso-8859-7, greek, greek8",
        "Languages": "Greek"
    },
    {
        "Codec": "iso8859_8",
        "Aliases": "iso-8859-8, hebrew",
        "Languages": "Hebrew"
    },
    {
        "Codec": "iso8859_9",
        "Aliases": "iso-8859-9, latin5, L5",
        "Languages": "Turkish"
    },
    {
        "Codec": "iso8859_10",
        "Aliases": "iso-8859-10, latin6, L6",
        "Languages": "Nordic languages"
    },
    {
        "Codec": "iso8859_11",
        "Aliases": "iso-8859-11, thai",
        "Languages": "Thai languages"
    },
    {
        "Codec": "iso8859_13",
        "Aliases": "iso-8859-13, latin7, L7",
        "Languages": "Baltic languages"
    },
    {
        "Codec": "iso8859_14",
        "Aliases": "iso-8859-14, latin8, L8",
        "Languages": "Celtic languages"
    },
    {
        "Codec": "iso8859_15",
        "Aliases": "iso-8859-15, latin9, L9",
        "Languages": "Western Europe"
    },
    {
        "Codec": "iso8859_16",
        "Aliases": "iso-8859-16, latin10, L10",
        "Languages": "South-Eastern Europe"
    },
    {
        "Codec": "johab",
        "Aliases": "cp1361, ms1361",
        "Languages": "Korean"
    },
    {
        "Codec": "koi8_r",
        "Aliases": None,
        "Languages": "Russian"
    },
    {
        "Codec": "koi8_t",
        "Aliases": None,
        "Languages": "Tajik New in version 3.5."
    },
    {
        "Codec": "koi8_u",
        "Aliases": None,
        "Languages": "Ukrainian"
    },
    {
        "Codec": "kz1048",
        "Aliases": "kz_1048, strk1048_2002, rk1048",
        "Languages": "Kazakh New in version 3.5."
    },
    {
        "Codec": "mac_cyrillic",
        "Aliases": "maccyrillic",
        "Languages": "Bulgarian, Byelorussian, Macedonian, Russian, Serbian"
    },
    {
        "Codec": "mac_greek",
        "Aliases": "macgreek",
        "Languages": "Greek"
    },
    {
        "Codec": "mac_iceland",
        "Aliases": "maciceland",
        "Languages": "Icelandic"
    },
    {
        "Codec": "mac_latin2",
        "Aliases": "maclatin2, maccentraleurope, mac_centeuro",
        "Languages": "Central and Eastern Europe"
    },
    {
        "Codec": "mac_roman",
        "Aliases": "macroman, macintosh",
        "Languages": "Western Europe"
    },
    {
        "Codec": "mac_turkish",
        "Aliases": "macturkish",
        "Languages": "Turkish"
    },
    {
        "Codec": "ptcp154",
        "Aliases": "csptcp154, pt154, cp154, cyrillic-asian",
        "Languages": "Kazakh"
    },
    {
        "Codec": "shift_jis",
        "Aliases": "csshiftjis, shiftjis, sjis, s_jis",
        "Languages": "Japanese"
    },
    {
        "Codec": "shift_jis_2004",
        "Aliases": "shiftjis2004, sjis_2004, sjis2004",
        "Languages": "Japanese"
    },
    {
        "Codec": "shift_jisx0213",
        "Aliases": "shiftjisx0213, sjisx0213, s_jisx0213",
        "Languages": "Japanese"
    },
    {
        "Codec": "utf_32",
        "Aliases": "U32, utf32",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_32_be",
        "Aliases": "UTF-32BE",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_32_le",
        "Aliases": "UTF-32LE",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_16",
        "Aliases": "U16, utf16",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_16_be",
        "Aliases": "UTF-16BE",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_16_le",
        "Aliases": "UTF-16LE",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_7",
        "Aliases": "U7, unicode-1-1-utf-7",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_8",
        "Aliases": "U8, UTF, utf8, cp65001",
        "Languages": "all languages"
    },
    {
        "Codec": "utf_8_sig",
        "Aliases": None,
        "Languages": "all languages"
    }
]
