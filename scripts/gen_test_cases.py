"""Generate data/test_cases.json — extreme stress-test cases for the tokenizer."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

cases = [
    # ── 1. EXTREME INDENTATION ─────────────────────────────────────────────
    " word",
    "  word",
    "   word",
    "    word",
    "       word",
    "        word",
    "         word",
    " a  b   c    d     e      f       g        h",
    "word  double  spaces  between  every  word",
    "    mixed  indents   here     yes",
    " ",
    "  ",
    "    ",
    "        ",
    " " * 16,
    " " * 32,
    " " * 64,
    " " * 128,
    " " * 256,
    "\t \t  \t    if x:",
    "\t\t\tdef foo():\n\t    return 1",
    "trailing spaces   \r\n    next line",
    "   \r\n",
    "end of file   ",

    # ── 2. UTF-8 BOUNDARIES ────────────────────────────────────────────────
    " \U0001f980 ",                             # 🦀 with spaces
    "    漢 ",                              # 漢 with 4-space indent
    "\U0001f525\t\U0001f525",                   # 🔥 tab 🔥
    "    \U0001f1f7\U0001f1fa    \U0001f1fa\U0001f1f8",  # flag emojis
    " \xe9 \xe0 \xfc \xf1 ō ",            # é à ü ñ ō
    # Zalgo combining characters
    "z̴̡̢̨̫͓͚̗̮̙͔̘"
    "̱͕̱̗̙̞͓̹̜̚̚͜"
    "ḁ̡̯͔̘̖̬̥l͓̼̲"
    "g͓̲͔̙̜͕͓̞̱̗̚"
    "o͙",
    "ПреведMedvedКакДелаWtfLolOmgSimdOptimization",
    "函数def関数한수fonctionфункцияFunktion",
    "مرحبابالعالم"
    "HelloWorldПриветМир"
    "こんにちは世界",

    # ── 3. CACHE OVERFLOW / LONG WORDS ────────────────────────────────────
    "a" * 512,                                  # 512-char word, stresses DP
    "abcde abcde abcde abcde abcde abcde  abcde   abcde    abcde     abcde",
    "word word word word  word word word   word word word    word word word     word",
    # minified JSON ~400 chars
    '{"a":1,"b":{"c":[1,2,3],"d":{"e":{"f":{"g":[{"h":"'
    + "i" * 200
    + '","j":null,"k":true,"l":false,"m":3.14159265358979323846}]}}}}},"n":"end"}',
    # minified JS ~300 chars
    (
        "function f(x){return x.split('').map(c=>c.charCodeAt(0))"
        ".filter(n=>n>32).reduce((a,b)=>a+b,0).toString(16)"
        ".padStart(8,'0')}const r=f('hello world test string with "
        "spaces and more content here to make it longer and longer');"
    ),

    # ── 4. CODE SYNTAX AND MIXED LANGUAGES ────────────────────────────────
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" * 3,
    (
        "fn encode(s: &str) -> Vec<u32> { s.split_whitespace()"
        ".flat_map(|w| w.match_indices(' ').collect::<Vec<_>>())"
        ".map(|(i, _)| i as u32).collect() }"
    ),
    (
        'let x = foo.unwrap().match_indices(" ").collect::<Vec<_>>()'
        ".iter().map(|(i, s)| (*i, s.len())).filter(|(_, n)| *n > 1).count();"
    ),
    (
        "impl<T: Send + Sync + 'static> Encoder<T>"
        " where T: Fn(&[u8]) -> Vec<u32> + Clone {\n"
        "    fn new(f: T) -> Self { Self { inner: Arc::new(f) } }\n}"
    ),
    (
        "        // Проверяем тут unsafe логику\n"
        "        let ptr = unsafe { slice.get_unchecked(i) };\n"
        "        // Если i >= len — сегфолт\n"
        "        debug_assert!(i < slice.len());"
    ),
    (
        "    # Функция с глубокой вложенностью\n"
        "    def foo(x):\n"
        "        if x > 0:\n"
        "            for i in range(x):\n"
        "                while True:\n"
        "                    try:\n"
        "                        yield i\n"
        "                    except StopIteration:\n"
        "                        break"
    ),
    "    def f():  # два пробела между def и #\n        pass",
    "x=1\ny=2\n\n\n\nz=3",
    "\n\n\n\n\n",
    "     \n     \n     \n     \n",
    "a\tb\t\tc\t\t\td",
    "a b  c   d    e     f      g       h        i         j          k",
    "\x01\x02\x03\x1f",                         # non-printable control chars
    "caf\xe9 r\xe9sum\xe9 na\xefve \xc5ngstr\xf6m",
    "\U00010000\U00010001\U00010002\U00010003 Linear B Syllabary",
    "    if True:\n        pass\n    elif False:\n        pass\n    else:\n        pass",
    " " * 256 + "// 256 пробелов потом комментарий",

    # ── 5. MID-WORD OOV — encodable chars after OOV gap must not become <unk>
    "hell\U0001f980o",                          # ASCII + OOV emoji + ASCII
    "fn\U0001f525main()",                       # code token + OOV + code token
    "word\U00010000next",                       # Linear B embedded in ASCII word
]

out = ROOT / "data" / "test_cases.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

# verify round-trip parse
with open(out, encoding="utf-8") as f:
    back = json.load(f)

assert len(back) == len(cases), "count mismatch"
print(f"Written {len(cases)} cases to {out.relative_to(ROOT)}")
