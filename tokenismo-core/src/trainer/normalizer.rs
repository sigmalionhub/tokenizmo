use unicode_normalization::UnicodeNormalization;

// Zero-width and directional control characters to strip entirely.
const STRIP_CHARS: &[char] = &[
    '\u{200B}', // ZERO WIDTH SPACE
    '\u{200C}', // ZERO WIDTH NON-JOINER
    '\u{200D}', // ZERO WIDTH JOINER
    '\u{200E}', // LEFT-TO-RIGHT MARK
    '\u{200F}', // RIGHT-TO-LEFT MARK
    '\u{202A}', // LEFT-TO-RIGHT EMBEDDING
    '\u{202B}', // RIGHT-TO-LEFT EMBEDDING
    '\u{202C}', // POP DIRECTIONAL FORMATTING
    '\u{202D}', // LEFT-TO-RIGHT OVERRIDE
    '\u{202E}', // RIGHT-TO-LEFT OVERRIDE
    '\u{FEFF}', // BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
    '\u{00AD}', // SOFT HYPHEN
];

pub fn normalize(text: &str) -> String {
    // Strip zero-width/directional characters.
    let stripped: String = text.chars().filter(|c| !STRIP_CHARS.contains(c)).collect();

    // Normalize line endings to \n.
    let lf = stripped.replace("\r\n", "\n").replace('\r', "\n");

    // Collapse 3+ consecutive newlines to \n\n.
    let collapsed = collapse_newlines(&lf);

    // NFC Unicode normalization.
    collapsed.nfc().collect()
}

fn collapse_newlines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut newline_count = 0usize;
    for ch in s.chars() {
        if ch == '\n' {
            newline_count += 1;
        } else {
            if newline_count >= 3 {
                out.push_str("\n\n");
            } else {
                for _ in 0..newline_count {
                    out.push('\n');
                }
            }
            newline_count = 0;
            out.push(ch);
        }
    }
    if newline_count >= 3 {
        out.push_str("\n\n");
    } else {
        for _ in 0..newline_count {
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_normalization() {
        // "é" as NFD (e + combining acute) should normalize to NFC (single codepoint).
        let nfd = "e\u{0301}";
        let result = normalize(nfd);
        assert_eq!(result, "\u{00E9}");
    }

    #[test]
    fn strips_zero_width() {
        let input = "hello\u{200B}world";
        assert_eq!(normalize(input), "helloworld");
    }

    #[test]
    fn normalizes_crlf() {
        assert_eq!(normalize("a\r\nb"), "a\nb");
        assert_eq!(normalize("a\rb"), "a\nb");
    }

    #[test]
    fn collapses_many_newlines() {
        assert_eq!(normalize("a\n\n\n\nb"), "a\n\nb");
        assert_eq!(normalize("a\n\nb"), "a\n\nb"); // 2 = keep
    }
}
