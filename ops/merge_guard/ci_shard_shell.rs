//! Conservative shell tokenization used by the frozen command coverage proof.
pub(super) fn shell_words(source: &str) -> Result<Vec<String>, &'static str> {
    let expanded = source.replace("\\\n", " ");
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut escape = false;
    let mut active = false;
    for c in expanded.chars() {
        if escape {
            word.push(c);
            escape = false;
            active = true;
            continue;
        }
        if c == '\\' && quote != Some('\'') {
            escape = true;
            active = true;
            continue;
        }
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => word.push(c),
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                active = true;
            }
            None if c.is_ascii_whitespace() => {
                if active {
                    words.push(std::mem::take(&mut word));
                    active = false;
                }
            }
            None => {
                word.push(c);
                active = true;
            }
        }
    }
    if escape || quote.is_some() {
        return Err("command_syntax_uncomparable");
    }
    if active {
        words.push(word);
    }
    Ok(words)
}
