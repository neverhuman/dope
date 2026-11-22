//! Argument-vector templates of an external arm (`dope_kpi emit-cli`).
//!
//! Each argument is text with `{name}` placeholders. A template is parsed and
//! checked once, when the arm file is loaded: an unknown placeholder, or one
//! that has no value in its phase, is an error before any command runs.
//! Rendering substitutes inside each argument and never joins or re-splits
//! arguments, so no value can change the shape of a command.

/// The two commands of an arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Fit,
    Sample,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Fit => "fit",
            Phase::Sample => "sample",
        }
    }

    /// The placeholder a command cannot do its job without.
    fn required(self) -> Var {
        match self {
            Phase::Fit => Var::Packet,
            Phase::Sample => Var::Out,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Var {
    Bin,
    FitRows,
    Packet,
    Card,
    Out,
    Rows,
    Seed,
    FitSeed,
    Task,
    Task2,
    Classes,
    Budget,
}

const VARS: [(&str, Var); 12] = [
    ("bin", Var::Bin),
    ("fit_rows", Var::FitRows),
    ("packet", Var::Packet),
    ("card", Var::Card),
    ("out", Var::Out),
    ("rows", Var::Rows),
    ("seed", Var::Seed),
    ("fit_seed", Var::FitSeed),
    ("task", Var::Task),
    ("task2", Var::Task2),
    ("classes", Var::Classes),
    ("budget", Var::Budget),
];

impl Var {
    fn name(self) -> &'static str {
        VARS.iter()
            .find(|(_, var)| *var == self)
            .map_or("", |v| v.0)
    }

    /// The fit rows are gone before sampling; the output file, the row count
    /// and the sample seed do not exist while fitting.
    fn available(self, phase: Phase) -> bool {
        match self {
            Var::FitRows => phase == Phase::Fit,
            Var::Out | Var::Rows | Var::Seed => phase == Phase::Sample,
            _ => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Piece {
    Text(String),
    Var(Var),
}

/// Values substituted into one command.
#[derive(Clone, Debug)]
pub struct Values<'a> {
    /// Absolute path of the arm's executable.
    pub bin: &'a str,
    pub fit_rows: &'a str,
    pub packet: &'a str,
    pub card: &'a str,
    pub out: &'a str,
    pub rows: u64,
    pub seed: u32,
    pub fit_seed: u32,
    /// `regression`, `binary` or `multiclass`.
    pub task: &'a str,
    /// 0 for regression, else the number of classes.
    pub classes: u32,
    /// `budget_bytes` of the arm; a template that uses `{budget}` requires it.
    pub budget: Option<u64>,
}

/// One parsed argument vector; element 0 is the program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Template(Vec<Vec<Piece>>);

/// Split one argument into text and placeholders. `{name}` is a placeholder
/// when `name` is made of letters, digits and `_`; any other brace is text.
/// The error is the unknown placeholder name.
fn parse_arg(text: &str) -> Result<Vec<Piece>, String> {
    let mut pieces = Vec::new();
    let mut literal = String::new();
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let name_len = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if name_len == 0 || !after[name_len..].starts_with('}') {
            literal.push_str(&rest[..=open]);
            rest = after;
            continue;
        }
        let name = &after[..name_len];
        let Some((_, var)) = VARS.iter().find(|(known, _)| *known == name) else {
            return Err(name.to_string());
        };
        literal.push_str(&rest[..open]);
        if !literal.is_empty() {
            pieces.push(Piece::Text(std::mem::take(&mut literal)));
        }
        pieces.push(Piece::Var(*var));
        rest = &after[name_len + 1..];
    }
    literal.push_str(rest);
    if !literal.is_empty() {
        pieces.push(Piece::Text(literal));
    }
    Ok(pieces)
}

impl Template {
    /// Parse the argument vector of `phase`. The error message starts with
    /// the phase name and the argument position; it holds template text only.
    pub fn parse(args: &[String], phase: Phase, has_budget: bool) -> Result<Self, String> {
        let label = phase.name();
        if args.first().is_none_or(|program| program.is_empty()) {
            return Err(format!("{label}: the command is empty"));
        }
        let mut template = Vec::with_capacity(args.len());
        for (position, arg) in args.iter().enumerate() {
            let pieces = parse_arg(arg)
                .map_err(|name| format!("{label}[{position}]: unknown placeholder {{{name}}}"))?;
            for piece in &pieces {
                let Piece::Var(var) = piece else { continue };
                if !var.available(phase) {
                    return Err(format!(
                        "{label}[{position}]: {{{}}} has no value in the {label} command",
                        var.name()
                    ));
                }
                if *var == Var::Budget && !has_budget {
                    return Err(format!(
                        "{label}[{position}]: {{budget}} needs budget_bytes"
                    ));
                }
            }
            template.push(pieces);
        }
        let required = phase.required();
        if !template
            .iter()
            .flatten()
            .any(|p| *p == Piece::Var(required))
        {
            return Err(format!(
                "{label}: the command must use {{{}}}",
                required.name()
            ));
        }
        Ok(Self(template))
    }

    /// The argument vector with every placeholder replaced by its value.
    pub fn render(&self, values: &Values) -> Vec<String> {
        let task2 = if values.task == "regression" {
            "regression"
        } else {
            "classification"
        };
        let value = |var: Var| match var {
            Var::Bin => values.bin.to_string(),
            Var::FitRows => values.fit_rows.to_string(),
            Var::Packet => values.packet.to_string(),
            Var::Card => values.card.to_string(),
            Var::Out => values.out.to_string(),
            Var::Rows => values.rows.to_string(),
            Var::Seed => values.seed.to_string(),
            Var::FitSeed => values.fit_seed.to_string(),
            Var::Task => values.task.to_string(),
            Var::Task2 => task2.to_string(),
            Var::Classes => values.classes.to_string(),
            Var::Budget => values.budget.unwrap_or(0).to_string(),
        };
        let argument = |pieces: &Vec<Piece>| {
            let parts = pieces.iter().map(|piece| match piece {
                Piece::Text(text) => text.clone(),
                Piece::Var(var) => value(*var),
            });
            parts.collect::<String>()
        };
        self.0.iter().map(argument).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn placeholders_are_split_from_text_and_other_braces_stay_literal() {
        let text = |s: &str| Piece::Text(s.to_string());
        assert_eq!(
            parse_arg("--out={out}.tmp").unwrap(),
            [text("--out="), Piece::Var(Var::Out), text(".tmp")]
        );
        assert_eq!(
            parse_arg("{seed}{fit_seed}").unwrap(),
            [Piece::Var(Var::Seed), Piece::Var(Var::FitSeed)]
        );
        assert!(parse_arg("").unwrap().is_empty());
        for literal in ["plain", "{}", "{", "}{", "{a b}", "{\"k\":1}", "{rows"] {
            assert_eq!(parse_arg(literal).unwrap(), [text(literal)], "{literal}");
        }
        assert_eq!(parse_arg("x{Rows}").unwrap_err(), "Rows");
        assert_eq!(parse_arg("{budget_bytes}").unwrap_err(), "budget_bytes");
        for (name, var) in VARS {
            assert_eq!(
                parse_arg(&format!("{{{name}}}")).unwrap(),
                [Piece::Var(var)]
            );
            assert_eq!(var.name(), name);
        }
    }

    #[test]
    fn templates_are_checked_per_phase() {
        let fit = |list: &[&str]| Template::parse(&args(list), Phase::Fit, false);
        let sample = |list: &[&str]| Template::parse(&args(list), Phase::Sample, false);
        assert!(fit(&[
            "{bin}",
            "fit",
            "{fit_rows}",
            "{packet}",
            "{card}",
            "{fit_seed}"
        ])
        .is_ok());
        assert!(sample(&["{bin}", "{packet}", "{card}", "{out}", "{rows}", "{seed}"]).is_ok());
        assert!(sample(&[
            "python3",
            "{bin}",
            "{out}",
            "{task}",
            "{task2}",
            "{classes}"
        ])
        .is_ok());
        let cases: [(Result<Template, String>, &str); 10] = [
            (
                fit(&["{bin}", "{pakcet}"]),
                "fit[1]: unknown placeholder {pakcet}",
            ),
            (fit(&[]), "fit: the command is empty"),
            (fit(&["", "{packet}"]), "fit: the command is empty"),
            (fit(&["{bin}"]), "fit: the command must use {packet}"),
            (
                fit(&["{bin}", "{packet}", "{out}"]),
                "fit[2]: {out} has no value in the fit command",
            ),
            (
                fit(&["{bin}", "{packet}", "-n{rows}"]),
                "fit[2]: {rows} has no value in the fit command",
            ),
            (
                fit(&["{bin}", "{packet}", "{seed}"]),
                "fit[2]: {seed} has no value in the fit command",
            ),
            (
                sample(&["{bin}", "{out}", "{fit_rows}"]),
                "sample[2]: {fit_rows} has no value in the sample command",
            ),
            (
                sample(&["{bin}", "{packet}"]),
                "sample: the command must use {out}",
            ),
            (
                fit(&["{bin}", "{packet}", "{budget}"]),
                "fit[2]: {budget} needs budget_bytes",
            ),
        ];
        for (parsed, message) in cases {
            assert_eq!(parsed.unwrap_err(), message);
        }
        let budgeted = args(&["{bin}", "{packet}", "{budget}"]);
        assert!(Template::parse(&budgeted, Phase::Fit, true).is_ok());
    }

    #[test]
    fn rendering_substitutes_inside_each_argument() {
        let values = Values {
            bin: "/opt/gen",
            fit_rows: "/w/fit rows.csv",
            packet: "/p/x.packet",
            card: "/p/x.card",
            out: "/w/out.csv",
            rows: 4096,
            seed: 7,
            fit_seed: 9,
            task: "multiclass",
            classes: 5,
            budget: Some(2048),
        };
        let fit = args(&[
            "{bin}",
            "fit",
            "{fit_rows}",
            "--task={task}/{task2}",
            "-k{classes}",
            "{budget}",
            "{fit_seed}",
            "{packet}",
            "{card}",
        ]);
        let fit = Template::parse(&fit, Phase::Fit, true).unwrap();
        assert_eq!(
            fit.render(&values),
            [
                "/opt/gen",
                "fit",
                // One argument stays one argument, whatever its value holds.
                "/w/fit rows.csv",
                "--task=multiclass/classification",
                "-k5",
                "2048",
                "9",
                "/p/x.packet",
                "/p/x.card"
            ]
        );
        let sample = args(&["env", "{bin}", "{out}", "{rows}", "{seed}", "{task2}", "{x"]);
        let sample = Template::parse(&sample, Phase::Sample, false).unwrap();
        let regression = Values {
            task: "regression",
            classes: 0,
            ..values
        };
        assert_eq!(
            sample.render(&regression),
            [
                "env",
                "/opt/gen",
                "/w/out.csv",
                "4096",
                "7",
                "regression",
                "{x"
            ]
        );
        let binary = Values {
            task: "binary",
            ..regression
        };
        assert_eq!(sample.render(&binary)[5], "classification");
    }
}
