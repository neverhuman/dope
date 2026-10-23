use serde::Serialize;

// This implementation uses a new identity deliberately: fold-local scaling,
// nested lambda selection and the bounded T03 policy differ from the proposal.
pub const PANEL_V1: &str = "landmarks-native-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Additive,
    Interaction,
    Kernel,
    Missingness,
    Dominance,
    Tree,
    Local,
    Neural,
    LearningCurve,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LandmarkId(pub u8);

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Recipe {
    pub id: LandmarkId,
    pub name: &'static str,
    pub family: Family,
}

pub const PANEL: [Recipe; 23] = [
    Recipe {
        id: LandmarkId(1),
        name: "lin",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(2),
        name: "log",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(3),
        name: "sqrt",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(4),
        name: "pow",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(5),
        name: "sat",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(6),
        name: "abs",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(7),
        name: "hinge",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(8),
        name: "step",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(9),
        name: "periodic",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(10),
        name: "bump",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(11),
        name: "rows",
        family: Family::Additive,
    },
    Recipe {
        id: LandmarkId(12),
        name: "pairs",
        family: Family::Interaction,
    },
    Recipe {
        id: LandmarkId(13),
        name: "rff",
        family: Family::Kernel,
    },
    Recipe {
        id: LandmarkId(14),
        name: "missing",
        family: Family::Missingness,
    },
    Recipe {
        id: LandmarkId(15),
        name: "best1",
        family: Family::Dominance,
    },
    Recipe {
        id: LandmarkId(16),
        name: "stump",
        family: Family::Tree,
    },
    Recipe {
        id: LandmarkId(17),
        name: "otree3",
        family: Family::Tree,
    },
    Recipe {
        id: LandmarkId(18),
        name: "gbdt",
        family: Family::Tree,
    },
    Recipe {
        id: LandmarkId(19),
        name: "knn1",
        family: Family::Local,
    },
    Recipe {
        id: LandmarkId(20),
        name: "knn16",
        family: Family::Local,
    },
    Recipe {
        id: LandmarkId(21),
        name: "net",
        family: Family::Neural,
    },
    Recipe {
        id: LandmarkId(22),
        name: "lin_quarter",
        family: Family::LearningCurve,
    },
    Recipe {
        id: LandmarkId(23),
        name: "gbdt_quarter",
        family: Family::LearningCurve,
    },
];
