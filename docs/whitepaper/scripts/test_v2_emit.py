"""v2 emitter formatting and the computed sentences, on a small fixture panel."""

import re
import unittest

import method_registry as M
import review_v2_emit as E


def contrast(code, auditor, size, hl, lo, hi, n=97, verdict=None):
    arm = M.BY_CODE[code]
    return {"method": arm.method, "configuration": arm.configuration, "auditor": auditor, "size": size,
            "n": n, "median": hl, "median_lo": lo, "median_hi": hi, "hl": hl, "hl_lo": lo, "hl_hi": hi,
            "wins": 60, "ties": 0, "losses": n - 60, "holm_p": 0.01, "holm_family_n": 12,
            "verdict": verdict or E_verdict(lo, hi), "equivalence": {"equivalent": False}, "yuen": {"equivalent": False}}


def E_verdict(lo, hi):
    return "above" if lo > 0 else ("below" if hi < 0 else "inseparable")


def level(code, auditor, size, median, n=97):
    arm = M.BY_CODE[code]
    return {"method": arm.method, "configuration": arm.configuration, "auditor": auditor, "size": size,
            "n": n, "n_fits": n, "median": median, "lo": median - 0.02, "hi": median + 0.02}


def panel():
    levels, contrasts = [], []
    for code, value in (("Dope", 0.9), ("TabSyn", 0.88), ("Arf", 0.83), ("Gauss", 0.66), ("Pred", 0.64), ("Boot", 1.0)):
        for auditor in ("catboost", "linear", "mlp"):
            for size in (1, 4):
                levels.append(level(code, auditor, size, value + (0.01 if size == 4 else 0.0)))
                if code != "Dope":
                    contrasts.append(contrast(code, auditor, size, 0.9 - value, 0.9 - value - 0.03, 0.9 - value + 0.03))
    return {"levels": levels, "contrasts": contrasts,
            "strongest": {"method": "TabSyn", "configuration": "scaled_200_vae_1000_diffusion", "median": 0.89,
                          "paired_n": 97},
            "seed_rank": {"n": 90, "seed11_top": 30, "expected": 18.0, "p_one_sided": 0.002},
            "seeds_23_71": [{"size": s, "auditor": a, "n": 95, "median": 0.88, "lo": 0.85, "hi": 0.9}
                            for s in (1, 4) for a in ("catboost", "linear", "mlp")],
            "two_by_two": {"n_common": 90, "bhist_seed11": 0.94, "bhist_seeds23_71": 0.9, "bnew_seed11": 0.88,
                           "bnew_seeds23_71": 0.86, "historical_seed11": 0.94},
            "fidelity": [], "official_tests_opened": False}


BYTES = {"DOPE|headline": {"median": 2000.0, "q1": 1500.0, "q3": 2600.0, "n": 97, "within_cap": 95,
                           "fits_ok": 485, "fits_within_cap": 480, "fits_total": 500},
         "TabSyn|scaled_200_vae_1000_diffusion": {"median": 4.2e7, "q1": 4.1e7, "q3": 4.3e7, "n": 100,
                                                  "within_cap": 0},
         "ARF|author_default": {"median": 1.4e6, "q1": 3e5, "q3": 2.6e6, "n": 100, "within_cap": 0},
         "GaussianCopula|native_selected": {"median": 5000.0, "q1": 2000.0, "q3": 20000.0, "n": 100,
                                            "within_cap": 62}}
PRIVACY = {"summaries": [{"method": "DOPE", "configuration": "features12_steps2048", "size": 4,
                          "metric": "c2st_catboost_auc", "median": 0.64, "lo": 0.58, "hi": 0.83, "n": 98}]}


class EmitterTests(unittest.TestCase):
    def test_number_formats(self):
        self.assertEqual(E.num(-0.0001), "0.000")
        self.assertEqual(E.num(None), E.DASH)
        self.assertEqual(E.pval(3.2e-5), "3.20\\times 10^{-5}")
        self.assertEqual(E.exact_bytes(2067), "2{,}067")
        self.assertEqual(E.human_bytes(42_261_045), "42.3\\,\\mathrm{MB}")
        self.assertEqual(E.ratio(21130.5), "2.1\\times 10^{4}")

    def test_macro_names_are_letters_and_unique(self):
        text = E.macros(panel(), BYTES, [], PRIVACY)
        names = re.findall(r"\\newcommand\{\\(\w+)\}", text)
        self.assertTrue(all(name.isalpha() for name in names))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue("\\newcommand{\\VStrongestName}{TabSyn (scaled schedule)}" in text)
        self.assertTrue("\\newcommand{\\VVerdictStrong}{cannot be separated from}" in text)
        self.assertTrue("\\newcommand{\\VVerdictArfCbFour}{is above}" in text)
        self.assertTrue("\\newcommand{\\VVerdictBootCbFour}{is below}" in text)
        self.assertTrue("\\newcommand{\\VDopeBytes}{2{,}000}" in text)

    def test_pareto_and_size_sentences_follow_the_data(self):
        text = E.macros(panel(), BYTES, [], PRIVACY)
        self.assertTrue("\\newcommand{\\VParetoSet}{DOPE (ours)}" in text)
        self.assertTrue("\\newcommand{\\VSizeRankSame}{is the same at both sizes}" in text)
        self.assertTrue("\\newcommand{\\VEquivSentence}{No generator comparison meets that rule.}" in text)

    def test_missing_strongest_refuses(self):
        broken = panel()
        broken["strongest"] = None
        with self.assertRaises(SystemExit):
            E.macros(broken, BYTES, [], PRIVACY)

    def test_tables_lead_with_the_strongest_comparator(self):
        table = E.headline_table(panel(), BYTES)
        body = [line for line in table.splitlines() if line.endswith("\\\\") and "&" in line][1:]
        self.assertTrue(body[0].startswith("TabSyn (scaled schedule)"))
        self.assertIn("\\begin{tabular}[t]", E.size_table(panel()))


if __name__ == "__main__":
    unittest.main()
