"""Paper-number tracing rejects a drifted watch log and an untraced decimal."""

import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.verify_paper_numbers import (
    REPO,
    main,
    manuscript_for_scan,
    parse_watch,
    untraced_numbers,
)


class VerifyPaperNumbers(unittest.TestCase):
    def test_watch_uses_the_last_counted_record(self):
        text = "\n".join(
            [
                '{"closed": 1, "planned": 800, "ok": 1, "new_generator_fits_started": 0}',
                '{"closed": 799, "planned": 800, "ok": 799, "new_generator_fits_started": 0, "utc": "t"}',
                '{"phase": "actual_reconciliation_exit", "exit_code": 0}',
            ]
        )
        watched = parse_watch(text)
        self.assertEqual(watched["closed"], 799)
        self.assertEqual(watched["planned"], 800)
        self.assertEqual(watched["terminal_exit_code"], 0)

    def test_untraced_decimal_is_reported(self):
        missing = untraced_numbers("The median is $0.12345$. Width 16 stays.", {"16"})
        self.assertEqual(missing, ["0.12345"])

    def test_style_filename_is_not_a_measurement(self):
        tex = "\\usepackage{iclr2027_conference,times}\nWidth 16 stays.\n"
        missing = untraced_numbers(manuscript_for_scan(tex), {"16"})
        self.assertEqual(missing, [])

    def test_generated_inputs_are_not_scanned(self):
        tex = "\\input{generated/numbers.tex}\nSee $0.12345$ in the table.\n"
        # The input line is skipped. A decimal on the next line is still visible.
        missing = untraced_numbers(tex, set())
        self.assertEqual(missing, ["0.12345"])

    def test_verifier_opens_only_committed_repo_files(self):
        opened = []
        real_open = Path.open

        def tracking_open(path, *args, **kwargs):
            opened.append(Path(path))
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", tracking_open):
            status = main([])
        self.assertEqual(status, 0)
        self.assertTrue(opened)
        repo = REPO.resolve()
        for path in opened:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(repo)
            except ValueError:
                self.fail(f"opened a path outside the repo: {path}")
            self.assertNotIn(".agent", relative.parts, path)
            self.assertNotIn("target", relative.parts, path)


if __name__ == "__main__":
    unittest.main()
