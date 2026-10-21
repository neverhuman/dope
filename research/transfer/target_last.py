"""Public intake API for the mandatory target-last table contract."""
from .tdl.target_last import (
    TargetLastError, NormalizedCsv, normalize_csv, normalize_files,
    require_target_last, schema_columns, schema_document, main,
)
__all__ = ["TargetLastError", "NormalizedCsv", "normalize_csv", "normalize_files",
           "require_target_last", "schema_columns", "schema_document"]
if __name__ == "__main__": main()
