"""Read the package version from Cargo without importing the research runtime."""
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 build isolation
    import tomli as tomllib

VERSION = tomllib.loads((Path(__file__).resolve().parents[2] / 'Cargo.toml').read_text())['package']['version']
