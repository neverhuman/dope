"""Guard the dated, offline corpus preparation exception."""
import ast
import unittest
from pathlib import Path


class BoundaryTests(unittest.TestCase):
    def test_preparation_has_no_product_routes_database_or_product_imports(self):
        root = Path(__file__).resolve().parents[3]
        package = root / "research/landmark_cache"
        forbidden = {"sqlite3", "psycopg", "sqlalchemy", "django", "flask", "fastapi",
                     "dope_kernel"}
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name.split(".")[0] for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".")[0]]
                self.assertFalse(forbidden.intersection(modules), path.name)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"exec", "eval", "__import__"}, path.name)
        # The product cannot reach this offline helper through either Python boundary.
        for directory in [root / "src", root / "validation/external"]:
            for path in directory.rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
                    names = [i.name for i in node.names] if isinstance(node, ast.Import) else []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names.append(node.module)
                    self.assertFalse(any(n.startswith("research.landmark_cache") for n in names),
                                     str(path.relative_to(root)))
