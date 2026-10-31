#!/usr/bin/env python3

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# -------------------------
# Configuration
# -------------------------

ROOT = Path(".").resolve()

PYTHON_EXTENSIONS = {".py"}
NODE_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

PYTHON_STD_LIB = set(sys.stdlib_module_names)

NODE_BUILTINS = {
    "fs", "path", "os", "http", "https", "url", "util",
    "events", "crypto", "stream", "buffer", "child_process",
    "zlib", "net", "tls", "dns", "readline", "tty",
    "worker_threads", "perf_hooks", "timers", "assert"
}

# Common import -> pip package mappings
PYTHON_PACKAGE_MAP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "google": "google-generativeai",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
}

# -------------------------
# Python Scanner
# -------------------------

python_modules = set()

for path in ROOT.rglob("*.py"):
    try:
        tree = ast.parse(path.read_text(encoding="utf8"))
    except Exception:
        continue

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for alias in node.names:
                python_modules.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                python_modules.add(node.module.split(".")[0])

python_modules -= PYTHON_STD_LIB

# -------------------------
# Node Scanner
# -------------------------

node_packages = set()

IMPORT_RE = re.compile(
    r'import.*?from\s+[\'"]([^\'"]+)[\'"]|'
    r'import\s+[\'"]([^\'"]+)[\'"]|'
    r'require\([\'"]([^\'"]+)[\'"]\)'
)

for ext in NODE_EXTENSIONS:
    for path in ROOT.rglob(f"*{ext}"):

        try:
            text = path.read_text(encoding="utf8")
        except Exception:
            continue

        for match in IMPORT_RE.finditer(text):
            pkg = next(filter(None, match.groups()))

            if pkg.startswith("."):
                continue

            pkg = pkg.split("/")[0]

            if pkg.startswith("@"):
                parts = pkg.split("/")
                if len(parts) >= 2:
                    pkg = "/".join(parts[:2])

            if pkg not in NODE_BUILTINS:
                node_packages.add(pkg)

# -------------------------
# Python Versions
# -------------------------

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata

requirements = []

for module in sorted(python_modules):

    package = PYTHON_PACKAGE_MAP.get(module, module)

    try:
        version = metadata.version(package)
        requirements.append(f"{package}=={version}")
    except Exception:
        requirements.append(package)

requirements = sorted(set(requirements))

with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

# -------------------------
# Node Versions
# -------------------------

node_dependencies = {}

for pkg in sorted(node_packages):

    try:
        result = subprocess.run(
            ["npm", "list", pkg, "--depth=0", "--json"],
            capture_output=True,
            text=True,
        )

        data = json.loads(result.stdout)

        deps = data.get("dependencies", {})

        if pkg in deps:
            node_dependencies[pkg] = "^" + deps[pkg]["version"]
        else:
            node_dependencies[pkg] = "*"

    except Exception:
        node_dependencies[pkg] = "*"

generated = {
    "dependencies": dict(sorted(node_dependencies.items()))
}

with open("package.generated.json", "w") as f:
    json.dump(generated, f, indent=2)

print("Generated:")
print("  requirements.txt")
print("  package.generated.json")
