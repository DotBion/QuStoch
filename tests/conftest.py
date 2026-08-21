import sys
from pathlib import Path

# The application modules live in src/ and are imported flat (no package), so make them
# importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
