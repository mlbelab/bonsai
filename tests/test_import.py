import subprocess
import sys


def test_import_does_not_load_matplotlib():
    subprocess.run(
        [sys.executable, "-c", "import bonsai, sys; assert 'matplotlib' not in sys.modules"],
        check=True,
    )
