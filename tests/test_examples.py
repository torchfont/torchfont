from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples")


@pytest.mark.parametrize(
    "script_name",
    [
        "google_fonts.py",
        "local_fonts.py",
        "source_han_sans.py",
        "subset_by_targets.py",
        "font_awesome.py",
        "material_design_icons.py",
    ],
)
def test_examples_are_import_safe(
    script_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = runpy.run_path(
        str(EXAMPLES_DIR / script_name),
        run_name="_example_import_check",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert callable(namespace["main"])


@pytest.mark.parametrize(
    "script_name",
    [
        "local_fonts.py",
        "subset_by_targets.py",
    ],
)
def test_local_examples_run_as_scripts(script_name: str) -> None:
    runpy.run_path(str(EXAMPLES_DIR / script_name), run_name="__main__")
