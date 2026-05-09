from __future__ import annotations

import pickle
import runpy
from pathlib import Path

import pytest

from examples.google_fonts import collate_fn as google_fonts_collate_fn

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


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
    assert callable(namespace.get("main")), (
        f"{script_name} must define a callable main()"
    )


@pytest.mark.parametrize(
    "script_name",
    [
        "local_fonts.py",
        "subset_by_targets.py",
    ],
)
def test_local_examples_run_as_scripts(script_name: str) -> None:
    runpy.run_path(str(EXAMPLES_DIR / script_name), run_name="__main__")


def test_google_fonts_collate_fn_is_picklable_for_dataloader_workers() -> None:
    pickle.dumps(google_fonts_collate_fn)
