import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize(
    "script_name",
    [
        "dataloader.py",
        "dataset.py",
        "torchvision_transform.py",
        "transform.py",
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
