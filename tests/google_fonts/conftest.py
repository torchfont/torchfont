import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--google-fonts",
        action="store_true",
        default=False,
        help="Run tests that scan the Google Fonts corpus",
    )
    parser.addoption(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit Google Fonts tests to N samples (default: all)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--google-fonts"):
        return
    skip = pytest.mark.skip(reason="pass --google-fonts to run")
    for item in items:
        if "google_fonts" in item.keywords:
            item.add_marker(skip)
