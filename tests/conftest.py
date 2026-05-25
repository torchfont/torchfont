import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--google-fonts",
        action="store_true",
        default=False,
        help="Run tests that scan the Google Fonts corpus",
    )
    parser.addoption(
        "--google-fonts-limit",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Cap Google Fonts tests at N samples (default: 1000000; 0 = no limit)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--google-fonts"):
        return
    skip = pytest.mark.skip(reason="pass --google-fonts to run")
    for item in items:
        if "google_fonts_full" in item.keywords:
            item.add_marker(skip)
