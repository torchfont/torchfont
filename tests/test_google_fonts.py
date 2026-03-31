import shutil
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest

from torchfont.datasets import google_fonts as google_fonts_module
from torchfont.datasets import GoogleFonts


@pytest.fixture
def clean_clone_dir(tmp_path: Path) -> Generator[Path, None, None]:
    clone_dir = tmp_path / "google_fonts_test"
    clone_dir.mkdir(parents=True, exist_ok=True)
    yield clone_dir
    if clone_dir.exists():
        shutil.rmtree(clone_dir)


@pytest.mark.network
@pytest.mark.slow
def test_google_fonts_fresh_clone(clean_clone_dir: Path) -> None:
    dataset = GoogleFonts(
        root=clean_clone_dir,
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    assert (clean_clone_dir / ".git").exists()
    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert dataset.url == "https://github.com/google/fonts"
    assert dataset.ref == "main"
    assert dataset.commit_hash is not None
    assert dataset.patterns == ("ufl/*/*.ttf",)
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample.types is not None
    assert sample.coords is not None
    assert sample.style_idx is not None
    assert sample.content_idx is not None


@pytest.mark.network
@pytest.mark.slow
def test_google_fonts_existing_clone() -> None:
    root = Path("data/google/fonts")

    dataset1 = GoogleFonts(
        root=root,
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    commit_hash_1 = dataset1.commit_hash

    dataset2 = GoogleFonts(
        root=root,
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=False,
    )

    commit_hash_2 = dataset2.commit_hash

    assert commit_hash_1 == commit_hash_2
    assert len(dataset1.style_classes) == len(dataset2.style_classes)
    assert len(dataset1) == len(dataset2)


@pytest.mark.network
@pytest.mark.slow
def test_google_fonts_update_existing(clean_clone_dir: Path) -> None:
    dataset1 = GoogleFonts(
        root=clean_clone_dir,
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    first_commit = dataset1.commit_hash

    dataset2 = GoogleFonts(
        root=clean_clone_dir,
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    second_commit = dataset2.commit_hash

    assert first_commit is not None
    assert second_commit is not None
    assert len(first_commit) == len(second_commit)


@pytest.mark.network
@pytest.mark.slow
def test_google_fonts_different_patterns() -> None:
    root = Path("data/google/fonts")

    dataset_apache = GoogleFonts(
        root=root,
        ref="main",
        patterns=("apache/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    dataset_ofl = GoogleFonts(
        root=root,
        ref="main",
        patterns=("ofl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=False,
    )

    assert len(dataset_apache) > 0
    assert len(dataset_ofl) > 0
    assert len(dataset_ofl) > len(dataset_apache)


@pytest.mark.network
@pytest.mark.slow
def test_google_fonts_custom_codepoint_filter() -> None:
    root = Path("data/google/fonts")

    dataset = GoogleFonts(
        root=root,
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x30, 0x3A),
        download=True,
    )

    assert len(dataset.content_classes) <= 10
    assert len(dataset) > 0


def test_google_fonts_uses_default_patterns_when_not_specified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def fake_font_repo_init(
        self: GoogleFonts,
        *,
        root: Path | str,
        url: str,
        ref: str,
        patterns: tuple[str, ...],
        codepoint_filter: range | None = None,
        transform: object | None = None,
        download: bool = False,
        depth: int = 1,
    ) -> None:
        called["root"] = root
        called["url"] = url
        called["ref"] = ref
        called["patterns"] = patterns
        called["codepoint_filter"] = codepoint_filter
        called["transform"] = transform
        called["download"] = download
        called["depth"] = depth

    monkeypatch.setattr(
        google_fonts_module.FontRepo,
        "__init__",
        fake_font_repo_init,
    )

    GoogleFonts(root="data/google/fonts", ref="main")

    assert called["root"] == "data/google/fonts"
    assert called["url"] == google_fonts_module.REPO_URL
    assert called["ref"] == "main"
    assert called["patterns"] == google_fonts_module.DEFAULT_PATTERNS
    assert called["codepoint_filter"] is None
    assert called["transform"] is None
    assert called["download"] is False
    assert called["depth"] == 1


def test_google_fonts_repr_includes_patterns() -> None:
    dataset = object.__new__(GoogleFonts)
    dataset.root = Path("tests/fonts").resolve()
    dataset.url = google_fonts_module.REPO_URL
    dataset.ref = "main"
    dataset.commit_hash = "abc123"
    dataset.patterns = ("ufl/*/*.ttf",)
    dataset._dataset = SimpleNamespace(
        sample_count=10,
        style_class_count=3,
        content_class_count=7,
    )

    assert repr(dataset) == (
        "GoogleFonts("
        f"root={str(dataset.root)!r}, "
        f"url={dataset.url!r}, "
        f"ref={dataset.ref!r}, "
        f"commit={dataset.commit_hash!r}, "
        f"patterns={dataset.patterns!r}, "
        "samples=10, "
        "styles=3, "
        "content_classes=7)"
    )
