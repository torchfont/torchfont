import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

from torchfont.datasets import FontRepo


@pytest.fixture
def clean_repo_dir(tmp_path: Path) -> Generator[Path, None, None]:
    repo_dir = tmp_path / "font_repo_test"
    repo_dir.mkdir(parents=True, exist_ok=True)
    yield repo_dir
    if repo_dir.exists():
        shutil.rmtree(repo_dir)


@pytest.mark.network
@pytest.mark.slow
def test_font_repo_init(clean_repo_dir: Path) -> None:
    dataset = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    assert (clean_repo_dir / ".git").exists()
    assert dataset.url == "https://github.com/google/fonts"
    assert dataset.ref == "main"
    assert dataset.commit_hash is not None
    assert dataset.patterns == ("ufl/*/*.ttf",)
    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


@pytest.mark.network
@pytest.mark.slow
def test_font_repo_reuse_existing(clean_repo_dir: Path) -> None:
    dataset1 = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    commit1 = dataset1.commit_hash

    dataset2 = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=False,
    )

    commit2 = dataset2.commit_hash

    assert commit1 == commit2
    assert len(dataset1) == len(dataset2)


@pytest.mark.network
@pytest.mark.slow
def test_font_repo_different_patterns(clean_repo_dir: Path) -> None:
    dataset_ufl = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    dataset_apache = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("apache/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=False,
    )

    assert len(dataset_ufl) > 0
    assert len(dataset_apache) > 0
    assert dataset_ufl.commit_hash == dataset_apache.commit_hash


@pytest.mark.network
@pytest.mark.slow
def test_font_repo_getitem(clean_repo_dir: Path) -> None:
    dataset = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x41, 0x5B),
        download=True,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert sample.types.dtype == torch.long
    assert sample.types.ndim == 1
    assert sample.coords.dtype == torch.float32
    assert sample.coords.ndim == 2
    assert sample.coords.shape[1] == 6
    assert isinstance(sample.style_idx, int)
    assert isinstance(sample.content_idx, int)


@pytest.mark.network
@pytest.mark.slow
def test_font_repo_update_ref(clean_repo_dir: Path) -> None:
    dataset1 = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    main_commit = dataset1.commit_hash

    dataset2 = FontRepo(
        root=clean_repo_dir,
        url="https://github.com/google/fonts",
        ref="main",
        patterns=("ufl/*/*.ttf",),
        codepoint_filter=range(0x80),
        download=True,
    )

    assert dataset2.commit_hash is not None
    assert main_commit is not None
    assert len(main_commit) == len(dataset2.commit_hash)
