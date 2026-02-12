from pathlib import Path

import pygit2
import pytest

from torchfont.io.git import ensure_repo

_AUTHOR = pygit2.Signature("TorchFont Tests", "tests@torchfont.dev")


def _commit_file(
    repo: pygit2.Repository,
    worktree: Path,
    relpath: str,
    *,
    content: str,
    message: str,
) -> str:
    file_path = worktree / relpath
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    repo.index.add(relpath)
    repo.index.write()
    tree = repo.index.write_tree()

    parents: list[pygit2.Oid] = []
    try:
        head = repo.head.peel(pygit2.Commit)
    except (KeyError, pygit2.GitError):
        pass
    else:
        parents.append(head.id)

    oid = repo.create_commit("HEAD", _AUTHOR, _AUTHOR, message, tree, parents)
    return str(oid)


def _init_origin_repo(tmp_path: Path) -> tuple[pygit2.Repository, str]:
    origin_worktree = tmp_path / "origin_worktree"
    origin_worktree.mkdir()
    work_repo = pygit2.init_repository(str(origin_worktree), initial_head="main")
    _commit_file(
        work_repo,
        origin_worktree,
        "fonts/Test-Regular.ttf",
        content="dummy font bytes",
        message="Initial font snapshot",
    )

    origin_bare = tmp_path / "origin.git"
    pygit2.init_repository(str(origin_bare), bare=True)
    work_repo.remotes.create("origin", str(origin_bare))
    work_repo.remotes["origin"].push(["refs/heads/main:refs/heads/main"])
    return work_repo, str(origin_bare)


def test_ensure_repo_requires_existing_repo_when_download_disabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"

    with pytest.raises(FileNotFoundError, match="download=True"):
        ensure_repo(
            root=root,
            url="https://example.com/fonts.git",
            ref="main",
            download=False,
        )

    assert not root.exists()
    assert not (root / ".git").exists()


def test_ensure_repo_persists_branch_ref_for_local_reuse(tmp_path: Path) -> None:
    _, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    synced_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )
    reused_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=False,
        depth=0,
    )

    assert synced_commit == reused_commit

    repo = pygit2.Repository(str(root))
    resolved, reference = repo.resolve_refish("main")
    assert str(resolved.id) == synced_commit
    assert reference is not None
    assert reference.name == "refs/heads/main"


def test_ensure_repo_updates_branch_when_download_enabled(tmp_path: Path) -> None:
    origin_repo, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    first_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )

    worktree = Path(origin_repo.workdir)
    latest_commit = _commit_file(
        origin_repo,
        worktree,
        "fonts/Test-Regular.ttf",
        content="updated dummy font bytes",
        message="Update font snapshot",
    )
    origin_repo.remotes["origin"].push(["refs/heads/main:refs/heads/main"])

    updated_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )
    reused_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=False,
        depth=0,
    )

    assert updated_commit == latest_commit
    assert reused_commit == latest_commit
    assert first_commit != updated_commit


def test_ensure_repo_resolves_local_revspec(tmp_path: Path) -> None:
    origin_repo, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    first_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )

    worktree = Path(origin_repo.workdir)
    _commit_file(
        origin_repo,
        worktree,
        "fonts/Test-Regular.ttf",
        content="new snapshot",
        message="Second snapshot",
    )
    origin_repo.remotes["origin"].push(["refs/heads/main:refs/heads/main"])

    ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )
    parent_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="main~1",
        download=False,
        depth=0,
    )

    assert parent_commit == first_commit


def test_ensure_repo_wraps_invalid_ref_spec_as_value_error(tmp_path: Path) -> None:
    _, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )

    with pytest.raises(ValueError, match="Unable to resolve ref"):
        ensure_repo(
            root=root,
            url=origin_url,
            ref="main..broken",
            download=False,
            depth=0,
        )


def test_ensure_repo_rejects_origin_url_mismatch(tmp_path: Path) -> None:
    _, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )

    with pytest.raises(ValueError, match="bound to remote"):
        ensure_repo(
            root=root,
            url="https://example.com/another.git",
            ref="main",
            download=False,
            depth=0,
        )


def test_ensure_repo_rejects_negative_depth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="depth must be >= 0"):
        ensure_repo(
            root=tmp_path / "cache",
            url="https://example.com/fonts.git",
            ref="main",
            download=False,
            depth=-1,
        )


@pytest.mark.parametrize(
    ("ref"),
    [
        "origin/main",
        "origin/main~1",
        "refs/remotes/origin/main",
    ],
)
def test_ensure_repo_rejects_remote_tracking_ref_when_download_enabled(
    tmp_path: Path,
    ref: str,
) -> None:
    _, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    with pytest.raises(ValueError, match="Remote-tracking ref"):
        ensure_repo(
            root=root,
            url=origin_url,
            ref=ref,
            download=True,
            depth=0,
        )


def test_ensure_repo_download_does_not_fetch_unrelated_branches(tmp_path: Path) -> None:
    origin_repo, origin_url = _init_origin_repo(tmp_path)
    head_commit = origin_repo.head.peel(pygit2.Commit)
    origin_repo.create_branch("extra-branch", head_commit)
    origin_repo.remotes["origin"].push(
        ["refs/heads/extra-branch:refs/heads/extra-branch"]
    )

    root = tmp_path / "cache"
    ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )

    repo = pygit2.Repository(str(root))
    assert repo.lookup_branch("main", pygit2.enums.BranchType.LOCAL) is not None
    assert repo.lookup_branch("extra-branch", pygit2.enums.BranchType.LOCAL) is None


def test_ensure_repo_rejects_revspec_when_download_enabled(
    tmp_path: Path,
) -> None:
    _, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    ensure_repo(
        root=root,
        url=origin_url,
        ref="main",
        download=True,
        depth=0,
    )

    with pytest.raises(ValueError, match="not supported with download=True"):
        ensure_repo(
            root=root,
            url=origin_url,
            ref="main~1",
            download=True,
            depth=0,
        )


def test_ensure_repo_supports_explicit_tag_ref(tmp_path: Path) -> None:
    origin_repo, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"

    head_commit = origin_repo.head.peel(pygit2.Commit)
    origin_repo.create_tag(
        "v1",
        head_commit.id,
        pygit2.enums.ObjectType.COMMIT,
        _AUTHOR,
        "v1",
    )
    origin_repo.remotes["origin"].push(["refs/tags/v1:refs/tags/v1"])

    synced_commit = ensure_repo(
        root=root,
        url=origin_url,
        ref="refs/tags/v1",
        download=True,
        depth=0,
    )
    assert synced_commit == str(head_commit.id)


def test_ensure_repo_rejects_existing_repo_without_origin(tmp_path: Path) -> None:
    _, origin_url = _init_origin_repo(tmp_path)
    root = tmp_path / "cache"
    pygit2.init_repository(str(root), initial_head="main")

    with pytest.raises(ValueError, match="does not define 'origin'"):
        ensure_repo(
            root=root,
            url=origin_url,
            ref="main",
            download=True,
            depth=0,
        )
