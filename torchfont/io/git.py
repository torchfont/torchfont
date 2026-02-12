"""Internal helpers for synchronizing Git repositories with progress reporting."""

import re
from pathlib import Path

import pygit2
from pygit2.remotes import TransferProgress
from rich.progress import Progress, TaskID

_ORIGIN = "origin"
_CHECKOUT_STRATEGY = pygit2.GIT_CHECKOUT_FORCE
_DEFAULT_DEPTH = 1
_LOCAL_BRANCH_REF_PREFIX = "refs/heads/"


class _RemoteCallbacks(pygit2.RemoteCallbacks):
    FETCH_TASK = "Receiving objects"
    PROGRESS_PATTERN = re.compile(r"(\w+(?:\s+\w+)*?):\s+(\d+)%\s+\((\d+)/(\d+)\)")

    def __init__(self, progress: Progress) -> None:
        super().__init__()
        self._progress = progress
        self._task_id: TaskID | None = None
        self._sideband_tasks: dict[str, TaskID] = {}

    def sideband_progress(self, string: str) -> None:
        if not string:
            return

        for match in self.PROGRESS_PATTERN.finditer(string):
            operation = match.group(1)
            current = int(match.group(3))
            total = int(match.group(4))

            task_id = self._sideband_tasks.get(operation)
            if task_id is None:
                task_id = self._progress.add_task(operation, total=total)
                self._sideband_tasks[operation] = task_id

            self._progress.update(task_id, completed=min(current, total))

    def transfer_progress(self, stats: TransferProgress) -> None:
        total = stats.total_objects
        received = stats.received_objects
        if total == 0:
            return

        if self._task_id is None:
            self._task_id = self._progress.add_task(self.FETCH_TASK, total=total)

        self._progress.update(self._task_id, completed=received)


class _CheckoutCallbacks(pygit2.CheckoutCallbacks):
    OPERATION = "Checking out files"

    def __init__(self, progress: Progress) -> None:
        super().__init__()
        self._progress = progress
        self._task_id: TaskID | None = None
        self._completed = False

    def checkout_progress(
        self,
        path: str | None,
        completed_steps: int,
        total_steps: int,
    ) -> None:
        _ = path

        if self._completed or total_steps == 0:
            return

        if self._task_id is None:
            self._task_id = self._progress.add_task(self.OPERATION, total=total_steps)

        self._progress.update(self._task_id, completed=completed_steps)

        if completed_steps >= total_steps:
            self._completed = True


def _open_repo_and_origin(
    path: Path,
    url: str,
    *,
    download: bool,
) -> tuple[pygit2.Repository, pygit2.Remote]:
    if (path / ".git").exists():
        repo = pygit2.Repository(str(path))
        try:
            remote = repo.remotes[_ORIGIN]
        except KeyError as exc:
            msg = (
                f"Existing repository at '{path}' does not define "
                f"'{_ORIGIN}' remote."
            )
            raise ValueError(msg) from exc
    else:
        if not download:
            msg = (
                f"Git repository not found at '{path}'. "
                "Run once with download=True to initialize the cache."
            )
            raise FileNotFoundError(msg)

        repo = pygit2.init_repository(str(path), origin_url=url)
        remote = repo.remotes[_ORIGIN]

    if remote.url != url:
        msg = (
            f"Existing repository at '{repo.workdir}' is bound to remote "
            f"'{remote.url}', but '{url}' was requested. Use a different root "
            "directory per source repository."
        )
        raise ValueError(msg)

    return repo, remote


def _fetch_refspecs_for_ref(ref: str) -> list[str]:
    if ref.startswith((f"{_ORIGIN}/", f"refs/remotes/{_ORIGIN}/")):
        msg = (
            f"Remote-tracking ref '{ref}' is not supported. "
            "Use 'main' or 'refs/heads/main' style refs."
        )
        raise ValueError(msg)

    if any(marker in ref for marker in ("~", "^", ":")):
        msg = (
            f"Ref expression '{ref}' is not supported with download=True. "
            "Fetch a concrete ref first, then resolve expressions with "
            "download=False."
        )
        raise ValueError(msg)

    if ref.startswith("refs/"):
        return [f"+{ref}:{ref}"]

    branch_ref = f"{_LOCAL_BRANCH_REF_PREFIX}{ref}"
    return [f"+{branch_ref}:{branch_ref}"]


def _checkout_ref(repo: pygit2.Repository, ref: str, *, progress: Progress) -> None:
    target, reference = repo.resolve_refish(ref)
    callbacks = _CheckoutCallbacks(progress)

    if reference is None:
        repo.checkout_tree(
            target,
            strategy=_CHECKOUT_STRATEGY,
            callbacks=callbacks,
        )
        repo.set_head(target.id)
        return

    repo.checkout(
        reference,
        strategy=_CHECKOUT_STRATEGY,
        callbacks=callbacks,
    )


def ensure_repo(
    root: Path | str,
    url: str,
    ref: str,
    *,
    download: bool,
    depth: int = _DEFAULT_DEPTH,
) -> str:
    """Ensure ``root`` hosts ``ref`` and return the synced commit hash.

    ``download`` controls whether a fetch is attempted. ``depth`` is forwarded
    to ``Remote.fetch`` (`1` shallow by default, `0` for full history).
    """
    if depth < 0:
        msg = f"depth must be >= 0, got {depth}"
        raise ValueError(msg)

    path = Path(root).expanduser().resolve()
    repo, remote = _open_repo_and_origin(path, url, download=download)

    with Progress() as progress:
        if download:
            remote.fetch(
                _fetch_refspecs_for_ref(ref),
                depth=depth,
                callbacks=_RemoteCallbacks(progress),
            )

        try:
            _checkout_ref(repo, ref, progress=progress)
        except (KeyError, pygit2.InvalidSpecError) as exc:
            msg = f"Unable to resolve ref '{ref}' in '{path}'."
            raise ValueError(msg) from exc

    commit, _ = repo.resolve_refish("HEAD")
    return str(commit.id)
