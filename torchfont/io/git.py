"""Internal helpers for synchronizing Git repositories with progress reporting."""

import re
from pathlib import Path

import pygit2
from pygit2.remotes import TransferProgress
from rich.progress import Progress, TaskID


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


def ensure_repo(
    root: Path | str,
    url: str,
    ref: str,
    *,
    download: bool,
) -> str:
    """Ensure ``root`` hosts ``ref`` and return the synced commit hash."""
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    git_dir = path / ".git"

    if git_dir.exists():
        repo = pygit2.Repository(str(path))
    else:
        repo = pygit2.init_repository(str(path), origin_url=url)

    with Progress() as progress:
        # Shallow fetch cannot efficiently re-download a commit hash that
        # the server does not advertise as a ref.  When the requested ref
        # already resolves locally to a direct OID (i.e. not a named
        # branch/tag), we skip the network round-trip entirely.
        try:
            _, reference = repo.resolve_refish(ref)
            need_fetch = download and reference is not None
        except KeyError:
            need_fetch = download

        if need_fetch:
            callbacks = _RemoteCallbacks(progress)
            repo.remotes["origin"].fetch([ref], depth=1, callbacks=callbacks)
            fetch_head = repo.lookup_reference("FETCH_HEAD")
            repo.checkout(
                fetch_head,
                strategy=pygit2.GIT_CHECKOUT_FORCE,
                callbacks=_CheckoutCallbacks(progress),
            )
        else:
            target = repo.revparse_single(ref)
            repo.checkout_tree(
                target,
                strategy=pygit2.GIT_CHECKOUT_FORCE,
                callbacks=_CheckoutCallbacks(progress),
            )
            repo.set_head(target.id)

    commit = repo.head.peel()
    return str(commit.id)
