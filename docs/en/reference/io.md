# IO API

## `torchfont.io.outline`

Shared constants for glyph command encoding.

```python
from torchfont.io.outline import CommandType, TYPE_DIM, COORD_DIM
```

### `CommandType: IntEnum`

```python
class CommandType(IntEnum):
    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6
```

### `TYPE_DIM: int`

Number of command types. Current value: `7`.

### `COORD_DIM: int`

Coordinate dimension. Current value: `6` (`[cx0, cy0, cx1, cy1, x, y]`).

---

## `torchfont.io.git`

Internal Git synchronization utility used by `FontRepo` and `GoogleFonts`.

```python
from torchfont.io.git import ensure_repo
```

### `ensure_repo(...) -> str`

```python
def ensure_repo(
    root: Path | str,
    url: str,
    ref: str,
    *,
    download: bool,
    depth: int = 1,
) -> str
```

Prepares/synchronizes `root` as a Git working tree and returns the checked-out
commit hash.

- `download=True`: fetch + force-checkout from remote
- `download=False`: local ref resolution + force-checkout
- `depth=1`: shallow fetch (default), `depth=0`: full history

Checkout uses a force strategy to align the working tree with `ref`.

Additional behavior:

- If `root/.git` does not exist and `download=False`, `FileNotFoundError` is raised.
- If `root/.git` exists but has no `origin` remote, `ValueError` is raised.
- If `root/.git` exists but `origin` URL differs from `url`, `ValueError` is raised.
- With `download=True`, remote-tracking refs (`origin/main`) and ref expressions
  (`main~1`, `HEAD^`, `a:b`) are rejected with `ValueError`.
- With `download=True`, branch shorthand is fetched as `refs/heads/<ref>`.
  Use explicit `refs/tags/...` when targeting tags.

Return value: `commit_hash` as `str`.

::: info
Most users should call `FontRepo` / `GoogleFonts` rather than using
`ensure_repo` directly.
:::
