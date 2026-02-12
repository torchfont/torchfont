# IO API

## `torchfont.io.outline`

グリフコマンドの共通定数です。

```python
from torchfont.io.outline import TYPE_TO_IDX, TYPE_DIM, COORD_DIM
```

### `TYPE_TO_IDX: dict[str, int]`

```python
{
    "pad": 0,
    "moveTo": 1,
    "lineTo": 2,
    "curveTo": 3,
    "closePath": 4,
    "eos": 5,
}
```

### `TYPE_DIM: int`

コマンド種別数。現在値は `6`。

### `COORD_DIM: int`

座標次元数。現在値は `6`（`[cp1_x, cp1_y, cp2_x, cp2_y, x, y]`）。

---

## `torchfont.io.git`

`FontRepo` / `GoogleFonts` の内部で使われる Git 同期関数です。

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

`root` を Git 作業ディレクトリとして整備し、指定 `ref` のコミットハッシュを返します。

- `download=True`: リモート fetch + force checkout
- `download=False`: ローカルで `ref` を解決して force checkout
- `depth=1`: shallow fetch（既定）、`depth=0`: 履歴全体を取得

チェックアウトは、作業ツリーを `ref` に揃える force 戦略で実行されます。

追加挙動:

- `root/.git` が存在せず `download=False` の場合は `FileNotFoundError` になります。
- `root/.git` が存在しても `origin` remote が未定義の場合は `ValueError` になります。
- `root/.git` が存在し、`origin` URL と `url` 引数が一致しない場合は `ValueError` になります。
- `download=True` では remote-tracking ref（`origin/main`）と revspec（リビジョン指定, 例: `main~1`, `HEAD^`, `a:b`）は `ValueError` になります。
- `download=True` で省略ブランチ名を渡した場合は `refs/heads/<ref>` として fetch されます。タグを対象にする場合は `refs/tags/...` を明示してください。

返り値は `commit_hash`（文字列）です。

::: info
通常は `FontRepo` / `GoogleFonts` から使えば十分です。`ensure_repo` を直接呼ぶのは、Git 同期処理を独自に制御したい場合だけで構いません。
:::
