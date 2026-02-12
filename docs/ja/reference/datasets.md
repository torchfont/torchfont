# データセット API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` は、すべて `torch.utils.data.Dataset` 互換のクラスを提供します。

## 使い分け

| クラス        | 入力元                  | 主な用途                               |
| ------------- | ----------------------- | -------------------------------------- |
| `FontFolder`  | ローカルディレクトリ    | 手元フォントからすぐ実験したい         |
| `FontRepo`    | 任意 Git リポジトリ     | 特定 OSS フォントを ref 固定で使いたい |
| `GoogleFonts` | Google Fonts リポジトリ | 大規模フォントを標準パターンで使いたい |

## FontFolder

```python
from torchfont.datasets import FontFolder
```

### コンストラクタ（`FontFolder`）

```python
FontFolder(
    root: Path | str,
    *,
    codepoint_filter: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
)
```

| 引数               | 型                                | 説明                               |
| ------------------ | --------------------------------- | ---------------------------------- |
| `root`             | `Path \| str`                     | フォント探索の起点ディレクトリ     |
| `codepoint_filter` | `Sequence[SupportsIndex] \| None` | 対象 Unicode codepoint を制限      |
| `patterns`         | `Sequence[str] \| None`           | gitignore 互換パターンでパスを絞る |
| `transform`        | `Callable \| None`                | `(types, coords)` へ適用する前処理 |

### 振る舞い

- 走査対象拡張子: `.ttf` / `.otf` / `.ttc` / `.otc`
- `__getitem__` は負インデックス対応（`dataset[-1]` など）
- 範囲外インデックスは `IndexError`

### 戻り値

```python
types, coords, style_idx, content_idx = dataset[idx]
```

| 要素          | 型                  | 形状           |
| ------------- | ------------------- | -------------- |
| `types`       | `torch.LongTensor`  | `(seq_len,)`   |
| `coords`      | `torch.FloatTensor` | `(seq_len, 6)` |
| `style_idx`   | `int`               | スカラー       |
| `content_idx` | `int`               | スカラー       |

### プロパティ

#### `targets -> torch.LongTensor`

全サンプルのラベル行列（`shape=(N, 2)`）

- `targets[:, 0]`: style index
- `targets[:, 1]`: content index

#### `content_classes -> list[str]`

コンテンツクラス名（1 文字 Unicode 文字列）の配列。

#### `content_class_to_idx -> dict[str, int]`

文字から content index へのマップ。

#### `style_classes -> list[str]`

スタイル名の配列。静的フォントは family/subfamily 名を使います。可変フォントは named instance があればそれを使い、ない場合は family/subfamily（または family のみ）へフォールバックします。named instance があっても名前が空の場合は family 名のみを使います。

#### `style_class_to_idx -> dict[str, int]`

スタイル名から style index へのマップ。重複名がある場合は `UserWarning` が出て、後から処理されたエントリで上書きされます。

### 例（`FontFolder`）

```python
dataset = FontFolder(
    root="~/fonts",
    codepoint_filter=range(0x41, 0x5B),  # A-Z
    patterns=("**/*.ttf", "!*Bold*"),
)
```

## FontRepo

```python
from torchfont.datasets import FontRepo
```

### コンストラクタ（`FontRepo`）

```python
FontRepo(
    root: Path | str,
    url: str,
    ref: str,
    *,
    patterns: Sequence[str],
    codepoint_filter: Sequence[SupportsIndex] | None = None,
    transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
    download: bool = False,
    depth: int = 1,
)
```

| 引数               | 型                                | 説明                             |
| ------------------ | --------------------------------- | -------------------------------- |
| `root`             | `Path \| str`                     | Git 作業ツリーを置くローカルパス |
| `url`              | `str`                             | リモート URL                     |
| `ref`              | `str`                             | Git 参照（`download=True` の制約は備考参照） |
| `patterns`         | `Sequence[str]`                   | フォント検出パターン             |
| `codepoint_filter` | `Sequence[SupportsIndex] \| None` | codepoint 制限                   |
| `transform`        | `Callable \| None`                | 前処理                           |
| `download`         | `bool`                            | `True` でリモート fetch を実行   |
| `depth`            | `int`                             | libgit2 の fetch 深さ（`1` shallow） |

### 追加プロパティ

| プロパティ    | 型    | 説明                             |
| ------------- | ----- | -------------------------------- |
| `url`         | `str` | 渡した URL 引数（保持値）        |
| `ref`         | `str` | 渡した ref                       |
| `commit_hash` | `str` | 最終的に checkout されたコミット |

### 備考

- Git 操作は `pygit2`（libgit2）で実行されます
- どちらのモードでも force checkout が実行されます
- `download=False` で `ref` がローカルで解決できない場合は例外になります
- `root/.git` が存在しない状態で `download=False` は `FileNotFoundError` になります
- `download=True` では remote-tracking ref（`origin/main`）と revspec（リビジョン指定, 例: `main~1`, `HEAD^`, `a:b`）は受け付けません
- `download=True` で省略ブランチ名を渡した場合は `refs/heads/<ref>` を fetch します。タグは `refs/tags/...` を明示してください
- 既存 `root/.git` の `origin` URL と `url` 引数が異なる場合は `ValueError` になります
- 既存 `root/.git` に `origin` remote がない場合も `ValueError` になります

### 例（`FontRepo`）

```python
dataset = FontRepo(
    root="data/fortawesome/font-awesome",
    url="https://github.com/FortAwesome/Font-Awesome",
    ref="7.x",
    patterns=("otfs/*.otf",),
    download=True,
)
```

## GoogleFonts

```python
from torchfont.datasets import GoogleFonts
```

### コンストラクタ（`GoogleFonts`）

```python
GoogleFonts(
    root: Path | str,
    ref: str,
    *,
    patterns: Sequence[str] | None = None,
    codepoint_filter: Sequence[int] | None = None,
    transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
    download: bool = False,
    depth: int = 1,
)
```

### 既定値

`patterns=None` のとき:

```python
(
    "apache/*/*.ttf",
    "ofl/*/*.ttf",
    "ufl/*/*.ttf",
    "!ofl/adobeblank/AdobeBlank-Regular.ttf",
)
```

### 備考（`GoogleFonts`）

- 取得元 URL は `https://github.com/google/fonts` に固定されています
- Git 同期仕様（`download` / `depth` / URL 整合性チェック）は `FontRepo` と同じです
- Google Fonts 用に専用 `root` ディレクトリを分けて運用してください

### 例（`GoogleFonts`）

```python
dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    codepoint_filter=range(0x30, 0x3A),
    download=True,
)
```
