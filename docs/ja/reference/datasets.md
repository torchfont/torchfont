# データセット API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` では、`GlyphDataset` が中心となる公開 Dataset API です。

## GlyphSample

```python
from torchfont.datasets import GlyphSample
```

`GlyphDataset.__getitem__` の返り値であり、`torchfont.transforms` 全体で使う
sample 型です。

## GlyphLocation

```python
from torchfont.datasets import GlyphLocation
```

`GlyphDataset.locate(idx)` が返す、サンプルの出自 metadata 型です。

## DatasetMetadata

```python
from torchfont.datasets import DatasetMetadata
```

`GlyphDataset.metadata` が返す構造化 label metadata 型です。

## GlyphDataset

```python
from torchfont.datasets import GlyphDataset
```

### コンストラクタ（`GlyphDataset`）

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[GlyphSample], GlyphSample] | None = None,
)
```

| 引数               | 型                                | 説明                               |
| ------------------ | --------------------------------- | ---------------------------------- |
| `root`             | `Path \| str`                     | フォント探索の起点ディレクトリ     |
| `codepoints` | `Sequence[SupportsIndex] \| None` | 対象 Unicode codepoint を制限      |
| `patterns`         | `Sequence[str] \| None`           | gitignore 互換パターンでパスを絞る |
| `transform`        | `Callable \| None`                | sample-first 前処理（`GlyphSample -> GlyphSample`） |

### 振る舞い

- 走査対象拡張子: `.ttf` / `.otf` / `.ttc` / `.otc`
- `root` は初期化時に絶対 `Path` へ解決される
- `codepoints` は index 化前に sort 済み・重複なしの整数列へ正規化される
- 不正な `codepoints` は `ValueError` になり、受け入れる値は Unicode scalar
  value（`0 <= cp <= 0x10FFFF` かつ surrogate を除く）に限る
- `__getitem__` は負インデックス対応（`dataset[-1]` など）
- 範囲外インデックスは `IndexError`

### 保持される設定値

- `dataset.root`: 解決済みの root `Path`
- `dataset.patterns`: パスフィルタの tuple、または `None`
- `dataset.codepoints`: sort 済み・重複なし codepoint の tuple、または `None`

### 戻り値

```python
sample = dataset[idx]
```

| 要素                 | 型                  | 形状           |
| -------------------- | ------------------- | -------------- |
| `sample.types`       | `torch.LongTensor`  | `(seq_len,)`   |
| `sample.coords`      | `torch.FloatTensor` | `(seq_len, 6)` |
| `sample.style_idx`   | `int`               | スカラー       |
| `sample.content_idx` | `int`               | スカラー       |

`sample` 自体の型は `GlyphSample` です。

### メソッド

#### `locate(idx) -> GlyphLocation`

dataset index を、そのサンプル元になった font source の位置情報へ戻します。

- `font_path`: フォントファイルの解決済みパス
- `face_idx`: ファイル内の 0 始まり face index
- `instance_idx`: 可変フォントの named instance index、静的フォントなら `None`
- `codepoint`: その glyph sample の Unicode codepoint
- `style_idx`: style class index
- `content_idx`: content class index

### プロパティ

#### `targets -> torch.LongTensor`

全サンプルのラベル行列（`shape=(N, 2)`）

- `targets[:, 0]`: style index
- `targets[:, 1]`: content index

#### `content_classes -> list[str]`

コンテンツクラス名（1 文字 Unicode 文字列）の配列。

#### `metadata -> DatasetMetadata`

ラベル metadata をまとめた構造化オブジェクト。

- `metadata.styles`: `StyleLabel` の tuple
- `metadata.contents`: `ContentLabel` の tuple
- `metadata.style_id_to_idx`: style `label_id` から style index へのマップ
- `metadata.style_name_to_idxs`: style 表示名から全 index へのマップ
- `metadata.content_id_to_idx`: content `label_id` から content index へのマップ

#### `content_class_to_idx -> dict[str, int]`

文字から content index へのマップ。

#### `content_labels -> list[ContentLabel]`

content ラベル metadata の配列。各要素は次を持ちます。

- `idx`: content index
- `label_id`: 衝突しない識別子（`content:U+XXXX`）
- `char`: 1 文字の Unicode 文字列
- `codepoint`: Unicode codepoint（`int`）

#### `content_label_to_idx -> dict[str, int]`

content `label_id` から content index へのマップ。

#### `style_classes -> list[str]`

スタイル名の配列。静的フォントは family/subfamily 名を使います。可変フォントは named instance があればそれを使い、ない場合は family/subfamily（または family のみ）へフォールバックします。named instance があっても名前が空の場合は family 名のみを使います。

#### `style_labels -> list[StyleLabel]`

style ラベル metadata の配列。各要素は次を持ちます。

- `idx`: style index
- `label_id`: 衝突しない識別子（`style:<idx>`）
- `name`: 表示名（重複可）

#### `style_label_to_idx -> dict[str, int]`

style `label_id` から style index へのマップ。

#### `style_name_to_idxs -> dict[str, list[int]]`

style の表示名から、該当する全 style index へのマップ。

上記の metadata 関連プロパティは、内部的には `dataset.metadata` の射影です。

### 例（`GlyphDataset`）

```python
dataset = GlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),  # A-Z
    patterns=("**/*.ttf", "!*Bold*"),
)
```

## `root` の考え方

- 普通のローカルフォントディレクトリ
- Git などで自分で clone した repository checkout
- TorchFont の外側で同期を管理する外部コーパス

TorchFont から見れば、どれも通常のローカルディレクトリです。ディスク上の
ファイルが更新されたら、ネイティブキャッシュを作り直すために Dataset
インスタンスも作り直してください。
