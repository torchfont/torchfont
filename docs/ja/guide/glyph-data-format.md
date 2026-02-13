# グリフデータ形式

<!-- markdownlint-disable MD013 -->

TorchFont の Dataset は、各グリフを次の 4 要素で返します。

```python
types, coords, style_idx, content_idx = dataset[i]
```

| 要素 | 型 | 形状 | 意味 |
| --- | --- | --- | --- |
| `types` | `torch.LongTensor` | `(seq_len,)` | ペンコマンド列 |
| `coords` | `torch.FloatTensor` | `(seq_len, 6)` | 各コマンドの座標 |
| `style_idx` | `int` | スカラー | 書体スタイルのクラスID |
| `content_idx` | `int` | スカラー | 文字内容のクラスID |

## `types` の定義

```python
from torchfont.io.outline import CommandType

print(CommandType.QUAD_TO, CommandType.QUAD_TO.value)
# CommandType.QUAD_TO 3
```

- `CommandType.END` はシーケンス終端
- `CommandType.PAD` は主に `pad_sequence` や `Patchify` によるパディングで出現

## `coords` の定義

各ステップは 6 次元です。

```text
[cx0, cy0, cx1, cy1, x, y]
```

- `CommandType.MOVE_TO` / `CommandType.LINE_TO`: 制御点は 0、終点 `(x, y)` を使用
- `CommandType.QUAD_TO`: 2 次ベジェの制御点 `(cx0, cy0)` と終点 `(x, y)` を使用（`cx1`,
  `cy1` は 0）
- `CommandType.CURVE_TO`: 3 次ベジェの制御点 `(cx0, cy0)` と `(cx1, cy1)` + 終点を使用
- `CommandType.CLOSE` / `CommandType.END` / `CommandType.PAD`: 座標は 0

::: info
座標値はフォントの `units_per_em` で正規化されています。
:::

## 2 次ベジェの扱い

2 次ベジェは `CommandType.QUAD_TO` としてそのまま出力されます。テンソル形状を固定するため、`CommandType.QUAD_TO` の座標は `[cx0, cy0, 0, 0, x, y]` です。

## スタイルとコンテンツのラベル

### `style_idx`

- 静的フォント: 通常は `Family + Subfamily` 形式（例: `Lato Regular`）
- 可変フォント: 名前付きインスタンスがあれば、その単位で 1 クラス
- 可変フォントでインスタンス名が空の場合: `Family` のみ
- 可変フォントで名前付きインスタンスがない場合: `Family + Subfamily`（または
  `Family`）へフォールバック

```python
print(dataset.style_classes[:5])
print(dataset.style_class_to_idx)
```

::: warning
`style_classes` には重複名が含まれることがあります。この場合、`style_class_to_idx` は同名のうち最後の要素だけを保持します。重複を区別した抽出が必要なら、`style_classes` を列挙して扱ってください。
:::

### `content_idx`

- Unicode 文字ごとのクラスID

```python
print(dataset.content_classes[:5])
print(dataset.content_class_to_idx)
```

## `targets` で一括取得

```python
t = dataset.targets  # shape: (N, 2), dtype: torch.long

style_all = t[:, 0]    # 1 列目: style_idx
content_all = t[:, 1]  # 2 列目: content_idx
```

## Transform 後の形状

`Patchify` を使うと `types` / `coords` の次元が増えます。

- 変換前: `types=(seq_len,)`, `coords=(seq_len, 6)`
- 変換後: `types=(num_patches, patch_size)`,
  `coords=(num_patches, patch_size, 6)`

この場合も `style_idx` / `content_idx` は同じです。
