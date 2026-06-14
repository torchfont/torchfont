# Glyphsets API

<!-- markdownlint-disable MD013 -->

Google Fonts の "glyphsets" リポジトリに定義された codepoint グループの集
合です。これらのグループを使うと、フォント内のグリフを特定のスクリプトに
関連するものだけに絞り込んだり、フォントサイズを抑えるためにグリフ数を制
限したりできます。

```python
from torchfont.glyphsets import LATIN_CORE, LATIN_KERNEL, get_glyphset_codepoints
```

## `LATIN_CORE`

```python
from torchfont.glyphsets import LATIN_CORE
```

**GF Latin Core** グリフセットの codepoint 一覧（整数のリスト）。西欧言語
のほとんどに必要な基本ラテン文字をカバーします。

## `LATIN_KERNEL`

```python
from torchfont.glyphsets import LATIN_KERNEL
```

**GF Latin Kernel** グリフセットの codepoint 一覧（整数のリスト）。最小限
のフォント構成に適した、より小さなラテン文字のサブセットをカバーします。

## `get_glyphset_codepoints`

```python
from torchfont.glyphsets import get_glyphset_codepoints
```

```python
codepoints = get_glyphset_codepoints(glyphset_name)
```

Google Fonts glyphset レジストリから指定された名前のグリフセットの
codepoint を整数のリストとして返します。

| 引数             | 型     | 説明                         |
| ---------------- | ------ | ---------------------------- |
| `glyphset_name`  | `str`  | 検索するグリフセットの名前   |

| 返り値          | 型            | 説明                            |
| --------------- | ------------- | ------------------------------- |
| `codepoints`    | `list[int]`   | 該当グリフセットの codepoint    |

存在しないグリフセット名が指定された場合は `ValueError` を送出します。

### 例

```python
codepoints = get_glyphset_codepoints("GF_Latin_Core")
```
