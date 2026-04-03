# checkout 済み Git リポジトリを使う

TorchFont の中心ワークフローでは、Git リポジトリの同期自体は
TorchFont の外で行います。clone / update した checkout を
`GlyphDataset` に渡してください。

## 基本形

```bash
git clone --depth 1 https://github.com/FortAwesome/Font-Awesome \
  data/fortawesome/font-awesome
```

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/fortawesome/font-awesome",
    patterns=("otfs/*.otf",),
)
```

## この使い方が向く場面

- 既存 OSS フォント資産をそのまま使いたい
- checkout の commit を自分で管理して再現性を持たせたい
- すべてのコーパスを同じ local-directory API から読みたい

## パターン構文

`patterns` は gitignore 互換のマッチングを使います。

| パターン     | 意味                                                         |
| ------------ | ------------------------------------------------------------ |
| `*.ttf`      | basename ベースで `.ttf` に一致（サブディレクトリ含む）      |
| `**/*.ttf`   | 任意深さを明示した再帰 `.ttf` マッチ                         |
| `otfs/*.otf` | `otfs/` 配下の `.otf` に一致                                 |
| `!*Bold*`    | `Bold` を含むパスを除外                                      |

::: info
`patterns` は候補パスを先に絞り込みます。その後、TorchFont は
`.ttf` / `.otf` / `.ttc` / `.otc` 拡張子のファイルだけを残します。
:::

## 実例

### Font Awesome

```python
GlyphDataset(
    root="data/fortawesome/font-awesome",
    patterns=("otfs/*.otf",),
)
```

### Material Design Icons

```python
GlyphDataset(
    root="data/google/material_design_icons",
    patterns=("variablefont/*.ttf",),
)
```

### Source Han Sans（TTC 含む）

```python
GlyphDataset(
    root="data/adobe-fonts/source-han-sans",
    patterns=("*.ttf.ttc",),
)
```

::: info
`*.ttf.ttc` は意図通りです。このリポジトリには `Something.ttf.ttc`
のような名前の TTC ファイルがあります。
:::

## 更新フロー

Git 側で checkout を更新し、そのあと Dataset インスタンスを作り直します。

```bash
git -C data/fortawesome/font-awesome fetch --depth 1 origin
git -C data/fortawesome/font-awesome checkout 7.x
```

TorchFont は Dataset オブジェクトの寿命中、ネイティブな indexing state を
保持します。ディスク上のファイルが変わったら、その state が checkout と
ずれないように Dataset も作り直してください。Dataset の利用中にファイルが
変わった場合の結果は未定義で、不正な sample や runtime error につながる
ことがあります。

## 再現性のためのメモ

必要なら commit hash を Git 側で保存してください。

```bash
git -C data/fortawesome/font-awesome rev-parse HEAD
```

## DataLoader との統合

[DataLoader との統合](/ja/guide/dataloader)で紹介している
`collate_fn` をそのまま使えます。
