# データセットとの連携

フォントリポジトリを Git サブモジュールとしてプロジェクトへ追加できます。Git が使用する
リビジョンを記録し、フォントファイルをプロジェクト自身の履歴から分離して管理します。

## リポジトリを追加する

`data/` 以下のパスを決め、リポジトリを追加します。

```bash
git submodule add --depth 1 https://github.com/google/fonts.git data/google/fonts
```

生成された `.gitmodules` と記録されたサブモジュールのリビジョンをコミットします。親
リポジトリをクローンした後は、次のコマンドでチェックアウトを初期化します。

```bash
git submodule update --init --depth 1
```

`CodepointDataset` の `root` にサブモジュールのパスを渡し、`patterns` でフォントファイルを
選択します。

```python
from torchfont.datasets import CodepointDataset

dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=("apache/*/*.ttf", "ofl/*/*.ttf", "ufl/*/*.ttf"),
)
```

## フォントリポジトリ

### Google Fonts

[Google Fonts リポジトリ](https://github.com/google/fonts) には多数のフォントファミリーが
収録されています。最上位の `apache`、`ofl`、`ufl` ディレクトリには、ライセンスごとに
分類されたバイナリフォントがあります。

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=("apache/*/*.ttf", "ofl/*/*.ttf", "ufl/*/*.ttf"),
)
```

### Material Design Icons

[Material Design Icons リポジトリ](https://github.com/google/material-design-icons) には、
複数のスタイルの従来版 Material Icons と、Material Symbols のバリアブルフォントが
収録されています。

```bash
git submodule add --depth 1 \
    https://github.com/google/material-design-icons.git \
    data/google/material-design-icons
```

```python
dataset = CodepointDataset(
    root="data/google/material-design-icons",
    patterns=("font/*.ttf", "font/*.otf", "variablefont/*.ttf"),
)
```

### Font Awesome

[Font Awesome リポジトリ](https://github.com/FortAwesome/Font-Awesome) の `otfs/` には、
無料版の Regular、Solid、Brand の各アイコンフォントが収録されています。

```bash
git submodule add --depth 1 \
    https://github.com/FortAwesome/Font-Awesome.git \
    data/fortawesome/font-awesome
```

```python
dataset = CodepointDataset(
    root="data/fortawesome/font-awesome",
    patterns=("otfs/*.otf",),
)
```

### Source Han Code JP

[Source Han Code JP](https://github.com/adobe-fonts/source-han-code-jp) は、日本語フォントを
個別の OpenType フォントと OpenType Collection で提供しています。`CodepointDataset` は
コレクション内の各フェイスを展開します。詳しくは
[フォントコレクション](./font-collections.md) を参照してください。

```bash
git submodule add --depth 1 \
    https://github.com/adobe-fonts/source-han-code-jp.git \
    data/adobe/source-han-code-jp
```

```python
dataset = CodepointDataset(
    root="data/adobe/source-han-code-jp",
    patterns=("OTC/*.ttc",),
)
```

フォントを使用または再配布する前に、各リポジトリのライセンスを確認してください。
