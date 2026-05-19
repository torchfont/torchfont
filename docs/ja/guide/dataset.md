# データセットのセットアップ

## Google Fonts とは

[Google Fonts](https://fonts.google.com/) は Google が公開しているフォントコレクションで、機械学習のデータセットとして非常に適しています。

- **大規模なコレクション**: 1,000 を超えるファミリー、数千のフォントファイルを含み、多様なスタイルや文字体系をカバーしています。
- **機械学習に適したライセンス**: 収録フォントは Apache、OFL、UFL といったオープンライセンスのみで構成されており、機械学習への利用に適しています。
- **GitHub で公開**: リポジトリが GitHub 上で公開されているため、`git` による取得・管理が容易で、CI やスクリプトへの組み込みも簡単です。
- **再現性の高さ**: Git リポジトリとして管理されているため、特定の commit を固定することでデータセットを完全に再現できます。

## サブモジュールとして追加する

Google Fonts リポジトリをサブモジュールとして追加します。
`--depth 1` で最新の commit のみを取得し、ダウンロードサイズを抑えます。

```bash
git submodule add --depth 1 https://github.com/google/fonts.git data/google/fonts
```

## データセットを読み込む

サブモジュールを取得したら、`GlyphDataset` を使って Google Fonts を読み込みます。
次のコードを実行してください。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="data/google/fonts")

print(f"{len(dataset)=}")
print(f"{len(dataset.style_classes)=}")
print(f"{len(dataset.content_classes)=}")
```

実行すると次のような出力が得られます。
`style_classes` はフォントスタイルの種類数、`content_classes` はデータセットに含まれるユニークなコードポイントの数です。

```
len(dataset)=13745683
len(dataset.style_classes)=9113
len(dataset.content_classes)=1112000
```

## `patterns` で絞り込む

`patterns` を指定することで、読み込むフォントファイルを絞り込めます。
Google Fonts では次の patterns を推奨します。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
)

print(f"{len(dataset)=}")
print(f"{len(dataset.style_classes)=}")
print(f"{len(dataset.content_classes)=}")
```

各パターンの意図は次のとおりです。

- **`apache/*/*.ttf`、`ofl/*/*.ttf`、`ufl/*/*.ttf`**: Google Fonts のライセンスディレクトリは
  この3つのみです。リポジトリにはこれら以外にもディレクトリがあり、テスト用フォントなどが
  含まれることがあるため、明示的に3ディレクトリだけを対象にします。
- **`/*/*.ttf`（`/**/*.ttf` ではない）**: Google Fonts の配布形式は TTF のみです。
  また、`/**/*.ttf` にするとサブディレクトリに置かれた余分なフォントが
  含まれてしまうため、1階層のみを対象にします。
- **`!ofl/adobeblank/AdobeBlank-Regular.ttf`**: AdobeBlank はフォント表示のフォールバック
  検証用に設計されたフォントで、対応するコードポイント数が極めて多い一方、
  グリフに実質的な字形データが含まれていません。機械学習のデータセットに混入すると
  品質を著しく低下させるため、除外することが望ましいです。

このコードを実行すると次のような出力が得られます。patterns を指定しない場合と比べて、
スタイル数とコンテンツクラス数（コードポイント数）が絞られていることが確認できます。

```
len(dataset)=12460609
len(dataset.style_classes)=8951
len(dataset.content_classes)=114254
```

### コードポイントを絞る

`codepoints` を指定すると、インデックス時に対象外のコードポイントを除外できます。

アルファベット26文字（A–Z）に絞るには、次のコードを実行してください。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
)
```

実行結果は次のとおりです。

```
len(dataset)=220939
len(dataset.style_classes)=8498
len(dataset.content_classes)=26
```

Unicode の基本多言語面（BMP、U+0000–U+FFFF）に絞るには、次のコードを実行してください。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x0000, 0x10000),
)
```

実行結果は次のとおりです。

```
len(dataset)=12027967
len(dataset.style_classes)=8951
len(dataset.content_classes)=60004
```

## サブモジュールの更新

`git submodule add` が生成する `.gitmodules` に次の設定を加えます。

```ini
[submodule "data/google/fonts"]
    path = data/google/fonts
    url = https://github.com/google/fonts.git
    branch = main
    shallow = true
    ignore = dirty
```

- `branch = main`: `git submodule update --remote` で追跡するブランチを指定します。
- `shallow = true`: `git submodule update` 実行時に shallow clone で取得します。`git submodule add` 時点での shallow 化は `--depth 1` で行い、この設定はそれ以降の更新にも shallow を維持するためのものです。
- `ignore = dirty`: サブモジュール内のファイル変更を `git status` に表示しません。データファイルが意図せずダーティ扱いになることを防ぎます。

Google Fonts の最新版に追従するには `git submodule update --remote` を使います。
`shallow = true` が設定されているため `--depth 1` の明示は不要です。

```bash
git submodule update --remote
```
