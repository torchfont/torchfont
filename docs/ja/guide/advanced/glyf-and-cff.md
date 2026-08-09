# glyf と CFF のアウトライン

OpenType フォントのアウトラインは、主に次の 2 形式で格納されます。

| アウトラインテーブル | 曲線の表現 | バリアブルフォントでの形式 |
| --- | --- | --- |
| TrueType `glyf` | 2 次ベジェ曲線 | `glyf` と `gvar` |
| PostScript CFF | 3 次ベジェ曲線 | CFF2 |

`.ttf` と `.otf` という拡張子は一般的な規約ですが、TorchFont の API の境界では
ありません。読み込まれた曲線が `ElementType.QUAD_TO` と
`ElementType.CURVE_TO` のどちらになるかはアウトラインテーブルによって決まります。

## 同じ Transform インターフェースで扱う

`LoadGlyph` はどちらの形式も共通の `Outline` 型へ変換します。そのため、線、サブパス、
座標、メタデータ、バッチ処理、幾何 Transform、レンダリングには同じ API を使用できます。

```python
from torchfont import transforms as T

transform = T.Compose(
    [
        T.LoadGlyph(location="random"),
        T.RemoveOverlaps(),
        T.RandomAffine(degrees=5.0),
        T.RenderBitmap(size=64),
    ]
)
```

このパイプラインは `glyf`、CFF、CFF2 を使用する静的フォントとバリアブルフォントの
どちらも受け取れます。選択した Transform が 2 次と 3 次のセグメントの両方に対応して
いれば、形式による分岐は必要ありません。

## 曲線表現を統一する

モデルがすべての `Outline` に同じ曲線次数を必要とする場合は、アウトラインパイプラインの
先頭で入力形式を変換します。

```python
# すべてのフォントを 3 次曲線に統一します。
cubic_transform = T.Compose([T.LoadGlyph(), T.QuadToCubic(merge_curves=True)])

# または、すべてのフォントを 2 次曲線に統一します。
quadratic_transform = T.Compose([T.LoadGlyph(), T.CubicToQuad()])
```

`QuadToCubic` は各 2 次セグメントを正確に変換します。`CubicToQuad` は、約 `1e-3` em
以内で近似するために、1 つの 3 次セグメントを複数の 2 次セグメントへ変換することが
あります。モデルが両方の要素型を扱える場合は元の表現を保ち、統一した表現が必要な場合に
のみ変換してください。
