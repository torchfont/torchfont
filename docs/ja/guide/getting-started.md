# クイックスタート

<!-- markdownlint-disable MD013 -->

## uv のインストール

PyTorch は CPU・CUDA・ROCm などのハードウェア環境ごとに異なるインデックスで配布されています。通常はインストール時にインデックスを手動で指定する必要があり、再現性の確保が難しくなりがちです。uv を使えばインデックスを `pyproject.toml` に一度設定するだけで、誰でも同じ環境を再現できます。

uv の [インストールガイド](https://docs.astral.sh/uv/getting-started/installation/) に従ってインストールしてください。

## PyTorch インデックスの設定

`pyproject.toml` に PyTorch のインデックスとソースを追加します。環境に合わせて選択してください。

::: code-group

```toml [CPU-only]
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

```toml [CUDA 11.8]
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu118" }]
```

```toml [CUDA 12.6]
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
```

```toml [CUDA 12.8]
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
```

```toml [CUDA 13.0]
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
```

```toml [ROCm 7.2]
[[tool.uv.index]]
name = "pytorch-rocm72"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-rocm72" }]
```

```toml [Intel GPUs]
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-xpu" }]
```

:::

複数プラットフォームへの対応など、より詳細な設定は [uv の PyTorch インテグレーションガイド](https://docs.astral.sh/uv/guides/integration/pytorch/) を参照してください。

## 依存関係のインストール

以下のコマンドで PyTorch と TorchFont をインストールします。

```bash
uv add torch torchfont
```

## 動作確認

カレントディレクトリのフォントを走査して Dataset を作成し、そのサンプル数を表示します。フォントがなければ `0` が表示されます。エラーなく実行できればインストール完了です。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root=".")
print(len(dataset))
```
