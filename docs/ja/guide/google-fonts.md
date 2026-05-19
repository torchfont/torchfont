# Google Fonts のセットアップ

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

Google Fonts の最新版に追従するには次のコマンドを実行します。
`.gitmodules` に `shallow = true` が設定されているため `--depth 1` の明示は不要です。

```bash
git submodule update --remote
```
