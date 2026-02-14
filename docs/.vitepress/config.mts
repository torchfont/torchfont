import { defineConfig } from 'vitepress'

export default defineConfig({
  base: process.env.READTHEDOCS_CANONICAL_URL
    ? new URL(process.env.READTHEDOCS_CANONICAL_URL).pathname.replace(/\/$/, '')
    : '',

  title: 'TorchFont',
  head: [
    [
      'link',
      { rel: 'icon', type: 'image/svg+xml', href: '/brand/torchfont-logomark.svg' },
    ],
  ],

  themeConfig: {
    logo: '/brand/torchfont-logomark.svg',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/torchfont/torchfont' },
    ],
    outline: {
      level: [2, 3],
    },
    search: {
      provider: 'local',
    },
  },

  locales: {
    ja: {
      label: '日本語',
      lang: 'ja-JP',
      description: 'フォントのための機械学習ライブラリ',
      themeConfig: {
        nav: [
          { text: 'ガイド', link: '/ja/guide/what-is-torchfont' },
          { text: 'リファレンス', link: '/ja/reference/datasets' },
          { text: 'サンプル', link: '/ja/examples/' },
        ],
        sidebar: {
          '/ja/guide/': [
            {
              text: 'ガイド',
              items: [
                { text: 'TorchFont とは', link: '/ja/guide/what-is-torchfont' },
                {
                  text: 'クイックスタート',
                  link: '/ja/guide/getting-started',
                },
                { text: 'グリフデータ形式', link: '/ja/guide/glyph-data-format' },
                { text: 'DataLoader との統合', link: '/ja/guide/dataloader' },
                { text: 'Google Fonts', link: '/ja/guide/google-fonts' },
                { text: 'Git リポジトリ', link: '/ja/guide/git-repos' },
                { text: 'サブセット抽出', link: '/ja/guide/subset' },
              ],
            },
          ],
          '/ja/reference/': [
            {
              text: 'リファレンス',
              items: [
                { text: 'データセット', link: '/ja/reference/datasets' },
                { text: 'トランスフォーム', link: '/ja/reference/transforms' },
                { text: 'IO ユーティリティ', link: '/ja/reference/io' },
              ],
            },
          ],
          '/ja/examples/': [
            {
              text: 'サンプル',
              items: [{ text: 'サンプル集', link: '/ja/examples/' }],
            },
          ],
        },
      },
    },

    en: {
      label: 'English',
      lang: 'en-US',
      description: 'A Machine Learning Library for Fonts',
      themeConfig: {
        nav: [
          { text: 'Guide', link: '/en/guide/what-is-torchfont' },
          { text: 'Reference', link: '/en/reference/datasets' },
          { text: 'Examples', link: '/en/examples/' },
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Guide',
              items: [
                { text: 'What is TorchFont', link: '/en/guide/what-is-torchfont' },
                { text: 'Quickstart', link: '/en/guide/getting-started' },
                { text: 'Glyph Data Format', link: '/en/guide/glyph-data-format' },
                { text: 'DataLoader Integration', link: '/en/guide/dataloader' },
                { text: 'Google Fonts', link: '/en/guide/google-fonts' },
                { text: 'Git Repositories', link: '/en/guide/git-repos' },
                { text: 'Subset Extraction', link: '/en/guide/subset' },
              ],
            },
          ],
          '/en/reference/': [
            {
              text: 'Reference',
              items: [
                { text: 'Datasets', link: '/en/reference/datasets' },
                { text: 'Transforms', link: '/en/reference/transforms' },
                { text: 'IO Utilities', link: '/en/reference/io' },
              ],
            },
          ],
          '/en/examples/': [
            {
              text: 'Examples',
              items: [{ text: 'Example Gallery', link: '/en/examples/' }],
            },
          ],
        },
      },
    },
  },
})
