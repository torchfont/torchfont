import { defineConfig } from 'vitepress'

const rtdBase = process.env.READTHEDOCS_CANONICAL_URL
  ? new URL(process.env.READTHEDOCS_CANONICAL_URL).pathname.replace(/\/$/, '')
  : ''
const faviconPath = `${rtdBase}/brand/torchfont-icon-light.svg`

export default defineConfig({
  base: rtdBase,
  vite: {
    build: {
      target: 'esnext',
    },
  },

  title: 'TorchFont',
  head: [
    [
      'link',
      { rel: 'icon', type: 'image/svg+xml', href: faviconPath },
    ],
  ],

  themeConfig: {
    logo: {
      light: '/brand/torchfont-icon-light.svg',
      dark: '/brand/torchfont-icon-dark.svg',
    },
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
    en: {
      label: 'English',
      lang: 'en-US',
      description: 'A Machine Learning Library for Vector Fonts',
      themeConfig: {
        nav: [
          { text: 'Guide', link: '/en/guide/basic/what-is-torchfont' },
          { text: 'Reference', link: '/en/reference/datasets' },
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Guide',
              items: [
                { text: 'What is TorchFont', link: '/en/guide/basic/what-is-torchfont' },
                { text: 'Quickstart', link: '/en/guide/basic/getting-started' },
                { text: 'Google Fonts Setup', link: '/en/guide/basic/google-fonts' },
                { text: 'Building a GlyphDataset', link: '/en/guide/basic/dataset' },
                { text: 'Glyph Data Format', link: '/en/guide/basic/glyph-data-format' },
                { text: 'Batching with DataLoader', link: '/en/guide/basic/dataloader' },
              ],
            },
            {
              text: 'Advanced Guides',
              items: [
                {
                  text: 'Dataset Integration',
                  link: '/en/guide/advanced/dataset-integration',
                },
                {
                  text: 'Font Collections',
                  link: '/en/guide/advanced/font-collections',
                },
                {
                  text: 'Variable Fonts',
                  link: '/en/guide/advanced/variable-fonts',
                },
                {
                  text: 'glyf and CFF Outlines',
                  link: '/en/guide/advanced/glyf-and-cff',
                },
                {
                  text: 'TorchVision Integration',
                  link: '/en/guide/advanced/torchvision',
                },
                {
                  text: 'Multiprocess Data Loading',
                  link: '/en/guide/advanced/multiprocessing',
                },
                {
                  text: 'torch.compile',
                  link: '/en/guide/advanced/torch-compile',
                },
              ],
            },
          ],
          '/en/reference/': [
            {
              text: 'Reference',
              items: [
                { text: 'Datasets', link: '/en/reference/datasets' },
                { text: 'Glyphsets', link: '/en/reference/glyphsets' },
                { text: 'Transforms', link: '/en/reference/transforms' },
                { text: 'Neural Networks', link: '/en/reference/nn' },
                { text: 'Core Types', link: '/en/reference/core-types' },
              ],
            },
          ],
        },
      },
    },

    ja: {
      label: '日本語',
      lang: 'ja-JP',
      description: 'ベクターフォントのための機械学習ライブラリ',
      themeConfig: {
        nav: [
          { text: 'ガイド', link: '/ja/guide/basic/what-is-torchfont' },
          { text: 'リファレンス', link: '/ja/reference/datasets' },
        ],
        sidebar: {
          '/ja/guide/': [
            {
              text: 'ガイド',
              items: [
                { text: 'TorchFont とは', link: '/ja/guide/basic/what-is-torchfont' },
                { text: 'クイックスタート', link: '/ja/guide/basic/getting-started' },
                { text: 'Google Fonts のセットアップ', link: '/ja/guide/basic/google-fonts' },
                { text: 'GlyphDataset の構築', link: '/ja/guide/basic/dataset' },
                { text: 'グリフデータ形式', link: '/ja/guide/basic/glyph-data-format' },
                { text: 'DataLoader によるバッチ処理', link: '/ja/guide/basic/dataloader' },
              ],
            },
            {
              text: '応用ガイド',
              items: [
                {
                  text: 'データセットとの連携',
                  link: '/ja/guide/advanced/dataset-integration',
                },
                {
                  text: 'フォントコレクション',
                  link: '/ja/guide/advanced/font-collections',
                },
                {
                  text: 'バリアブルフォント',
                  link: '/ja/guide/advanced/variable-fonts',
                },
                {
                  text: 'glyf と CFF のアウトライン',
                  link: '/ja/guide/advanced/glyf-and-cff',
                },
                {
                  text: 'TorchVision との連携',
                  link: '/ja/guide/advanced/torchvision',
                },
                {
                  text: 'マルチプロセス読み込み',
                  link: '/ja/guide/advanced/multiprocessing',
                },
                {
                  text: 'torch.compile',
                  link: '/ja/guide/advanced/torch-compile',
                },
              ],
            },
          ],
          '/ja/reference/': [
            {
              text: 'リファレンス',
              items: [
                { text: 'データセット', link: '/ja/reference/datasets' },
                { text: 'グリフセット', link: '/ja/reference/glyphsets' },
                { text: 'トランスフォーム', link: '/ja/reference/transforms' },
                { text: 'ニューラルネットワーク', link: '/ja/reference/nn' },
                { text: '基本型', link: '/ja/reference/core-types' },
              ],
            },
          ],
        },
      },
    },
  },
})
