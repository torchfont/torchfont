from torchfont.glyphsets import LATIN_CORE, LATIN_KERNEL, get_glyphset_codepoints


def test_latin_core() -> None:
    assert set(get_glyphset_codepoints("GF_Latin_Core")) == set(LATIN_CORE)
    assert ord("á") in LATIN_CORE


def test_latin_kernel() -> None:
    assert set(get_glyphset_codepoints("GF_Latin_Kernel")) == set(LATIN_KERNEL)
    assert ord("á") not in LATIN_KERNEL
