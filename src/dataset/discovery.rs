use std::path::{Path, PathBuf};

use ignore::{WalkBuilder, overrides::OverrideBuilder};

use crate::error::Error;

pub(crate) fn canonicalize_root(root: &str) -> Result<PathBuf, Error> {
    let expanded = shellexpand::tilde(root);
    let path = PathBuf::from(expanded.as_ref());
    std::fs::canonicalize(&path).map_err(|err| {
        Error::Io(std::io::Error::new(
            err.kind(),
            format!("failed to resolve font root '{}': {err}", path.display()),
        ))
    })
}

pub(crate) fn discover_font_files(
    root: &Path,
    patterns: Option<&[String]>,
) -> Result<Vec<PathBuf>, Error> {
    let mut builder = WalkBuilder::new(root);
    builder.standard_filters(false);
    if let Some(patterns) = patterns.filter(|p| !p.is_empty()) {
        builder.overrides(build_overrides(root, patterns)?);
    }

    let mut files = Vec::new();

    for result in builder.build() {
        let entry = result.map_err(|err| {
            let kind = err
                .io_error()
                .map_or(std::io::ErrorKind::Other, std::io::Error::kind);
            Error::Io(std::io::Error::new(
                kind,
                format!("failed to walk '{}': {err}", root.display()),
            ))
        })?;

        let path = entry.path();

        if entry.file_type().is_some_and(|ft| ft.is_file()) && has_font_extension(path) {
            files.push(path.to_path_buf());
        }
    }

    files.sort_unstable();
    Ok(files)
}

fn has_font_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "ttf" | "otf" | "ttc" | "otc"
            )
        })
}

fn build_overrides(root: &Path, patterns: &[String]) -> Result<ignore::overrides::Override, Error> {
    let mut builder = OverrideBuilder::new(root);
    for pattern in patterns {
        builder
            .add(pattern)
            .map_err(|err| Error::Parse(format!("invalid pattern '{pattern}': {err}")))?;
    }
    builder
        .build()
        .map_err(|err| Error::Parse(format!("failed to compile patterns: {err}")))
}
