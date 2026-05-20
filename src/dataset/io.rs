use crate::error::Error;
use ignore::{WalkBuilder, overrides::OverrideBuilder};
use std::path::{Path, PathBuf};

pub(crate) fn canonicalize_root(root: &str) -> Result<PathBuf, Error> {
    let expanded = shellexpand::tilde(root);
    let path = PathBuf::from(expanded.as_ref());
    std::fs::canonicalize(&path).map_err(|err| {
        Error::Io(format!(
            "failed to resolve font root '{}': {err}",
            path.display()
        ))
    })
}

pub(crate) fn discover_font_files(
    root: &Path,
    patterns: Option<&[String]>,
) -> Result<Vec<String>, Error> {
    let mut builder = WalkBuilder::new(root);
    builder.standard_filters(false);
    builder.filter_entry(|entry| !is_vcs_metadata_dir(entry));
    if let Some(patterns) = patterns.filter(|p| !p.is_empty()) {
        builder.overrides(build_overrides(root, patterns)?);
    }

    let mut files = Vec::new();

    for result in builder.build() {
        let entry = result
            .map_err(|err| Error::Io(format!("failed to walk '{}': {err}", root.display())))?;

        let path = entry.path();

        if entry.file_type().is_some_and(|ft| ft.is_file()) && has_font_extension(path) {
            files.push(path.to_string_lossy().into_owned());
        }
    }

    files.sort_unstable();
    Ok(files)
}

fn is_vcs_metadata_dir(entry: &ignore::DirEntry) -> bool {
    entry.file_type().is_some_and(|ft| ft.is_dir())
        && entry
            .file_name()
            .to_str()
            .is_some_and(|name| matches!(name, ".git" | ".hg" | ".svn"))
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
            .map_err(|err| Error::Io(format!("invalid pattern '{pattern}': {err}")))?;
    }
    builder
        .build()
        .map_err(|err| Error::Io(format!("failed to compile patterns: {err}")))
}
