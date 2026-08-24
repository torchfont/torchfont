use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::error::Error;

/// Builds entries from font files on every available core, in file order.
///
/// Parsing one font file is independent of every other, so files are claimed
/// from a shared cursor and their results are reordered afterwards. Font sizes
/// differ by orders of magnitude, so claiming one file at a time balances the
/// work far better than splitting the list into equal shares. The result is
/// what running `build` over `files` sequentially would produce, including
/// which file's error is reported.
pub(crate) fn build_from_files<T: Send>(
    files: &[PathBuf],
    build: impl Fn(&Path) -> Result<Vec<T>, Error> + Sync,
) -> Result<Vec<T>, Error> {
    let threads = thread::available_parallelism().map_or(1, |count| count.get());
    let cursor = AtomicUsize::new(0);
    let mut built: Vec<(usize, Result<Vec<T>, Error>)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.min(files.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let index = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(path) = files.get(index) else { break };
                        local.push((index, build(path)));
                    }
                    local
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("font file workers do not panic"))
            .collect()
    });
    built.sort_unstable_by_key(|(index, _)| *index);
    let mut entries = Vec::new();
    for (_, result) in built {
        entries.extend(result?);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::error::Error;

    use super::build_from_files;

    fn numbered_files(count: usize) -> Vec<PathBuf> {
        (0..count)
            .map(|index| PathBuf::from(index.to_string()))
            .collect()
    }

    fn number(path: &Path) -> usize {
        path.to_str().unwrap().parse().unwrap()
    }

    #[test]
    fn builds_entries_in_file_order() {
        let files = numbered_files(1000);
        let entries = build_from_files(&files, |path| Ok(vec![number(path)])).unwrap();
        assert_eq!(entries, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn reports_the_error_of_the_first_failing_file() {
        let files = numbered_files(1000);
        let error = build_from_files(&files, |path| match number(path) {
            index @ (100 | 500) => Err(Error::Parse(index.to_string())),
            index => Ok(vec![index]),
        })
        .unwrap_err();
        assert!(matches!(error, Error::Parse(message) if message == "100"));
    }

    #[test]
    fn builds_an_empty_file_list() {
        let entries = build_from_files(&[], |path| Ok(vec![number(path)])).unwrap();
        assert!(entries.is_empty());
    }
}
