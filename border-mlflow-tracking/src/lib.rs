#![doc = include_str!("../README.md")]
mod client;
mod experiment;
mod recorder;
mod run;
use anyhow::Result;
pub use client::{GetExperimentIdError, MlflowTrackingClient};
use experiment::Experiment;
pub use recorder::MlflowTrackingRecorder;
pub use run::Run;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Code adapted from <https://stackoverflow.com/questions/26593387>.
fn system_time_as_millis() -> u128 {
    let time = SystemTime::now();
    time.duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis()
}

/// Get the directory to which artifacts will be saved.
pub(crate) fn get_artifact_base(run: Run) -> Result<PathBuf> {
    let artifact_uri: PathBuf = run
        .clone()
        .info
        .artifact_uri
        .expect("Failed to get artifact_uri")
        .into();
    let artifact_uri = artifact_uri.strip_prefix("mlflow-artifacts:/")?;
    let path: PathBuf = std::env::var("MLFLOW_DEFAULT_ARTIFACT_ROOT")
        .expect("MLFLOW_DEFAULT_ARTIFACT_ROOT must be set")
        .into();
    Ok(path.join(artifact_uri))
}

// /// https://stackoverflow.com/questions/26958489/how-to-copy-a-folder-recursively-in-rust
// fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
//     fs::create_dir_all(&dst)?;
//     for entry in fs::read_dir(src)? {
//         let entry = entry?;
//         let ty = entry.file_type()?;
//         if ty.is_dir() {
//             copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
//         } else {
//             fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
//         }
//     }
//     Ok(())
// }
