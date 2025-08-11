//! Converts SAC policy for an agent without any backend (tch or candle).
//!
//! You need to prepare the model parameter files by `sac_pendulum_tch.rs` in advance.
//!

#[cfg(all(feature = "tch", not(feature = "candle")))]
mod tch;
#[cfg(all(feature = "tch", not(feature = "candle")))]
use tch::{create_mlp, load_sac_model};

#[cfg(all(feature = "candle", not(feature = "tch")))]
mod candle;
#[cfg(all(feature = "candle", not(feature = "tch")))]
use candle::{create_mlp, load_sac_model};

use anyhow::Result;
use border_policy_no_backend::Mlp as MlpNoBackend;
use std::{fs, io::Write};

fn serialize_to_file(mlp: &MlpNoBackend, dest_path: &str) -> Result<()> {
    let encoded = bincode::serialize(mlp)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(dest_path)?;
    file.write_all(&encoded)?;
    Ok(())
}

fn main() -> Result<()> {
    #[cfg(all(feature = "tch", not(feature = "candle")))]
    let (src_path, dest_path) = {
        ("../sac_pendulum_tch/model/best", "./model/from_tch/mlp.bincode")
    };

    #[cfg(all(feature = "candle", not(feature = "tch")))]
    let (src_path, dest_path) = {
        ("../sac_pendulum/model/best", "./model/from_candle/mlp.bincode")
    };

    let sac = load_sac_model(src_path)?;
    let mlp = create_mlp(&sac);
    serialize_to_file(&mlp, dest_path)?;

    Ok(())
}
