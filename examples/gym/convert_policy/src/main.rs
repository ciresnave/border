//! Converts SAC policy for an agent without any backend (tch or candle).
//!
//! You need to prepare the model parameter files by `sac_pendulum_tch.rs` in advance.
//!
use anyhow::Result;
use border_core::{Agent, Configurable, dummy::*};
use border_policy_no_backend::Mlp as MlpNoBackend;
use border_tch_agent::{
    model::ModelBase,
    mlp::{Mlp, Mlp2, MlpConfig},
    sac::{ActorConfig, CriticConfig, SacConfig, Sac},
};
use std::{fs, io::Write};

const DIM_OBS: i64 = 3;
const DIM_ACT: i64 = 1;

type Sac_ = Sac<DummyEnv, Mlp, Mlp2, DummyReplayBuffer>;

fn create_sac_config() -> SacConfig<Mlp, Mlp2> {
    // Omit learning related parameters
    let actor_config = ActorConfig::default()
        .out_dim(DIM_ACT)
        .pi_config(MlpConfig::new(DIM_OBS, vec![64, 64], DIM_ACT, false));
    let critic_config = CriticConfig::default().q_config(MlpConfig::new(
        DIM_OBS + DIM_ACT,
        vec![64, 64],
        1,
        false,
    ));
    SacConfig::default()
        .actor_config(actor_config)
        .critic_config(critic_config)
        .device(tch::Device::Cpu)
}

fn load_sac_model(src_path: &str) -> Result<Sac_> {
    let config = create_sac_config();
    let mut sac = Sac_::build(config);
    sac.load_params(src_path.as_ref())?;
    Ok(sac)
}

fn create_mlp(sac: &Sac_) -> MlpNoBackend {
    let vs = sac.get_policy_net().get_var_store();
    let w_names = ["mlp.al0.weight", "mlp.al1.weight", "ml.weight"];
    let b_names = ["mlp.al0.bias", "mlp.al1.bias", "ml.bias"];
    MlpNoBackend::from_varstore(vs, &w_names, &b_names)
}

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
    let  src_path = "../sac_pendulum_tch/model/best";
    let dest_path = "./model/mlp.bincode";

    let sac = load_sac_model(src_path)?;
    let mlp = create_mlp(&sac);
    serialize_to_file(&mlp, dest_path)?;

    Ok(())
}

// #[test]
// fn test() -> Result<()> {
//     let src_path = "/root/border/border/examples/gym/model/tch/sac_pendulum/best";
//     let dest_path = "/root/border/border/examples/gym/model/edge/sac_pendulum/best/mlp.bincode";

//     let sac = load_sac_model(src_path)?;
//     let mlp = create_mlp(&sac);
//     serialize_to_file(&mlp, dest_path)?;

//     Ok(())
// }
