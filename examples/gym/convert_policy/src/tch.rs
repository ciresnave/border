use anyhow::Result;
use border_core::{Agent, Configurable, dummy::*};
use border_policy_no_backend::Mlp as MlpNoBackend;
use border_tch_agent::{
    model::ModelBase,
    mlp::{Mlp, Mlp2, MlpConfig},
    sac::{ActorConfig, CriticConfig, SacConfig, Sac},
};

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

pub fn load_sac_model(src_path: &str) -> Result<Sac_> {
    let config = create_sac_config();
    let mut sac = Sac_::build(config);
    sac.load_params(src_path.as_ref())?;
    Ok(sac)
}

pub fn create_mlp(sac: &Sac_) -> MlpNoBackend {
    let vs = sac.get_policy_net().get_var_store();
    let w_names = ["mlp.al0.weight", "mlp.al1.weight", "ml.weight"];
    let b_names = ["mlp.al0.bias", "mlp.al1.bias", "ml.bias"];
    MlpNoBackend::from_varstore(vs, &w_names, &b_names)
}
