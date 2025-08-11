use anyhow::Result;
use border_candle_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    sac::{Sac, SacConfig},
    util::{actor::GaussianActorConfig, critic::MultiCriticConfig},
    Activation,
};
use border_core::{dummy::*, Agent, Configurable};
use border_policy_no_backend::Mlp as MlpNoBackend;

const DIM_OBS: i64 = 3;
const DIM_ACT: i64 = 1;

type Sac_ = Sac<DummyEnv, Mlp, Mlp2, DummyReplayBuffer>;

fn create_sac_config() -> SacConfig<Mlp, Mlp2> {
    // Omit learning related parameters
    let actor_config = GaussianActorConfig::default()
        .out_dim(DIM_ACT)
        .policy_config(MlpConfig::new(
            DIM_OBS,
            vec![64, 64],
            DIM_ACT,
            Activation::None,
        ));
    let critic_config = MultiCriticConfig::default()
        .q_config(MlpConfig::new(
            DIM_OBS + DIM_ACT,
            vec![64, 64],
            1,
            Activation::None,
        ))
        .n_nets(1);
    SacConfig::default()
        .actor_config(actor_config)
        .critic_config(critic_config)
        .device(candle_core::Device::Cpu)
}

pub fn load_sac_model(src_path: &str) -> Result<Sac_> {
    let config = create_sac_config();
    let mut sac = Sac_::build(config);
    sac.load_params(src_path.as_ref())?;
    Ok(sac)
}

pub fn create_mlp(sac: &Sac_) -> MlpNoBackend {
    let vm = sac.get_policy_net().get_var_map();
    let w_names = ["actor.mlp.ln0.weight", "actor.mlp.ln1.weight", "actor.mean.weight"];
    let b_names = ["actor.mlp.ln0.bias", "actor.mlp.ln1.bias", "actor.mean.bias"];
    MlpNoBackend::from_varmap(vm, &w_names, &b_names)
}
