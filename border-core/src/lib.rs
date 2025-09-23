#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
pub mod dummy;
pub mod error;
mod evaluator;
pub mod record;

mod base;
pub use base::{
    Act, Agent, Configurable, Env, ExperienceBuffer, Info, NullReplayBuffer, Obs, Policy,
    ReplayBuffer, Step, StepProcessor, TransitionBatch,
};

mod trainer;
pub use evaluator::{DefaultEvaluator, Evaluator, NegLossEvaluator};
pub use trainer::{Sampler, Trainer, TrainerConfig};
