use border_core::{Agent, Env, NullReplayBuffer, Policy};
use serde::{Deserialize, Serialize};

use crate::{Mat, Mlp};

/// MLP-based agent for reinforcement learning.
///
/// This agent uses a multilayer perceptron (MLP) as its policy network.
/// The MLP outputs actions in the range [-1, 1] due to the tanh activation
/// function applied to the output layer.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MlpAgent {
    mlp: Mlp,
}

impl<E> Policy<E> for MlpAgent
where
    E: Env,
    E::Obs: Into<Mat>,
    E::Act: From<Mat>,
{
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let obs_mat: Mat = obs.clone().into();
        let act_mat = self.mlp.forward(&obs_mat);
        act_mat.into()
    }
}

/// This Agent trait implementation does nothing, but is required when passing
/// this policy to border_core::Evaluator for evaluation. The Evaluator accepts
/// trait objects that implement the Agent trait, and trait objects cannot be
/// upcast to Policy trait objects, so the Agent trait object is used instead.
impl<E> Agent<E, NullReplayBuffer> for MlpAgent
where
    E: Env,
    E::Obs: Into<Mat>,
    E::Act: From<Mat>,
{
}

impl MlpAgent {
    /// Creates a new MlpAgent with the given MLP.
    ///
    /// # Arguments
    ///
    /// * `mlp` - The MLP network to use as the policy
    ///
    /// # Returns
    ///
    /// A new MlpAgent instance
    pub fn new(mlp: Mlp) -> Self {
        Self { mlp }
    }

    /// Returns a reference to the underlying MLP.
    pub fn mlp(&self) -> &Mlp {
        &self.mlp
    }
}
