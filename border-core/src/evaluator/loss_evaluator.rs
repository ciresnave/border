use super::Evaluator;
use crate::{record::Record, Agent, Env, ReplayBuffer};
use anyhow::Result;

/// Evaluate the agent with negative loss.
///
/// This evaluator computes the negative loss of the agent on a batch of experiences.
/// The negative loss is returned as the evaluation score, so that lower loss corresponds to higher score.
/// It uses the [`Agent::loss()`] method to compute the loss on the provided batch.
///
pub struct NegLossEvaluator<E: Env, R: ReplayBuffer>
where
    <R as ReplayBuffer>::Batch: Clone,
{
    /// Batch
    batch: R::Batch,

    phantom: std::marker::PhantomData<E>,
}

impl<E: Env, R: ReplayBuffer> Evaluator<E, R> for NegLossEvaluator<E, R>
where
    <R as ReplayBuffer>::Batch: Clone,
{
    /// Evaluate the agent on a batch of experiences.
    fn evaluate(&mut self, policy: &mut Box<dyn Agent<E, R>>) -> Result<(f32, Record)> {
        let loss = policy.loss(self.batch.clone());
        let record = Record::from_scalar("Loss", loss);
        Ok((-1.0 * loss, record))
    }
}

impl<E: Env, R: ReplayBuffer> NegLossEvaluator<E, R>
where
    <R as ReplayBuffer>::Batch: Clone,
{
    /// Constructs a new [`NegLossEvaluator`].
    pub fn new(batch: R::Batch) -> Result<Self> {
        Ok(Self {
            batch,
            phantom: std::marker::PhantomData,
        })
    }
}
