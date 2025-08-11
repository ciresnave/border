#![doc = include_str!("../README.md")]
mod actor;
mod actor_manager;
mod async_trainer;
mod error;
mod messages;
mod replay_buffer_proxy;
mod sync_model;
pub mod util;

pub use actor::{actor_stats_fmt, Actor, ActorStat};
pub use actor_manager::{ActorManager, ActorManagerConfig};
pub use async_trainer::{AsyncTrainStat, AsyncTrainer, AsyncTrainerConfig};
pub use error::BorderAsyncTrainerError;
pub use messages::PushedItemMessage;
pub use replay_buffer_proxy::{ReplayBufferProxy, ReplayBufferProxyConfig};
pub use sync_model::SyncModel;

/// Agent and Env for testing.
#[cfg(test)]
pub mod test {
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};

    /// Obs for testing.
    #[derive(Clone, Debug)]
    pub struct TestObs {
        obs: usize,
    }

    impl border_core::Obs for TestObs {
        fn len(&self) -> usize {
            1
        }
    }

    /// Batch of obs for testing.
    pub struct TestObsBatch {
        obs: Vec<usize>,
    }

    impl border_generic_replay_buffer::BatchBase for TestObsBatch {
        fn new(capacity: usize) -> Self {
            Self {
                obs: vec![0; capacity],
            }
        }

        fn push(&mut self, i: usize, data: Self) {
            self.obs[i] = data.obs[0];
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let obs = ixs.iter().map(|ix| self.obs[*ix]).collect();
            Self { obs }
        }
    }

    impl From<TestObs> for TestObsBatch {
        fn from(obs: TestObs) -> Self {
            Self { obs: vec![obs.obs] }
        }
    }

    /// Act for testing.
    #[derive(Clone, Debug)]
    pub struct TestAct {
        act: usize,
    }

    impl border_core::Act for TestAct {}

    /// Batch of act for testing.
    pub struct TestActBatch {
        act: Vec<usize>,
    }

    impl From<TestAct> for TestActBatch {
        fn from(act: TestAct) -> Self {
            Self { act: vec![act.act] }
        }
    }

    impl border_generic_replay_buffer::BatchBase for TestActBatch {
        fn new(capacity: usize) -> Self {
            Self {
                act: vec![0; capacity],
            }
        }

        fn push(&mut self, i: usize, data: Self) {
            self.act[i] = data.act[0];
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let act = ixs.iter().map(|ix| self.act[*ix]).collect();
            Self { act }
        }
    }

    /// Info for testing.
    pub struct TestInfo {}

    impl border_core::Info for TestInfo {}

    /// Environment for testing.
    pub struct TestEnv {
        state_init: usize,
        state: usize,
    }

    impl border_core::Env for TestEnv {
        type Config = usize;
        type Obs = TestObs;
        type Act = TestAct;
        type Info = TestInfo;

        fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> anyhow::Result<Self::Obs> {
            self.state = self.state_init;
            Ok(TestObs { obs: self.state })
        }

        fn reset_with_index(&mut self, _ix: usize) -> anyhow::Result<Self::Obs> {
            self.state = self.state_init;
            Ok(TestObs { obs: self.state })
        }

        fn step_with_reset(
            &mut self,
            a: &Self::Act,
        ) -> (border_core::Step<Self>, border_core::record::Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = border_core::Step {
                obs: TestObs { obs: self.state },
                act: a.clone(),
                reward: vec![0.0],
                is_terminated: vec![0],
                is_truncated: vec![0],
                info: TestInfo {},
                init_obs: Some(TestObs {
                    obs: self.state_init,
                }),
            };
            return (step, border_core::record::Record::empty());
        }

        fn step(&mut self, a: &Self::Act) -> (border_core::Step<Self>, border_core::record::Record)
        where
            Self: Sized,
        {
            self.state = self.state + a.act;
            let step = border_core::Step {
                obs: TestObs { obs: self.state },
                act: a.clone(),
                reward: vec![0.0],
                is_terminated: vec![0],
                is_truncated: vec![0],
                info: TestInfo {},
                init_obs: Some(TestObs {
                    obs: self.state_init,
                }),
            };
            return (step, border_core::record::Record::empty());
        }

        fn build(config: &Self::Config, _seed: i64) -> anyhow::Result<Self>
        where
            Self: Sized,
        {
            Ok(Self {
                state_init: *config,
                state: 0,
            })
        }
    }

    type ReplayBuffer =
        border_generic_replay_buffer::GenericReplayBuffer<TestObsBatch, TestActBatch>;

    /// Agent for testing.
    pub struct TestAgent {}

    #[derive(Clone, Deserialize, Serialize)]
    /// Config of agent for testing.
    pub struct TestAgentConfig;

    impl border_core::Agent<TestEnv, ReplayBuffer> for TestAgent {
        fn train(&mut self) {}

        fn is_train(&self) -> bool {
            false
        }

        fn eval(&mut self) {}

        fn opt_with_record(&mut self, _buffer: &mut ReplayBuffer) -> border_core::record::Record {
            border_core::record::Record::empty()
        }

        fn save_params(&self, _path: &Path) -> anyhow::Result<Vec<PathBuf>> {
            Ok(vec![])
        }

        fn load_params(&mut self, _path: &Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn as_any_ref(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl border_core::Policy<TestEnv> for TestAgent {
        fn sample(&mut self, _obs: &TestObs) -> TestAct {
            TestAct { act: 1 }
        }
    }

    impl border_core::Configurable for TestAgent {
        type Config = TestAgentConfig;

        fn build(_config: Self::Config) -> Self {
            Self {}
        }
    }

    impl crate::SyncModel for TestAgent {
        type ModelInfo = usize;

        fn model_info(&self) -> (usize, Self::ModelInfo) {
            (0, 0)
        }

        fn sync_model(&mut self, _model_info: &Self::ModelInfo) {
            // nothing to do
        }
    }
}
