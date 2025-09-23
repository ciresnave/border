Asynchronous trainer with parallel sampling processes.

The code might look like below.

```
# use serde::{Deserialize, Serialize};
# use border_generic_replay_buffer::test::{
#     TestAgent, TestAgentConfig, TestEnv, TestObs, TestObsBatch,
#     TestAct, TestActBatch
# };
# use border_core::Env as _;
# use border_async_trainer::{
#     //test::{TestAgent, TestAgentConfig, TestEnv},
#     ActorManager, ActorManagerConfig, AsyncTrainer, AsyncTrainerConfig,
# };
# use border_generic_replay_buffer::{
#     GenericReplayBuffer, GenericReplayBufferConfig,
#     SimpleStepProcessorConfig, SimpleStepProcessor
# };
# use border_core::{
#     record::{Recorder, NullRecorder}, DefaultEvaluator,
# };
#
# use std::path::{Path, PathBuf};
#
# fn agent_config() -> TestAgentConfig {
#     TestAgentConfig
# }
#
# fn env_config() -> usize {
#     0
# }

type Env = TestEnv;
type ObsBatch = TestObsBatch;
type ActBatch = TestActBatch;
type ReplayBuffer = GenericReplayBuffer<ObsBatch, ActBatch>;
type StepProcessor = SimpleStepProcessor<Env, ObsBatch, ActBatch>;

// Create a new agent by wrapping the existing agent in order to implement SyncModel.
struct TestAgent2(TestAgent);

impl border_core::Configurable for TestAgent2 {
    type Config = TestAgentConfig;

    fn build(config: Self::Config) -> Self {
        Self(TestAgent::build(config))
    }
}

impl border_core::Agent<Env, ReplayBuffer> for TestAgent2 {
    // Boilerplate code to delegate the method calls to the inner agent.
    fn train(&mut self) {
        self.0.train();
     }

     // For other methods ...
#     fn is_train(&self) -> bool {
#         self.0.is_train()
#     }
#
#     fn eval(&mut self) {
#         self.0.eval();
#     }
#
#     fn opt_with_record(&mut self, buffer: &mut ReplayBuffer) -> border_core::record::Record {
#         self.0.opt_with_record(buffer)
#     }
#
#     fn save_params(&self, path: &Path) -> anyhow::Result<Vec<PathBuf>> {
#         self.0.save_params(path)
#     }
#
#     fn load_params(&mut self, path: &Path) -> anyhow::Result<()> {
#         self.0.load_params(path)
#     }
#
#     fn opt(&mut self, buffer: &mut ReplayBuffer) {
#         self.0.opt_with_record(buffer);
#     }
#
#     fn as_any_ref(&self) -> &dyn std::any::Any {
#         self
#     }
#
#     fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
#         self
#     }
}

impl border_core::Policy<Env> for TestAgent2 {
      // Boilerplate code to delegate the method calls to the inner agent.
      // ...
#     fn sample(&mut self, obs: &TestObs) -> TestAct {
#         self.0.sample(obs)
#     }
}

impl border_async_trainer::SyncModel for TestAgent2{
    // Self::ModelInfo shold include the model parameters.
    type ModelInfo = usize;


    fn model_info(&self) -> (usize, Self::ModelInfo) {
        // Extracts the model parameters and returns them as Self::ModelInfo.
        // The first element of the tuple is the number of optimization steps.
        (0, 0)
    }

    fn sync_model(&mut self, _model_info: &Self::ModelInfo) {
        // implements synchronization of the model based on the _model_info
    }
}

let agent_configs: Vec<_> = vec![agent_config()];
let env_config_train = env_config();
let env_config_eval = env_config();
let replay_buffer_config = GenericReplayBufferConfig::default();
let step_proc_config = SimpleStepProcessorConfig::default();
let actor_man_config = ActorManagerConfig::default();
let async_trainer_config = AsyncTrainerConfig::default();
let mut recorder: Box<dyn Recorder<_, _>> = Box::new(NullRecorder::new());
let mut evaluator = DefaultEvaluator::<TestEnv, ReplayBuffer>::new(&env_config_eval, 0, 1).unwrap();

border_async_trainer::util::train_async::<TestAgent2, _, _, StepProcessor>(
    &agent_config(),
    &agent_configs,
    &env_config_train,
    &env_config_eval,
    &step_proc_config,
    &replay_buffer_config,
    &actor_man_config,
    &async_trainer_config,
    &mut recorder,
    &mut evaluator,
);
```

Training process consists of the following two components:

* [`ActorManager`] manages [`Actor`]s, each of which runs a thread for interacting
  [`Agent`] and [`Env`] and taking samples. Those samples will be sent to
  the replay buffer in [`AsyncTrainer`].
* [`AsyncTrainer`] is responsible for training of an agent. It also runs a thread
  for pushing samples from [`ActorManager`] into a replay buffer.

The `Agent` must implement [`SyncModel`] trait in order to synchronize the model of
the agent in [`Actor`] with the trained agent in [`AsyncTrainer`]. The trait has
the ability to import and export the information of the model as
[`SyncModel`]`::ModelInfo`.

The `Agent` in [`AsyncTrainer`] is responsible for training, typically with a GPU,
while the `Agent`s in [`Actor`]s in [`ActorManager`] is responsible for sampling
using CPU.

Both [`AsyncTrainer`] and [`ActorManager`] are running in the same machine and
communicate by channels.

[`Agent`]: border_core::Agent
[`Env`]: border_core::Env