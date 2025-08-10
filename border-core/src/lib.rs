#![warn(missing_docs)]
//! Core components for reinforcement learning.
//!
//! # Observation and Action
//!
//! The [`Obs`] and [`Act`] traits provide abstractions for observations and actions in environments.
//!
//! # Environment
//!
//! The [`Env`] trait serves as the fundamental abstraction for environments. It defines four associated types:
//! `Config`, `Obs`, `Act`, and `Info`. The `Obs` and `Act` types represent concrete implementations of
//! environment observations and actions, respectively. These types must implement the [`Obs`] and [`Act`] traits.
//! Environments implementing [`Env`] generate [`Step<E: Env>`] objects at each interaction step through the
//! [`Env::step()`] method. The [`Info`] type stores additional information from each agent-environment interaction,
//! which may be empty (implemented as a zero-sized struct). The `Config` type represents environment configurations
//! and is used during environment construction.
//!
//! # Policy
//!
//! The [`Policy<E: Env>`] trait represents a decision-making policy. The [`Policy::sample()`] method takes an
//! `E::Obs` and generates an `E::Act`. Policies can be either probabilistic or deterministic, depending on the
//! implementation.
//!
//! # Agent
//!
//! In this crate, an [`Agent<E: Env, R: ReplayBuffer>`] is defined as a trainable [`Policy<E: Env>`].
//! Agents operate in either training or evaluation mode. During training, the agent's policy may be probabilistic
//! to facilitate exploration, while in evaluation mode, it typically becomes deterministic.
//!
//! The [`Agent::opt()`] method executes a single optimization step. The specific implementation of an optimization
//! step varies between agents and may include multiple stochastic gradient descent steps. Training samples are
//! obtained from the [`ReplayBuffer`].
//!
//! This trait also provides methods for saving and loading trained policy parameters to and from a directory.
//!
//! # Batch
//!
//! The [`TransitionBatch`] trait represents a batch of transitions in the form `(o_t, r_t, a_t, o_t+1)`.
//! This trait is used for training [`Agent`]s using reinforcement learning algorithms.
//!
//! # Replay Buffer and Experience Buffer
//!
//! The [`ReplayBuffer`] trait provides an abstraction for replay buffers. Its associated type
//! [`ReplayBuffer::Batch`] represents samples used for training [`Agent`]s. Agents must implement the
//! [`Agent::opt()`] method, where [`ReplayBuffer::Batch`] must have appropriate type or trait bounds
//! for training the agent.
//!
//! While [`ReplayBuffer`] focuses on generating training batches, the [`ExperienceBuffer`] trait
//! handles sample storage. The [`ExperienceBuffer::push()`] method stores samples of type
//! [`ExperienceBuffer::Item`], typically obtained through environment interactions.
//!
//! ## Reference Implementation
//!
//! [`SimpleReplayBuffer<O, A>`] implements both [`ReplayBuffer`] and [`ExperienceBuffer`].
//! This type takes two parameters, `O` and `A`, representing observation and action types in the replay buffer.
//! Both `O` and `A` must implement [`BatchBase`], which provides sample storage functionality similar to `Vec<T>`.
//! The associated types `Item` and `Batch` are both [`GenericTransitionBatch`], representing sets of
//! `(o_t, r_t, a_t, o_t+1)` transitions.
//!
//! # Step Processor
//!
//! The [`StepProcessor`] trait plays a crucial role in the training pipeline by transforming environment
//! interactions into training samples. It processes [`Step<E: Env>`] objects, which contain the current
//! observation, action, reward, and next observation, into a format suitable for the replay buffer.
//!
//! The [`SimpleStepProcessor<E, O, A>`] is a concrete implementation that:
//! 1. Maintains the previous observation to construct complete transitions
//! 2. Converts environment-specific observations and actions (`E::Obs` and `E::Act`) into batch-compatible
//!    types (`O` and `A`) using the `From` trait
//! 3. Generates [`GenericTransitionBatch`] objects containing the complete transition
//!    `(o_t, a_t, o_t+1, r_t, is_terminated, is_truncated)`
//! 4. Handles episode termination by properly resetting the previous observation
//!
//! This processor is essential for implementing temporal difference learning algorithms, as it ensures
//! that transitions are properly formatted and stored in the replay buffer for training.
//!
//! [`SimpleStepProcessor<E, O, A>`] can be used with [`SimpleReplayBuffer<O, A>`]. It converts `E::Obs` and
//! `E::Act` into their respective [`BatchBase`] types and generates [`GenericTransitionBatch`]. This conversion
//! relies on the trait bounds `O: From<E::Obs>` and `A: From<E::Act>`.
//!
//! # Trainer
//!
//! The [`Trainer`] manages the training loop and related objects. A [`Trainer`] instance is configured with
//! training parameters such as the maximum number of optimization steps and the directory for saving agent
//! parameters during training. The [`Trainer::train`] method executes online training of an agent in an environment.
//! During the training loop, the agent interacts with the environment to collect samples and perform optimization
//! steps, while simultaneously recording various metrics.
//!
//! # Evaluator
//!
//! The [`Evaluator<E, P>`] trait is used to evaluate a policy's (`P`) performance in an environment (`E`).
//! An instance of this type is provided to the [`Trainer`] for policy evaluation during training.
//! [`DefaultEvaluator<E, P>`] serves as the default implementation of [`Evaluator<E, P>`]. This evaluator
//! runs the policy in the environment for a specified number of episodes. At the start of each episode,
//! the environment is reset using [`Env::reset_with_index()`] to control specific evaluation conditions.
//!
//! [`SimpleReplayBuffer`]: generic_replay_buffer::SimpleReplayBuffer
//! [`SimpleReplayBuffer<O, A>`]: generic_replay_buffer::SimpleReplayBuffer
//! [`BatchBase`]: generic_replay_buffer::BatchBase
//! [`GenericTransitionBatch`]: generic_replay_buffer::GenericTransitionBatch
//! [`SimpleStepProcessor`]: generic_replay_buffer::SimpleStepProcessor
//! [`SimpleStepProcessor<E, O, A>`]: generic_replay_buffer::SimpleStepProcessor
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
pub use evaluator::{DefaultEvaluator, Evaluator};
pub use trainer::{Sampler, Trainer, TrainerConfig};
