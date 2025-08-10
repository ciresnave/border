Generic implementation of replay buffers for reinforcement learning.

This module provides a flexible implementation of replay buffers
that can handle arbitrary observation and action types. It supports both
standard experience replay and prioritized experience replay (PER).

# Key Components

- [`GenericReplayBuffer`]: A generic replay buffer implementation
- [`GenericTransitionBatch`]: A generic batch structure for transitions
- [`SimpleStepProcessor`]: A processor for converting environment steps to transitions
- [`PerConfig`]: Configuration for prioritized experience replay

# Features

- Generic type support for observations and actions
- Efficient batch processing
- Prioritized experience replay with importance sampling
- Configurable weight normalization
- Step processing for non-vectorized environments

# [`BatchBase`]

# [`GenericTransitionBatch`]

The [`TransitionBatch`] trait represents a batch of transitions in the form `(o_t, r_t, a_t, o_t+1)`.
This trait is used for training [`Agent`]s using reinforcement learning algorithms.

# [`GenericReplayBuffer`]

[`GenericReplayBuffer<O, A>`] implements both [`ReplayBuffer`] and [`ExperienceBuffer`].
This type takes two parameters, `O` and `A`, representing observation and action types in the replay buffer.
Both `O` and `A` must implement [`BatchBase`], which provides sample storage functionality similar to `Vec<T>`.
The associated types `Item` and `Batch` are both [`GenericTransitionBatch`], representing sets of
`(o_t, r_t, a_t, o_t+1)` transitions.

# [`SimpleStepProcessor`]

The [`SimpleStepProcessor<E, O, A>`] is a concrete implementation that:
1. Maintains the previous observation to construct complete transitions
2. Converts environment-specific observations and actions (`E::Obs` and `E::Act`) into batch-compatible
   types (`O` and `A`) using the `From` trait
3. Generates [`GenericTransitionBatch`] objects containing the complete transition
   `(o_t, a_t, o_t+1, r_t, is_terminated, is_truncated)`
4. Handles episode termination by properly resetting the previous observation

This processor is essential for implementing temporal difference learning algorithms, as it ensures
that transitions are properly formatted and stored in the replay buffer for training.

[`SimpleStepProcessor<E, O, A>`] can be used with [`GenericReplayBuffer<O, A>`]. It converts `E::Obs` and
`E::Act` into their respective [`BatchBase`] types and generates [`GenericTransitionBatch`]. This conversion
relies on the trait bounds `O: From<E::Obs>` and `A: From<E::Act>`.

[`GenericReplayBuffer`]: crate::GenericReplayBuffer
[`GenericReplayBuffer<O, A>`]: crate::GenericReplayBuffer
[`BatchBase`]: crate::BatchBase
[`GenericTransitionBatch`]: crate::GenericTransitionBatch
[`SimpleStepProcessor`]: crate::SimpleStepProcessor
[`SimpleStepProcessor<E, O, A>`]: crate::SimpleStepProcessor
[`BatchBase`]: crate::BatchBase
[`ReplayBuffer`]: border_core::ReplayBuffer
[`ExperienceBuffer`]: border_core::ExperienceBuffer
[`Agent`]: border_core::Agent
