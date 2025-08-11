/// Message containing a [`ReplayBuffer`](border_core::ReplayBuffer)`::Item`.
///
/// It will be sent from [`Actor`](crate::Actor) to [`ActorManager`](crate::ActorManager).
pub struct PushedItemMessage<T> {
    /// ID of [`Actor`](crate::Actor) which generates samples (`pushed_item`).
    pub id: usize,

    /// A batch.
    pub pushed_items: Vec<T>,
}
