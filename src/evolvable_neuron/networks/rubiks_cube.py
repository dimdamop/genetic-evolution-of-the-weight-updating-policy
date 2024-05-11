from typing import Callable, Sequence

from chex import Array
import haiku as hk
import jax.numpy as jnp

from jumanji.environments.logic.rubiks_cube.constants import Face
from jumanji.environments.logic.rubiks_cube.env import Observation
from jumanji.training.networks.actor_critic import FeedForwardNetwork
from .layers import Linear, MLP


def torso(
    cube_embed_dim: int, time_limit: int, step_count_embed_dim: int
) -> Callable[[Observation], Array]:
    def torso_network_fn(observation: Observation) -> Array:
        # Cube embedding
        cube_embedder = hk.Embed(vocab_size=len(Face), embed_dim=cube_embed_dim)
        cube_embedding = cube_embedder(observation.cube).reshape(*observation.cube.shape[:-3], -1)

        # Step count embedding
        step_count_embedder = Linear(step_count_embed_dim)
        step_count_embedding = step_count_embedder(observation.step_count[:, None] / time_limit)

        return jnp.concatenate([cube_embedding, step_count_embedding], axis=-1)

    return torso_network_fn


def actor(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
    num_actions: int,
) -> FeedForwardNetwork:
    torso_network_fn = torso(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )

    def network_fn(observation: Observation) -> Array:
        embedding = torso_network_fn(observation)
        logits = MLP((*dense_layer_dims, num_actions))(embedding)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def critic(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
) -> FeedForwardNetwork:
    torso_network_fn = torso(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )

    def network_fn(observation: Observation) -> Array:
        embedding = torso_network_fn(observation)
        value = MLP((*dense_layer_dims, 1))(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
