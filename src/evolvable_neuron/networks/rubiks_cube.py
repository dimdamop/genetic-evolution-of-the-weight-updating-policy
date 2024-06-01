import flax.linen as nn
from flax.typing import Array

from . import base


class Torso(nn.Module):
    cube_embed_dim: int
    time_limit: int
    step_count_embed_dim: int
    face_len: int

    @nn.compact
    def __call__(
        self,
        observation_cube: Array,
        observation_step_count: Array,
        step_count_embedder_state: Array,
    ) -> Array:

        # Cube embedding
        cube_embedding = base.Embed(
            vocab_size=self.face_len,
            embed_dim=self.cube_embed_dim,
        )(observation_cube).reshape(*observation_cube.shape[:-3], -1)

        # Step count embedding
        step_count_embedding = base.Dense(self.step_count_embed_dim)(
            observation_step_count[:, None] / self.time_limit, step_count_embedder_state
        )

        return jnp.concatenate([cube_embedding, step_count_embedding], axis=-1)


class Actor(nn.Module):
    cube_embed_dim: int
    time_limit: int
    step_count_embed_dim: int
    face_len: int
    dense_layer_dims: Tuple[int]
    num_actions: int

    @nn.compact
    def __call__(self, observation_cube: Array, observation_step_count: Array) -> Array:

        embedding = Torso(
            cube_embed_dim=self.cube_embed_dim,
            time_limit=self.time_limit,
            step_count_embed_dim=self.step_count_embed_dim,
            face_len=self.face_len,
        )(observation_cube=observation_cube, observation_step_count=observation_step_count)

        return base.MLP((*self.dense_layer_dims, self.num_actions))(embedding)


class Critic(nn.Module):
    cube_embed_dim: int
    time_limit: int
    step_count_embed_dim: int
    dense_layer_dims: Tuple[int]

    @nn.compact
    def __call__(self, observation_cube: Array, observation_step_count: Array) -> Array:

        embedding = Torso(
            cube_embed_dim=self.cube_embed_dim,
            time_limit=self.time_limit,
            step_count_embed_dim=self.step_count_embed_dim,
            face_len=self.face_len,
        )(observation_cube=observation_cube, observation_step_count=observation_step_count)

        return jnp.squeeze(base.MLP((*self.dense_layer_dims, 1))(embedding), value=-1)

