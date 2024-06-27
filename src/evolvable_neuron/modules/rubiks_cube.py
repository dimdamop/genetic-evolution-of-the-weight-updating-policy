# Copyright 2022 InstaDeep Ltd. All rights reserved.
# Copyright 2024 Dimitrios Damopoulos. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
#
# This source code is a modified version of `jumanji.training.rubiks_cube.actor_critic`, aiming
# mainly at building on top of `flax` instead of `haiku`.

import flax.linen as nn
from flax.typing import Array, Tuple
from jax import numpy as jnp
from jumanji.environments.logic.rubiks_cube.types import Observation

from evolvable_neuron.modules import base


class Torso(nn.Module):
    cube_embed_dim: int
    time_limit: int
    step_count_embed_dim: int
    face_len: int

    def setup(self):
        self.obs_embedder = base.Embed(
            vocab_size=self.face_len,
            embed_dim=self.cube_embed_dim,
        )
        self.step_embedder = nn.Dense(self.step_count_embed_dim)

    def __call__(self, observation: Observation) -> Array:
        embedded_obs = self.obs_embedder(observation.cube).reshape(*observation.cube.shape[:-3], -1)
        embedded_step = self.step_embedder(observation.step_count[:, None] / self.time_limit)
        return jnp.concatenate([embedded_obs, embedded_step], axis=-1)


class Actor(nn.Module):
    cube_embed_dim: int
    time_limit: int
    step_count_embed_dim: int
    dense_layer_dims: Tuple[int]
    num_actions: int
    face_len: int = 6

    def setup(self):
        self.obs_embedder = Torso(
            cube_embed_dim=self.cube_embed_dim,
            time_limit=self.time_limit,
            step_count_embed_dim=self.step_count_embed_dim,
            face_len=self.face_len,
        )
        self.mlp = base.MLP((*self.dense_layer_dims, self.num_actions), depth=1)

    def __call__(self, observation: Observation) -> Array:
        embedded_obs = self.obs_embedder(observation)
        latent = self.mlp(embedded_obs)
        return latent


class Critic(nn.Module):
    cube_embed_dim: int
    time_limit: int
    step_count_embed_dim: int
    dense_layer_dims: Tuple[int]
    face_len: int = 6

    def setup(self):
        self.obs_embedder = Torso(
            cube_embed_dim=self.cube_embed_dim,
            time_limit=self.time_limit,
            step_count_embed_dim=self.step_count_embed_dim,
            face_len=self.face_len,
        )
        self.mlp = base.MLP((*self.dense_layer_dims, 1), depth=1)

    def __call__(self, observation: Array) -> Array:
        embedded_obs = self.obs_embedder(observation)
        latent = self.mlp(embedded_obs)
        return jnp.squeeze(latent, axis=-1)
