import pytest

from evolvable_neuron.networks import Actor, Critic


@pytest.fixture
def rubiks_partly_scrambled_cube_actor():
    return Actor(
        cube_embed_dim=4,
        dense_layer_dims=[256, 256],
        time_limit=20,
        step_count_embed_dim=4,
        num_actions=18,
    )


@pytest.fixture
def rubiks_partly_scrambled_cube_critic():
    return Critic(
        cube_embed_dim=4,
        dense_layer_dims=[256, 256],
        time_limit=20,
        step_count_embed_dim=4,
    )
