"""Basic environment for the roller skating minigame (codingame olymbits)"""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState:
    positions: jnp.ndarray
    risks: jnp.ndarray
    timer: int

@struct.dataclass
class EnvParams(environment.EnvParams):
    track_length: int = 10


class Rollers(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""

        actions = jnp.hstack([action, jax.random.randint(key, (2,), 0, 4)])
        # state = lax.select(state.timer > 0, self.step_(state, actions), state)
        state = lax.cond(state.timer > 0, lambda state:self.step_(state, actions), lambda s:s, state)
        done = state.timer == 0
        reward = done & (state.positions[0] == jnp.max(state.positions))
        return (
                lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                jnp.array(reward),
                done,
                {}
                )

    def step_(self, state:EnvState, actions) -> EnvState:
        positions = state.positions
        risks = state.risks
        for player in range(3):
            action = actions[player]
            pos = positions[player]
            risk = risks[player]
            pos,risk = lax.switch(
                action,
                [lambda pos,risk:(pos+1, risk-1),
                lambda pos,risk:(pos+2, risk),
                lambda pos,risk:(pos+2, risk+1),
                lambda pos,risk:(pos+3, risk+2)],
            pos, risk)
            positions = positions.at[player].set(pos)
            risks = risks.at[player].set(risk)
        for i in range(3):
            # FIXME: handle loops
            clash = risks[i] >= 0 & ((positions[(i+1)%3] == positions[i])
                                            | (positions[(i+2)%3] == positions[i]))
            risks = risks.at[i].set(lax.select(clash==1, risks[i]+2, risks[i]))
            risks = risks.at[i].set(lax.select(risks[i] >= 5, -2, risks[i]))
        return state.replace(positions=positions,
                    risks=risks,
                  timer=state.timer-1)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(
            positions=jnp.array([0, 0, 0]),
            risks=jnp.array([0, 0, 0]),
            timer=15,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        rel1 = jnp.clip(state.positions[0]-state.positions[1], -5, 5)
        rel2 = jnp.clip(state.positions[0]-state.positions[1], -5, 5)
        return jnp.hstack([state.positions/45, state.risks/5, state.timer/15, rel1/5, rel2/5])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        done = state.timer == 0
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Rollers"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [
                0,0,0, # positions
                -1,-1,-1, # risks
                0, # time
                -1,-1 # relative positions
            ]
        )
        high = jnp.array(
            [
                1,1,1, # positions
                1,1,1, # risks
                1, # time
                1,1 # relative positions
            ]
        )
        return spaces.Box(low, high, (9,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "positions": spaces.Box(0, 1, (3,), jnp.float32),
                "risks": spaces.Box(-1, 1, (3,), jnp.float32),
                "timer": spaces.Box(0, 1, (), jnp.float32),
            }
        )
