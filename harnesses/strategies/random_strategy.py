"""Random baseline strategy. Useful as a comparison."""

import random
from arcengine import GameAction
from ..base import Strategy
from ..perception import GameObservation


class RandomStrategy(Strategy):
    name = "random"

    def __init__(self, action_space: list[int] = None, **kwargs):
        self.action_space = action_space or [1, 2, 3, 4]

    def reset(self):
        pass

    def choose_action(self, obs: GameObservation) -> GameAction:
        _by_id = {a.value: a for a in GameAction}
        available = obs.available_actions or [_by_id[a] for a in self.action_space]
        return random.choice(available)

    def on_step_result(self, action: GameAction, obs: GameObservation):
        pass
