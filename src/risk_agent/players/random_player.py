import random

from risk_agent.game_elements.action import Action
from risk_agent.game_elements.game_state import GameState
from risk_agent.players.player import Player


class RandomPlayer(Player):
    """
    A player that makes random decisions.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.player_id = player_id

    def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """
        Decide on an action to take based on the current game state and legal actions.
        """
        return random.choice(legal_actions)
