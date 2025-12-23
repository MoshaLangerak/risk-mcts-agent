from abc import ABC, abstractmethod

from risk_agent.game_elements.action import Action
from risk_agent.game_elements.game_state import GameState


class Player(ABC):
    """
    Abstract base class for a player in the game.
    """

    @abstractmethod
    def __init__(self, player_id: int) -> None:
        pass

    @abstractmethod
    def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """
        Decide on an action to take based on the current game state and legal actions.
        """
        pass

    def notify_game_state_update(self, game_state: GameState) -> None:
        """
        Notify the player of an action taken by another player.
        """
        pass
