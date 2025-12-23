import random

from risk_agent.engine.game_engine import GameEngine
from risk_agent.game_elements.action import (
    Action,
    AttackAction,
    EndPhaseAction,
    FortifyAction,
    ReinforceAction,
    TradeCardsAction,
)
from risk_agent.game_elements.game_state import GameState
from risk_agent.players.player import Player


class BasicHeuristicPlayer(Player):
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
        # Based on the current phase, we can make a simple heuristic decision
        match game_state.current_turn_phase:
            case 'trade_cards':
                # If a card trade with a value of >= 8 is available, take it
                selected_actions = [
                    action
                    for action in legal_actions
                    if isinstance(action, TradeCardsAction) and action.value >= 8
                ]
                if selected_actions:
                    return random.choice(selected_actions)
            case 'reinforce':
                # Otherwise, only reinforce to border territories
                selected_actions = [
                    action
                    for action in legal_actions
                    if isinstance(action, ReinforceAction)
                    and action.territory
                    in GameEngine.get_border_territories(game_state, self.player_id)
                ]
                if selected_actions:
                    return random.choice(selected_actions)
            case 'attack':
                # Try to attack only is it has a ratio of at least 1.5 attacking armies to defending armies
                selected_actions = [
                    action
                    for action in legal_actions
                    if (
                        isinstance(action, AttackAction)
                        and action.attacking_armies >= 1.5 * action.defending_armies
                    )
                    or isinstance(action, EndPhaseAction)
                ]
                if selected_actions:
                    return random.choice(selected_actions)
            case 'fortify':
                # Try to fortify from a non-border territory to a border territory
                border_territories = GameEngine.get_border_territories(
                    game_state, self.player_id
                )
                selected_actions = [
                    action
                    for action in legal_actions
                    if isinstance(action, FortifyAction)
                    and action.from_territory not in border_territories
                    and action.to_territory in border_territories
                ]

                if selected_actions:
                    return random.choice(selected_actions)

        # If no specific action is selected, choose any legal action
        return random.choice(legal_actions)
