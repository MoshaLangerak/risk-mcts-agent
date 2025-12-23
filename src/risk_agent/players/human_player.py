from risk_agent.game_elements.action import Action
from risk_agent.game_elements.game_state import GameState
from risk_agent.players.player import Player


class HumanPlayer(Player):
    """
    A player that proposed the actions via the console and asks a user to input their choice.
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
        print(f"Player {self.player_id}, it's your turn.")
        print('Legal actions:')
        for i, action in enumerate(legal_actions):
            print(f'{i}: {action}')

        while True:
            try:
                choice = int(input('Choose an action by number: '))
                if 0 <= choice < len(legal_actions):
                    return legal_actions[choice]
                else:
                    print('Invalid choice. Please try again.')
            except ValueError:
                print('Invalid input. Please enter a number.')
