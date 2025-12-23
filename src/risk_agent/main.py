import logging
import logging.config

import yaml

from risk_agent.engine.game_manager import GameManager


def main() -> None:
    """
    Main function to run the game.
    """
    # Load logging configuration
    with open('./data/logging.yaml') as file:
        logging_config = yaml.safe_load(file)
    logging.config.dictConfig(logging_config)

    # Initialize the game manager and load settings
    game_manager = GameManager(logging_config=logging_config)
    game_manager.load_settings('./data/settings.yaml')

    game_manager.run_game()


if __name__ == '__main__':
    main()
