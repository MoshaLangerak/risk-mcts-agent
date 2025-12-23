# Expose the main player class and config directly
from .config import MCTSConfig
from .player import ConfidentMCTSPlayer, MCTSPlayer

# This allows you to write 'from risk_agent.players.mcts import MCTSPlayer'
# instead of 'from risk_agent.players.mcts.player import MCTSPlayer'
