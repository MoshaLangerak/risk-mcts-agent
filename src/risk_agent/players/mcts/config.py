from dataclasses import dataclass


@dataclass(frozen=True)
class MCTSConfig:
    """
    Configuration options for the MCTS algorithm.
    """

    # Game-specific parameters
    max_attacking_armies: int = 100
    max_defending_armies: int = 100
    number_of_players: int = 4

    # General MCTS parameters
    C: float = 1.41  # Exploration constant for UCT
    search_policy: str = 'max^n'  # Search policy for MCTS, max^n, Paranoid or Confident
    playout_policy: str = (
        'Random'  # Playout policy for MCTS, Random, Heuristic, or Confident
    )
    reinforce_all_heuristic: bool = False  # Heuristic to reinforce all troops
    fortify_all_heuristic: bool = False  # Heuristic to fortify all troops
    gamma: float = 1.0  # Discount factor for rewards in playouts

    # Termination condition parameters
    stopping_condition: str = 'TimeBased'  # TimeBased or IterationBased
    selection_policy: str = 'MaxChild'  # MaxChild or RobustChild
    evaluative_policy: str = (
        'MaxRobustChild'  # Dummy, MaxRobustChild, or PolicyConvergence
    )
    think_time: float = 1.0  # Time allowed for thinking when using TimeBased stopping
    max_iterations: int = 1000  # Maximum iterations in case of IterationBased stopping
    policy_convergence_window: int = 10
