import abc
import time


class StoppingCondition(abc.ABC):
    """
    Abstract base class for stopping conditions in MCTS.
    """

    def __init__(self, stopping_condition_parameters: dict) -> None:
        self.stopping_condition_parameters = stopping_condition_parameters

    @abc.abstractmethod
    def start(self) -> None:
        """Called once at the beginning of the thinking loop."""
        pass

    @abc.abstractmethod
    def is_met(self) -> bool:
        """Returns True if the thinking loop should stop."""
        raise NotImplementedError

    def update(self) -> None:
        """Called once per MCTS iteration. Optional for some strategies."""
        pass


class TimeBasedStoppingCondition(StoppingCondition):
    """
    A stopping condition that triggers after a specified time limit.
    """

    def __init__(self, think_time: float, selection_policy: str) -> None:
        try:
            self.think_time = think_time
            self.selection_policy = selection_policy
            self.end_time = 0
        except ValueError:
            raise ValueError('Missing parameters in stopping condition parameters.')

    def start(self) -> None:
        self.end_time = time.time() + self.think_time

    def is_met(self) -> bool:
        return time.time() >= self.end_time


class IterationBasedStoppingCondition(StoppingCondition):
    """
    Stops after a maximum number of iterations
    or when the selection_policy criterium is met.
    """

    def __init__(
        self, max_iterations: int, selection_policy: str, evaluative_policy: str
    ) -> None:
        try:
            self.max_iterations = max_iterations
            self.selection_policy = selection_policy
            self.evaluative_policy = evaluative_policy
        except ValueError:
            raise ValueError('Missing parameters in stopping condition parameters.')

        self.current_iterations = 0

    def start(self) -> None:
        self.current_iterations = 0

    def is_met(self) -> bool:
        return self.current_iterations >= self.max_iterations

    def update(self) -> None:
        """This strategy needs to count iterations."""
        self.current_iterations += 1
