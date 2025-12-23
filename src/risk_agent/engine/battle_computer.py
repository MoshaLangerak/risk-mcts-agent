import logging
import os
import random

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix

logger = logging.getLogger(__name__)


class BattleComputer:
    def __init__(self, max_attacking_armies: int, max_defending_armies: int) -> None:
        self.max_attacking_armies = max_attacking_armies
        self.max_defending_armies = max_defending_armies

        stationary_distribution_file_path = f'data/stationary_distributions/stationary_distribution_{max_attacking_armies}_{max_defending_armies}.npz'  # noqa: E501

        # Transition probabilities for the smallest battles, from Osborne (2003)
        self.transition_probabilities = {
            (3, 2): [
                (2, 0, 2275 / 7776),  # Attacker loses 2, Defender loses 0
                (1, 1, 2611 / 7776),  # Attacker loses 1, Defender loses 1
                (0, 2, 2890 / 7776),  # Attacker loses 0, Defender loses 2
            ],
            (3, 1): [
                (1, 0, 441 / 1296),  # Attacker loses 1, Defender loses 0
                (0, 1, 855 / 1296),  # Attacker loses 0, Defender loses 1
            ],
            (2, 2): [
                (2, 0, 581 / 1296),  # Attacker loses 2, Defender loses 0
                (1, 1, 420 / 1296),  # Attacker loses 1, Defender loses 1
                (0, 2, 295 / 1296),  # Attacker loses 0, Defender loses 2
            ],
            (2, 1): [
                (1, 0, 91 / 216),  # Attacker loses 1, Defender loses 0
                (0, 1, 125 / 216),  # Attacker loses 0, Defender loses 1
            ],
            (1, 2): [
                (1, 0, 161 / 216),  # Attacker loses 1, Defender loses 0
                (0, 1, 55 / 216),  # Attacker loses 0, Defender loses 1
            ],
            (1, 1): [
                (1, 0, 21 / 36),  # Attacker loses 1, Defender loses 0
                (0, 1, 15 / 36),  # Attacker loses 0, Defender loses 1
            ],
        }

        # Load or compute the stationary distribution
        if os.path.exists(stationary_distribution_file_path):
            self.stationary_distribution = sp.load_npz(
                stationary_distribution_file_path
            )
        else:
            probability_matrix = self._initialise_probability_matrix()
            probability_matrix = self._fill_probability_matrix(probability_matrix)
            self.stationary_distribution = self._compute_stationary_distribution(
                probability_matrix
            )
            directory = os.path.dirname(stationary_distribution_file_path)
            os.makedirs(directory, exist_ok=True)
            sp.save_npz(
                stationary_distribution_file_path,
                self.stationary_distribution,
            )

    def _initialise_probability_matrix(self) -> lil_matrix:
        """
        Initialise a probability matrix for the battle outcomes.

        Returns:
            A sparse matrix of size (max_attacking_armies + 1) * (max_defending_armies + 1)
            by (max_attacking_armies + 1) * (max_defending_armies + 1).
        """
        size = (
            (self.max_attacking_armies + 1) * (self.max_defending_armies + 1),
            (self.max_attacking_armies + 1) * (self.max_defending_armies + 1),
        )
        return lil_matrix(size, dtype=float)

    def _fill_probability_matrix(self, matrix: lil_matrix) -> lil_matrix:
        """
        Fill the probability matrix with the transition probabilities.
        """

        def state_to_index(state: tuple[int, int]) -> int:
            """
            Convert a state (attacker_armies, defender_armies) to a matrix index.
            """
            return state[0] * (self.max_defending_armies + 1) + state[1]

        for attacker_armies in range(0, self.max_attacking_armies + 1):
            for defender_armies in range(0, self.max_defending_armies + 1):
                current_state_idx = state_to_index((attacker_armies, defender_armies))

                if attacker_armies == 0 or defender_armies == 0:
                    matrix[current_state_idx, current_state_idx] = 1.0
                    continue

                engaged_attacker_armies = min(attacker_armies, 3)
                engaged_defender_armies = min(defender_armies, 2)

                battle_outcomes = self.transition_probabilities.get(
                    (engaged_attacker_armies, engaged_defender_armies), []
                )

                for attacker_loss, defender_loss, probability in battle_outcomes:
                    new_attacker_armies = max(0, attacker_armies - attacker_loss)
                    new_defender_armies = max(0, defender_armies - defender_loss)

                    new_state_idx = state_to_index(
                        (new_attacker_armies, new_defender_armies)
                    )

                    matrix[current_state_idx, new_state_idx] = probability
        return matrix

    def _compute_stationary_distribution(self, matrix: lil_matrix) -> csr_matrix:
        """
        Compute the stationary distribution of the Markov chain represented by the matrix.
        """
        sparse_matrix = matrix.tocsr()

        # Compute the stationary distribution using the power method
        stationary_distribution = sp.linalg.matrix_power(
            sparse_matrix, sparse_matrix.shape[0] + sparse_matrix.shape[1]
        )

        return stationary_distribution.tocsr()

    def _index_to_state(self, index: int) -> tuple[int, int]:
        """
        Convert a matrix index back to a state (attacker_armies, defender_armies).
        """
        return (
            int(index // (self.max_defending_armies + 1)),
            int(index % (self.max_defending_armies + 1)),
        )

    def get_outcome_probabilities(
        self, attacking_armies: int, defending_armies: int
    ) -> np.ndarray:
        """
        Get the outcome probabilities for a given number of attacking and defending armies.
        """
        if attacking_armies < 0 or defending_armies < 0:
            raise ValueError('Number of armies must be non-negative.')

        if attacking_armies > self.max_attacking_armies:
            raise ValueError(
                f'Number of attacking armies exceeds maximum ({self.max_attacking_armies}).'
            )
        if defending_armies > self.max_defending_armies:
            raise ValueError(
                f'Number of defending armies exceeds maximum ({self.max_defending_armies}).'
            )

        return self.stationary_distribution[
            attacking_armies * (self.max_defending_armies + 1) + defending_armies
        ]

    def get_all_outcomes(
        self, attacking_armies: int, defending_armies: int
    ) -> list[tuple[float, tuple[int, int]]]:
        """
        Gets all possible outcomes and their probabilities for a battle scenario.

        Returns:
            A list of tuples, where each tuple contains (probability, (final_attackers, final_defenders)),
            sorted from most likely to least likely.
        """
        prob_row = self.get_outcome_probabilities(attacking_armies, defending_armies)

        outcomes = []
        # .indices holds the column index of each non-zero element (the final state)
        # .data holds the value of each non-zero element (the probability)
        for i in range(prob_row.nnz):
            probability = prob_row.data[i]
            state_index = prob_row.indices[i]
            state = self._index_to_state(state_index)
            outcomes.append((probability, state))

        # Sort by probability in descending order
        outcomes.sort(key=lambda x: x[1][0], reverse=True)

        # Filter out outcomes with zero probability
        outcomes = [(float(prob), state) for prob, state in outcomes if prob > 0]

        return outcomes

    def get_attacker_win_rate(
        self, attacking_armies: int, defending_armies: int
    ) -> float:
        """
        Get the total probabilities of attacker win rate given the number of
        attacking and defending armies.
        """
        all_outcomes = self.get_all_outcomes(attacking_armies, defending_armies)

        attacker_win_prob = 0.0
        for prob, state in all_outcomes:
            final_attackers, final_defenders = state
            if final_defenders == 0 and final_attackers > 0:
                attacker_win_prob += prob

        return attacker_win_prob

    def get_outcome(
        self, attacking_armies: int, defending_armies: int
    ) -> tuple[int, int]:
        """
        Randomly determine the outcome of a battle given the number of attacking and defending armies.
        """
        probabilities = self.get_outcome_probabilities(
            attacking_armies, defending_armies
        )

        possible_outcomes = probabilities.indices
        probabilities = probabilities.data

        outcome_index = random.choices(possible_outcomes, weights=probabilities, k=1)[0]
        return self._index_to_state(outcome_index)


if __name__ == '__main__':
    import time

    # configure logging
    logging.basicConfig(level=logging.INFO)
    logger.info('Starting BattleComputer simulation...')

    start_time = time.time()
    battle_computer = BattleComputer(max_attacking_armies=500, max_defending_armies=500)
    end_time = time.time()
    logger.info(f'BattleComputer initialized in {end_time - start_time:.2f} seconds.')

    attacking_armies = 3
    defending_armies = 2

    print('Stationary Distribution Stats:')
    print(f'Shape: {battle_computer.stationary_distribution.shape}')
    print(
        f'Total elements: {battle_computer.stationary_distribution.shape[0] * battle_computer.stationary_distribution.shape[1]}'
    )
    print(f'Number of non-zero elements: {battle_computer.stationary_distribution.nnz}')

    logger.info(
        f'Running a single random simulation for {attacking_armies} attackers vs. '
        f'{defending_armies} defenders.'
    )
    outcome = battle_computer.get_outcome(attacking_armies, defending_armies)
    logger.info(
        f'Simulated Final Outcome: {outcome} '
        f'(Attacking armies: {attacking_armies}, Defending armies: {defending_armies})'
    )

    logger.info(
        f'Showing all possible outcomes for {attacking_armies} attackers vs. '
        f'{defending_armies} defenders:\n'
    )

    all_outcomes = battle_computer.get_all_outcomes(attacking_armies, defending_armies)

    for probability, state in all_outcomes:
        final_attackers, final_defenders = state
        logger.info(
            f'Probability: {probability:.4f}, '
            f'Final State: (Attackers: {final_attackers}, Defenders: {final_defenders})'
        )

    attacker_win_prob = 0

    for probability, state in all_outcomes:
        final_attackers, final_defenders = state
        # Attacker wins if defenders are eliminated
        if final_defenders == 0 and final_attackers > 0:
            attacker_win_prob += probability

    logger.info(f'Total Attacker Win Probability: {attacker_win_prob:.2%}')
