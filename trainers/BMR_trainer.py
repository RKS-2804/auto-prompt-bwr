import numpy as np
from trainers.base_trainer import TrainerBase

class BMRTrainer(TrainerBase):
    """
    Implementation of the BMR algorithm as a trainer.
    """

    def __init__(self, maxiter, patience, train_seed, data_seed, num_compose, num_candidates, cfg=None):
        super(BMRTrainer, self).__init__(maxiter, patience, train_seed, data_seed, num_compose, num_candidates)
        self.population = []
        self.scores = []
        self.best_solution = None
        self.best_score = -np.inf
        self.worst_solution = None
        self.worst_score = np.inf
        self.cfg = cfg

    def initialize_population(self, initial_candidates):
        """
        Initialize the population with given candidates.
        """
        self.population = initial_candidates
        self.scores = [self.score(candidate) for candidate in self.population]
        self.update_best_and_worst()

    def update_best_and_worst(self):
        """
        Update the best and worst solutions in the population.
        """
        self.best_score = max(self.scores)
        self.best_solution = self.population[np.argmax(self.scores)]
        self.worst_score = min(self.scores)
        self.worst_solution = self.population[np.argmin(self.scores)]

    def update_population(self):
        """
        Perform updates to the population based on the BMR algorithm.
        """
        new_population = []
        for candidate in self.population:
            r4 = np.random.rand()
            if r4 > 0.5:
                # Equation (1)
                new_candidate = candidate + r4 * (self.best_solution - abs(candidate))
            else:
                # Equation (2)
                new_candidate = candidate - r4 * (self.worst_solution - abs(candidate))
            new_score = self.score(new_candidate)
            # Accept the new candidate only if it improves the score
            if new_score > self.scores[self.population.index(candidate)]:
                new_population.append(new_candidate)
                self.scores[self.population.index(candidate)] = new_score
            else:
                new_population.append(candidate)
        self.population = new_population
        self.update_best_and_worst()

    def has_converged(self):
        """
        Check if the algorithm has converged based on the configured criteria.
        """
        if self.cfg and self.cfg.TRAINER.BMR.CONVERGENCE_CRITERIA == "patience":
            return self.is_run_out_of_patience
        # Additional criteria can be added here
        return False

    def train(self, initial_candidates):
        """
        Train using the BMR algorithm.
        """
        self.initialize_population(initial_candidates)
        current_iteration = 0

        while current_iteration < self.maxiter and not self.has_converged():
            current_iteration += 1
            self.update_population()
            if self.cfg and self.cfg.LOGGING.ENABLED:
                wandb.log({"iteration": current_iteration, "best_score": self.best_score})

        return self.best_solution, self.best_score