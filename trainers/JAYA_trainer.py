from trainers.base_trainer import TrainerBase
import numpy as np
import wandb

class JAYATrainer(TrainerBase):
    """
    Implementation of the JAYA algorithm as a trainer.
    """

    def __init__(self, maxiter, patience, train_seed, data_seed, num_compose, num_candidates, cfg=None):
        super(JAYATrainer, self).__init__(maxiter, patience, train_seed, data_seed, num_compose, num_candidates)
        self.population = []
        self.scores = []
        self.best_solution = None
        self.best_score = -np.inf
        self.worst_solution = None
        self.worst_score = np.inf
        self.cfg = cfg
        self.max_population_size = cfg.TRAINER.JAYA.MAX_POPULATION_SIZE if cfg else 50
        self.convergence_criteria = cfg.TRAINER.JAYA.CONVERGENCE_CRITERIA if cfg else "patience"

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
        Perform deterministic updates to the population based on the JAYA algorithm.
        Only accept new candidates if they improve upon the old ones.
        """
        new_population = []
        for i, candidate in enumerate(self.population):
            r1, r2 = np.random.rand(), np.random.rand()
            new_candidate = candidate + r1 * (self.best_solution - abs(candidate)) - r2 * (self.worst_solution - abs(candidate))
            new_score = self.score(new_candidate)
            # Accept the new candidate only if it improves the score
            if new_score > self.scores[i]:
                new_population.append(new_candidate)
                self.scores[i] = new_score
            else:
                new_population.append(candidate)
        self.population = new_population
        self.update_best_and_worst()

    def has_converged(self):
        """
        Check if the algorithm has converged based on the configured criteria.
        """
        if self.convergence_criteria == "patience":
            return self.is_run_out_of_patience
        # Additional criteria can be added here
        return False

    def train(self, initial_candidates):
        """
        Train using the JAYA algorithm.
        """
        self.initialize_population(initial_candidates)
        current_iteration = 0

        while current_iteration < self.maxiter and not self.has_converged():
            current_iteration += 1
            self.update_population()
            wandb.log({"iteration": current_iteration, "best_score": self.best_score})

        return self.best_solution, self.best_score