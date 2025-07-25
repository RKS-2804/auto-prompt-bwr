import unittest
from trainers.JAYA_trainer import JAYATrainer

class TestJAYATrainer(unittest.TestCase):
    def setUp(self):
        self.maxiter = 10
        self.patience = 3
        self.train_seed = 42
        self.data_seed = 42
        self.num_compose = 2
        self.num_candidates = 5
        self.trainer = JAYATrainer(
            maxiter=self.maxiter,
            patience=self.patience,
            train_seed=self.train_seed,
            data_seed=self.data_seed,
            num_compose=self.num_compose,
            num_candidates=self.num_candidates
        )
        self.cfg = {
            "TRAINER": {
                "JAYA": {
                    "MAX_POPULATION_SIZE": 10,
                    "CONVERGENCE_CRITERIA": "patience"
                }
            }
        }

    def test_initialization(self):
        self.assertEqual(self.trainer.maxiter, self.maxiter)
        self.assertEqual(self.trainer.patience, self.patience)
        self.assertEqual(self.trainer.train_seed, self.train_seed)
        self.assertEqual(self.trainer.data_seed, self.data_seed)
        self.assertEqual(self.trainer.num_compose, self.num_compose)
        self.assertEqual(self.trainer.num_candidates, self.num_candidates)

    def test_initialize_population(self):
        initial_candidates = [1, 2, 3, 4, 5]
        self.trainer.initialize_population(initial_candidates)
        self.assertEqual(len(self.trainer.population), len(initial_candidates))
        self.assertEqual(self.trainer.population, initial_candidates)

    def test_update_population(self):
        initial_candidates = [1, 2, 3, 4, 5]
        self.trainer.initialize_population(initial_candidates)
        self.trainer.update_population()
        self.assertEqual(len(self.trainer.population), len(initial_candidates))

    def test_convergence(self):
        self.trainer.patience_counter = self.patience + 1
        self.assertTrue(self.trainer.has_converged())

    def test_configuration_integration(self):
        trainer_with_cfg = JAYATrainer(
            maxiter=self.maxiter,
            patience=self.patience,
            train_seed=self.train_seed,
            data_seed=self.data_seed,
            num_compose=self.num_compose,
            num_candidates=self.num_candidates,
            cfg=self.cfg
        )
        self.assertEqual(trainer_with_cfg.max_population_size, 10)
        self.assertEqual(trainer_with_cfg.convergence_criteria, "patience")

    unittest.main()
from Plum.utils.nat_inst_gpt3 import query_mistral_model

def test_mistral_model():
    """
    Test function for querying the Mistral model via OpenRouter API.
    """
    prompt = "Explain the theory of relativity."
    try:
        response = query_mistral_model(prompt)
        print("API Response:", response)
    except Exception as e:
        print("Error:", e)

# Run the test
if __name__ == "__main__":
    test_mistral_model()