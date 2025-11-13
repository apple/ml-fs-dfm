import torch
import torch.nn.functional as F
import unittest
from discrete_solver_all import BaseJumperSolver, MixtureDiscreteEulerSolver


class TestMixtureDiscreteEulerSolver(unittest.TestCase):
    def setUp(self):
        self.vocabulary_size = 10
        self.batch_size = 2
        self.seq_length = 5

        class DummySolver(BaseJumperSolver):
            pass

        self.solver = MixtureDiscreteEulerSolver(
            model=None, path=None, vocabulary_size=self.vocabulary_size
        )
        self.solver.mask_token = -1

    def test_step_no_controlled_unmasking(self):
        x_t = torch.randint(0, self.vocabulary_size, (self.batch_size, self.seq_length))
        u = torch.rand(self.batch_size, self.seq_length, self.vocabulary_size)
        h = torch.rand(self.batch_size)

        x_next = self.solver._step(x_t, u, h, controlled_unmasking=False)
        self.assertEqual(x_next.shape, x_t.shape)
        self.assertTrue(torch.all(x_next >= 0) and torch.all(x_next < self.vocabulary_size))

    def test_step_controlled_unmasking(self):
        x_t = torch.randint(0, self.vocabulary_size, (self.batch_size, self.seq_length))
        u = torch.rand(self.batch_size, self.seq_length, self.vocabulary_size)
        h = torch.rand(self.batch_size)

        x_next = self.solver._step(x_t, u, h, controlled_unmasking=True)
        self.assertEqual(x_next.shape, x_t.shape)
        self.assertTrue(torch.all(x_next >= 0) and torch.all(x_next < self.vocabulary_size))

    def test_step_with_mask_token(self):
        self.solver.mask_token = 0
        x_t = torch.randint(0, self.vocabulary_size, (self.batch_size, self.seq_length))
        x_t[:, :2] = 0  # set mask token
        u = torch.rand(self.batch_size, self.seq_length, self.vocabulary_size)
        h = torch.rand(self.batch_size)

        x_next = self.solver._step(x_t, u, h, unmask_change=False, controlled_unmasking=True)
        self.assertEqual(x_next.shape, x_t.shape)
        self.assertTrue(torch.all(x_next >= 0) and torch.all(x_next < self.vocabulary_size))
        self.assertTrue(torch.all(x_next[:, 2:] == x_t[:, 2:]))  # unchanged non-mask tokens


if __name__ == '__main__':
    unittest.main()
