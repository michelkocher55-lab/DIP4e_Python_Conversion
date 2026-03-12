import unittest
import numpy as np
from helpers.libdip.neuralNet4e import neuralNet4e

class TestNeuralNet4e(unittest.TestCase):

    def test_train_xor(self):
        """Test XOR problem training."""
        # XOR problem
        X = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]]) # 2x4
        # R: Class 1 (0), Class 2 (1).
        # We need 2 output nodes for one-hot? Or 1 node?
        # neuralNet4e supports vectors. Let's use 2 output nodes.
        # 0^0=0 -> [1, 0]
        # 0^1=1 -> [0, 1]
        # 1^0=1 -> [0, 1]
        # 1^1=0 -> [1, 0]
        
        R = np.array([[1, 0, 0, 1],
                      [0, 1, 1, 0]]) # 2x4
        
        input_data = {
            'X': X,
            'R': R,
            'Epochs': 2000 # Enough to converge
        }
        
        specs = {
            'Nodes': [2, 4, 2], # 2 In -> 4 Hidden -> 2 Out
            'Activation': 'sigmoid',
            'Mode': 'train',
            'Correction': 0.5
        }
        
        # Train
        np.random.seed(42)
        output = neuralNet4e(input_data, specs)
        
        # Check convergence
        self.assertTrue(output['MSE'][-1] < 0.1)
        self.assertEqual(output['RecogRate'], 100.0)
        
        # Classify Check
        specs['Mode'] = 'classify'
        specs['W'] = output['W']
        specs['b'] = output['b']
        input_data['Epochs'] = 1 # ignored
        
        out_cls = neuralNet4e(input_data, specs)
        
        expected_class = np.argmax(R, axis=0) # [0, 1, 1, 0]
        np.testing.assert_array_equal(out_cls['Class'], expected_class)
        
        print("NeuralNet4e XOR Test Passed.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
