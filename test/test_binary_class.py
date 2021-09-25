# Standard library imports
import unittest

# Third party imports
import numpy as np
import tensorflow as tf
import torch

# Local imports

class Test_Binary_Class(unittest.TestCase):

    def test_nullify(self):

        input_numpy = np.array([[1.,0.,0.,1.,0.,1.,1.,0.]])
        
        self.assertEqual(nullify_all(input_numpy), np.array([[0.,0.,0.,0.,0.,0.,0.,0.]]), 
                         'Numpy array test')
    
    def test_oneify(self):
        
        input_numpy = np.array([[1.,0.,0.,1.,0.,1.,1.,0.]])
        
        self.assertEqual(oneify_all(input_numpy), np.array([[1.,1.,1.,1.,1.,1.,1.,1.]]), 
                         'Numpy array test')
    
    
if __name__ == '__main__':
    unittest.main()
    
