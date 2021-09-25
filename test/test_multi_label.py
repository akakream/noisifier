import unittest
import numpy as np
from noisifier.multi_label import Multi_Label_Noisifier

class TestMultiLabelNoisifier(unittest.TestCase):

    def testRandomNoisePerSample(self):
        noisifier = Multi_Label_Noisifier()
        sample_rate = 0.5
        class_rate = 0.5
        row_sum = 5
        check_value = 25.
        
        y = np.array([
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.],
            [1.,0.,0.,1.,0.,1.,0.,1.,1.,0.]
            ])

        y_noisy, noisy_samples = noisifier.random_noise_per_sample(y, sample_rate, class_rate, True)

        condition = np.sum(np.absolute(y_noisy-y), axis=1)
        condition = np.sum(np.where(condition == row_sum, condition, 0.))

        self.assertEqual(condition, check_value, "Should be 25")

if __name__ == '__main__':
    unittest.main()
