import numpy as np

class Binary_Class_Noisifier:
    def __init__(self):
        self.name = 'Binary_Class_Noisifier'
    
    def __repr__(self):
        return f'Noisifier({self.name})'

    def __str__(self):
        return self.name

    def __check_input(self, y_batch):
        # Check input type
        if not isinstance(y_batch, np.ndarray):
            raise Exception('The input for noisifier is not a numpy array.')
        
        # Check input dimensions
        if len(y_batch.shape) != 2:
            raise Exception('The input for noisifier must be a 2D numpy array.')

        # Check if the shape is right
        if y_batch.shape[1] != 1:
            raise Exception('The input shape is wrong. Make it (n,1)')
        
    def __get_label_info(self, y_batch):
        """
        Get y_batch information: Number of zeros and ones, and their indices
        """
        zero_indices = np.where(y_batch == 0.)[0]
        one_indices = np.where(y_batch == 1.)[0]
        num_of_zeros = len(zero_indices)
        num_of_ones = len(one_indices)
        
        return zero_indices, one_indices, num_of_zeros, num_of_ones
    
    def __choose_by_rate(self, zero_noise_rate, one_noise_rate, zero_indices, one_indices, num_of_zeros, num_of_ones):
        """
        Choose zero and one indices accordign to the given rate
        """
        num_zeros_chosen = int(num_of_zeros * zero_noise_rate)
        num_ones_chosen = int(num_of_ones * one_noise_rate)
        
        chosen_zeros = np.random.choice(zero_indices, num_zeros_chosen, replace=False)
        chosen_ones = np.random.choice(one_indices, num_ones_chosen, replace=False)
        
        return chosen_zeros, chosen_ones
    
    def _get_noisy_labels(self, y_batch, y_batch_noisy):
        return np.argwhere(y_batch != y_batch_noisy)
    
    def nullify_all(self, y_batch):
        """
        Labels every sample as 0
        """
        
        # Check the input
        self.__check_input(y_batch)
        
        y_batch_noisy = np.copy(y_batch)
        
        y_batch_noisy = 0. * y_batch_noisy
        
        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy) 
    
    
    def oneify_all(self, y_batch):
        """
        Labels every sample as 1
        """
        
        # Check the input
        self.__check_input(y_batch)
        
        y_batch_noisy = np.copy(y_batch)

        y_batch_noisy = 0. * y_batch_noisy + 1.
        
        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)
    
    
    def random_nullify(self, y_batch, noise_rate, seed=False):
        """
        Turns positive labels to negatives randomly by the given noise_rate
        """

        # Check the input
        self.__check_input(y_batch)

        # Set the seed if True for consistent randomization
        if seed == True:
            np.random.seed(0)
        
        y_batch_noisy = np.copy(y_batch)
        
        zero_indices, one_indices, num_of_zeros, num_of_ones = self.__get_label_info(y_batch_noisy)
        chosen_zeros, chosen_ones = self.__choose_by_rate(noise_rate, noise_rate, zero_indices, one_indices, num_of_zeros, num_of_ones)

        y_batch_noisy[chosen_ones] = 0.        
        
        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)        

        
    def random_oneify(self, y_batch, noise_rate, seed=False):
        """
        Turns negative labels to positives randomly by the given noise rate
        """
        
        # Check the input
        self.__check_input(y_batch)

        # Set the seed if True for consistent randomization
        if seed == True:
            np.random.seed(0)
        
        y_batch_noisy = np.copy(y_batch)
        
        zero_indices, one_indices, num_of_zeros, num_of_ones = self.__get_label_info(y_batch_noisy)
        chosen_zeros, chosen_ones = self.__choose_by_rate(noise_rate, noise_rate, zero_indices, one_indices, num_of_zeros, num_of_ones)

        y_batch_noisy[chosen_zeros] = 1.

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    
    def random_zoneify(self, y_batch, zero_noise_rate, one_noise_rate, seed=False):
        """
        Turns negative labels to positives and negatives to positives randomly by the given noise rates
        """
        
        # Check the input
        self.__check_input(y_batch)

        # Set the seed if True for consistent randomization
        if seed == True:
            np.random.seed(0)
        
        y_batch_noisy = np.copy(y_batch)

        zero_indices, one_indices, num_of_zeros, num_of_ones = self.__get_label_info(y_batch_noisy)
        chosen_zeros, chosen_ones = self.__choose_by_rate(zero_noise_rate, one_noise_rate, zero_indices, one_indices, num_of_zeros, num_of_ones)

        y_batch_noisy[chosen_zeros] = 1.
        y_batch_noisy[chosen_ones] = 0.        

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    
    def random_noise(self, y_batch, noise_rate, seed=False):
        """
        Noisify the input randomly by the given noise_rate
        """
        
        # Check the input
        self.__check_input(y_batch)

        # Set the seed if True for consistent randomization
        if seed == True:
            np.random.seed(0)
        
        y_batch_noisy = np.copy(y_batch)
        
        input_len = len(y_batch_noisy)
        num_to_flip = int(input_len * noise_rate)
        
        chosen_entries = np.random.choice(input_len, num_to_flip, replace=False)

        y_batch_noisy[chosen_entries] = -y_batch_noisy[chosen_entries] + 1

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

