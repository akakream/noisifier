import numpy as np

class Multi_Label_Noisifier:
    def __init__(self):
        self.name = 'Multi_Label_Noisifier'
    
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

    def _get_noisy_labels(self, y_batch, y_batch_noisy):
        return np.argwhere(y_batch != y_batch_noisy)

    def random_noise_per_sample(self, y_batch, sample_rate, class_rate, seed=False):
        '''
        y_batch = shape must be (batch_size,classes)
        sample_rate = between 0 and 1.0 (float). The percentage of samples that will be noisified in the mini batch.
        class_rate = between 0 and 1.0 (float). The percentage of classes that will be noisified for the sample.
        '''
        if seed == True:
            np.random.seed(0)
        
        num_samples, num_classes = y_batch.shape

        num_noisy_samples = round(num_samples * sample_rate)
        num_noisy_classes = round(num_classes * class_rate)

        random_samples = np.random.choice(range(0,num_samples),num_noisy_samples, replace=False)
        random_samples = np.expand_dims(random_samples, axis=0)
        random_classes = np.random.choice(range(0,num_classes),num_noisy_classes, replace=False)
        random_classes = np.expand_dims(random_classes, axis=1)

        y_batch_noisy = np.copy(y_batch)

        y_batch_noisy[random_samples, random_classes] = -y_batch_noisy[random_samples, random_classes] + 1

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    def mix_label_noise(self, y_batch, rate, seed=False):
        '''
        This function is for multi label data,
        and flips (rate) percentage of positive labels in the batch to 
        negative labels, and flips (rate) percentage of negative labels in
        the batch to positive labels.
        Let y_batch be in the following shape: (batch_size, classes).
        Give a rate in float between 0. and 1.
        (Rate * number of zeros) zeros will be flipped. 
        (Rate * number of ones) ones will be flipped. 
        '''
        if seed == True:
            np.random.seed(0)

        num_samples, num_classes = y_batch.shape
        
        y_batch_noisy = np.copy(y_batch)
        
        y_batch_noisy = y_batch_noisy.flatten()

        # Get the indices of zeros and map them to a 1-d array
        zero_indices = np.where(y_batch_noisy == 0)[0]
        # Get the indices of ones and map them to a 1-d array
        one_indices = np.where(y_batch_noisy == 1)[0]

        # Calculate the number of zeros to get flipped
        chosen_size_zeros = int(len(zero_indices) * rate)
        # Choose the indices of zeros to be flipped
        chosen_zeros = np.random.choice(zero_indices, chosen_size_zeros, replace=False)
        # Calculate the number of ones to get flipped
        chosen_size_ones = int(len(one_indices) * rate)
        # Choose the indices of ones to be flipped
        chosen_ones = np.random.choice(one_indices, chosen_size_ones, replace=False)
        
        # Flip the zeros
        y_batch_noisy[chosen_zeros] = 1
        # Flip the ones
        y_batch_noisy[chosen_ones] = 0
        # Reshape the batch
        y_batch_noisy = np.reshape(y_batch_noisy, (num_samples, num_classes))

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    def add_missing_noise(self, y_batch, rate, seed=False):
        '''
        This function is for multi label data,
        and flips positive labels into negative labels for the given rate.
        Let y_batch be in the following shape: (batch_size, classes).
        Give a rate in float between 0. and 1.
        (Rate * number of zeros) zeros will be flipped. 
        '''
        if seed == True:
            np.random.seed(0)

        num_samples, num_classes = y_batch.shape
        
        y_batch_noisy = np.copy(y_batch)
        
        y_batch_noisy = y_batch_noisy.flatten()

        # Get the indices of zeros and map them to a 1-d array
        zero_indices = np.where(y_batch_noisy == 0)[0]

        # Calculate the number of zeros to get flipped
        chosen_size = int(len(zero_indices) * rate)
        # Choose the indices of zeros to be flipped
        chosen_ones = np.random.choice(zero_indices, chosen_size, replace=False)

        # Flip labels
        y_batch_noisy[chosen_ones] = 1
        y_batch_noisy = np.reshape(y_batch_noisy, (num_samples, num_classes))

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    def add_extra_noise(self, y_batch, rate, seed=False):
        '''
        This function is for multi label data,
        and flips negative labels into positive labels for the given rate.
        Let y_batch be in the following shape: (batch_size, classes).
        Give a rate in float between 0. and 1.
        (Rate * number of ones) ones will be flipped. 
        '''
        if seed == True:
            np.random.seed(0)

        num_samples, num_classes = y_batch.shape
        
        y_batch_noisy = np.copy(y_batch)
        
        y_batch = y_batch_noisy.flatten()

        # Get the indices of ones and map them to a 1-d array
        one_indices = np.where(y_batch_noisy == 1)[0]

        # Calculate the number of ones to get flipped
        chosen_size = int(len(one_indices) * rate)
        # Choose the indices of ones to be flipped
        chosen_ones = np.random.choice(one_indices, chosen_size, replace=False)

        # Flip labels
        y_batch_noisy[chosen_ones] = 0
        y_batch_noisy = np.reshape(y_batch_noisy, (num_samples, num_classes))

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    def add_mix_noise(self, y_batch, rate, seed=False):
        '''
        This function is for multi label data,
        and flips labels into their opposites for the given rate.
        Let y_batch be in the following shape: (batch_size, classes).
        Give a rate in float between 0. and 1.
        '''
        if seed == True:
            np.random.seed(0)

        num_samples, num_classes = y_batch.shape
        
        y_batch_noisy = np.copy(y_batch)
        
        y_batch_noisy = y_batch_noisy.flatten()

        # Get the indices and map them to a 1-d array
        indices = num_samples * num_classes

        # Calculate the number of classes to get flipped
        chosen_size = int(indices * rate)
        # Choose the indices of classes to be flipped
        chosen_ones = np.random.choice(indices, chosen_size, replace=False)

        # Flip labels
        y_batch_noisy[chosen_ones] = -y_batch_noisy[chosen_ones]+1
        y_batch_noisy = np.reshape(y_batch_noisy, (num_samples, num_classes))

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    def add_classwise_noise(self, y_batch, rate, seed=False):
        '''
        This function adds class-wise label noise to dataset.
        According to the rate that is given, it chooses the same percentage
        of labels for each class and noisifies them.
        Let y_batch be in the following shape: (batch_size, classes).
        Give a rate in float between 0. and 1.
        '''
        if seed == True:
            np.random.seed(0)

        num_samples, num_classes = y_batch.shape
        
        y_batch_noisy = np.copy(y_batch)
        
        chosen_size = int(num_samples * rate)

        for cl in range(num_classes):
            random_samples = np.random.choice(range(0,num_samples),chosen_size, replace=False)
            y_batch_noisy[random_samples,cl] = -y_batch_noisy[random_samples,cl] + 1

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)
