import numpy as np

class Multi_Class_Noisifier:
    def __init__(self):
        self.name = 'Multi_Class_Noisifier'
    
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

        # Check if the input is multi-dimensional
        if y_batch.shape[0] <= 1:
            raise Exception('The input shape is wrong.')

    def _get_noisy_labels(self, y_batch, y_batch_noisy):
        return np.argwhere(y_batch != y_batch_noisy)

    def flip(self, y_batch, P, seed=0):

        y_batch_noisy = np.copy(y_batch)

        labels = y_batch_noisy.shape[0]
        flipper = np.random.RandomState(seed)

        for i in np.arange(labels):

            P_row_index = np.where(y_batch[i] == 1)[0]

            flipped = flipper.multinomial(1, P[P_row_index][0], 1)

            index = np.where(flipped == 1)[1]
            
            # Flip the values
            y_batch_noisy[i][P_row_index], y_batch_noisy[i][index] = y_batch_noisy[i][index], y_batch_noisy[i][P_row_index]

        return y_batch_noisy, self._get_noisy_labels(y_batch, y_batch_noisy)

    def symmetry_flip(self, y_batch, noise_rate, seed=0):
        '''
        Creates Symmetry flipping matrix and flips the labels accordingly
        Entry (i,j) = Probability of flipping i to j
        A 5-class example with noise_rate = 0.5:
        [[0.5 0.125 0.125 0.125 0.125]
            [0.125 0.5 0.125 0.125 0.125]
            [0.125 0.125 0.5 0.125 0.125]
            [0.125 0.125 0.125 0.5 0.125]
            [0.125 0.125 0.125 0.125 0.5]]
        '''

        self.__check_input(y_batch)
        
        num_samples, num_classes = y_batch.shape
        
        P = np.ones((num_classes, num_classes))
        P = (noise_rate / (num_classes - 1)) * P

        if noise_rate > 0.0:
            P[0, 0] = 1. - noise_rate
            for i in range(1, num_classes-1):
                P[i, i] = 1. - noise_rate
            P[num_classes-1, num_classes-1] = 1. - noise_rate

        y_batch, noisy_labels = self.flip(y_batch, P, seed)

        return y_batch, noisy_labels

    def pair_flip(self, y_batch, noise_rate, seed=0):
        '''
        Creates Pair flipping matrix and flips the labels accordingly
        Entry (i,j) = Probability of flipping i to j
        A 5-class example with noise_rate = 0.45:
        [[0.55 0.45 0. 0. 0.]
            [0. 0.55 0.45 0. 0.]
            [0. 0. 0.55 0.45 0.]
            [0. 0. 0. 0.55 0.45]
            [0.45 0. 0. 0. 0.55]]
        '''
        
        self.__check_input(y_batch)
        
        num_samples, num_classes = y_batch.shape
        
        P = np.eye(num_classes)

        if noise_rate > 0.0:
            P[0, 0], P[0, 1] = 1. - noise_rate, noise_rate
            for i in range(1, num_classes-1):
                P[i, i], P[i, i + 1] = 1. - noise_rate, noise_rate
            P[num_classes-1, num_classes-1], P[num_classes-1, 0] = 1. - noise_rate, noise_rate

        y_batch, noisy_labels = self.flip(y_batch, P, seed)
            
        return y_batch, noisy_labels
