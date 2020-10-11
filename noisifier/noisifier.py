import numpy as np

class Noisifier:
    
    def __init__(self):
        self.name = 'noisifier'

    def __repr__(self):
        return f'Noisifier({self.name})'

    def __str__(self):
        return self.name
    
    def wrong_flip(self, y_train, P, random_state_seed=0):

        y_train_copy = np.copy(y_train)

        labels = y_train_copy.shape[0]
        flipper = np.random.RandomState(random_state_seed)

        # Reshape and normalize (sum=1) to draw a sample from multinomial distribution
        reshaped_P = np.reshape(P, P.shape[0]*P.shape[1])
        reshaped_P /= np.sum(reshaped_P)

        for i in np.arange(labels):

            flipped = flipper.multinomial(1, reshaped_P, 1)

            index = np.where(flipped == 1)

            row_index = int(index[1]) // P.shape[0]
            column_index = int(index[1]) % P.shape[1]
            
            # Flip the values
            y_train_copy[i][row_index], y_train_copy[i][column_index] = y_train_copy[i][column_index], y_train_copy[i][row_index]

        return y_train_copy

    def flip(self, y_train, P, random_state_seed=0):

        y_train_copy = np.copy(y_train)

        labels = y_train_copy.shape[0]
        flipper = np.random.RandomState(random_state_seed)

        for i in np.arange(labels):

            P_row_index = np.where(y_train[i] == 1)[0]

            flipped = flipper.multinomial(1, P[P_row_index][0], 1)

            index = np.where(flipped == 1)[1]
           
            # Flip the values
            y_train_copy[i][P_row_index], y_train_copy[i][index] = y_train_copy[i][index], y_train_copy[i][P_row_index]

        return y_train_copy

    def symmetry_flipping(self, y_train, noise_rate, num_classes, random_state_seed):
        '''
        Creates Symmetry flipping matrix and flips the labels accordingly
        A 5-class example with noise_rate = 0.5:
        [[0.5 0.125 0.125 0.125 0.125]
         [0.125 0.5 0.125 0.125 0.125]
         [0.125 0.125 0.5 0.125 0.125]
         [0.125 0.125 0.125 0.5 0.125]
         [0.125 0.125 0.125 0.125 0.5]]
        '''

        P = np.ones((num_classes, num_classes))
        P = (noise_rate / (num_classes - 1)) * P

        if noise_rate > 0.0:
            P[0, 0] = 1. - noise_rate
            for i in range(1, num_classes-1):
                P[i, i] = 1. - noise_rate
            P[num_classes-1, num_classes-1] = 1. - noise_rate

        y_train_noisified = self.flip(y_train, P, random_state_seed)

        noise = (y_train_noisified != y_train).mean()
        #print('Noise %.2f' % noise)

        return y_train_noisified

    def pair_flipping(self, y_train, noise_rate, num_classes, random_state_seed):
        '''
        Creates Pair flipping matrix and flips the labels accordingly
        A 5-class example with noise_rate = 0.45:
        [[0.55 0.45 0. 0. 0.]
         [0. 0.55 0.45 0. 0.]
         [0. 0. 0.55 0.45 0.]
         [0. 0. 0. 0.55 0.45]
         [0.45 0. 0. 0. 0.55]]
        '''
        
        P = np.eye(num_classes)

        if noise_rate > 0.0:
            P[0, 0], P[0, 1] = 1. - noise_rate, noise_rate
            for i in range(1, num_classes-1):
                P[i, i], P[i, i + 1] = 1. - noise_rate, noise_rate
            P[num_classes-1, num_classes-1], P[num_classes-1, 0] = 1. - noise_rate, noise_rate

        y_train_noisified = self.flip(y_train, P, random_state_seed)
            
        noise = (y_train_noisified != y_train).mean()
        #print('Noise %.2f' % noise)

        return y_train_noisified

    def noisify(self, y_train, noise_type, noise_rate, num_classes, random_state_seed=0):

        if noise_type == 'symmetry':
            noisy_y = self.symmetry_flipping(y_train, noise_rate, num_classes, random_state_seed)
        elif noise_type == 'pair':
            noisy_y = self.pair_flipping(y_train, noise_rate, num_classes, random_state_seed)
        else:
            raise ValueError('noise_type is not valid')
            
        return noisy_y

    def random_multi_label_noise(self, y_batch, sample_rate, class_rate, seed=False):
        '''
        y_batch = sahpe must be (batch_size,classes)
        sample_rate = between 0 and 1.0 (float). The percentage of samples that will be noisified in the mini batch.
        class_rate = between 0 and 1.0 (float). The percentage of classes that will be noisified for the sample.
        '''
        if seed == True:
            np.random.seed(0)
        
        num_samples = y_batch.shape[0]
        num_classes = y_batch.shape[1]
        y_batch = y_batch.numpy()

        num_noisy_samples = round(num_samples * sample_rate)
        num_noisy_classes = round(num_classes * class_rate)

        random_samples = np.random.choice(range(0,num_samples),num_noisy_samples, replace=False)
        random_samples = np.expand_dims(random_samples, axis=0)
        random_classes = np.random.choice(range(0,num_classes),num_noisy_classes, replace=False)
        random_classes = np.expand_dims(random_classes, axis=1)

        y_batch[random_samples, random_classes] = -y_batch[random_samples, random_classes] + 1

        return y_batch

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

        num_samples = y_batch.shape[0]
        num_classes = y_batch.shape[1]
        y_batch = y_batch.numpy()
        y_batch = y_batch.flatten()

        # Get the indices of zeros and map them to a 1-d array
        zero_indices = np.where(y_batch == 0)[0]

        # Calculate the number of zeros to get flipped
        chosen_size = int(len(zero_indices) * rate)
        # Choose the indices of zeros to be flipped
        chosen_ones = np.random.choice(zero_indices, chosen_size, replace=False)

        # Flip labels
        y_batch[chosen_ones] = 1
        y_batch = np.reshape(y_batch, (num_samples, num_classes))

        return y_batch

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

        num_samples = y_batch.shape[0]
        num_classes = y_batch.shape[1]
        y_batch = y_batch.numpy()
        y_batch = y_batch.flatten()

        # Get the indices of ones and map them to a 1-d array
        one_indices = np.where(y_batch == 1)[0]

        # Calculate the number of ones to get flipped
        chosen_size = int(len(one_indices) * rate)
        # Choose the indices of ones to be flipped
        chosen_ones = np.random.choice(one_indices, chosen_size, replace=False)

        # Flip labels
        y_batch[chosen_ones] = 0
        y_batch = np.reshape(y_batch, (num_samples, num_classes))

        return y_batch

    def add_mix_noise(self, y_batch, rate, seed=False):
        '''
        This function is for multi label data,
        and flips labels into their opposites for the given rate.
        Let y_batch be in the following shape: (batch_size, classes).
        Give a rate in float between 0. and 1.
        '''
        if seed == True:
            np.random.seed(0)

        num_samples = y_batch.shape[0]
        num_classes = y_batch.shape[1]
        y_batch = y_batch.numpy()
        y_batch = y_batch.flatten()

        # Get the indices and map them to a 1-d array
        indices = num_samples * num_classes

        # Calculate the number of classes to get flipped
        chosen_size = int(indices * rate)
        # Choose the indices of classes to be flipped
        chosen_ones = np.random.choice(indices, chosen_size, replace=False)

        # Flip labels
        y_batch[chosen_ones] = -y_batch[chosen_ones]+1
        y_batch = np.reshape(y_batch, (num_samples, num_classes))

        return y_batch

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

        num_samples = y_batch.shape[0]
        num_classes = y_batch.shape[1]
        y_batch = y_batch.numpy()
        
        chosen_size = int(num_samples * rate)

        for cl in range(num_classes):
            random_samples = np.random.choice(range(0,num_samples),chosen_size, replace=False)
            y_batch[random_samples,cl] = -y_batch[random_samples,cl] + 1

        return y_batch

