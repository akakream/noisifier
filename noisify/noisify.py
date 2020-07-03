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

    def noisify(self, y_train, noise_type, noise_rate, num_classes, random_state_seed):
        if noise_type == 'symmetry':
            self.symmetry_flipping(y_train, noise_rate, num_classes, random_state_seed)
        elif noise_type == 'pair':
            self.pair_flipping(y_train, noise_rate, num_classes, random_state_seed)

