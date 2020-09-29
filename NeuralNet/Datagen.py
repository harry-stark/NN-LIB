import numpy as np
class DataGenerator():
    def __init__(self, data, target, batch_size, shuffle=True):
        self.shuffle      = shuffle
        if shuffle:
            shuffled_indices = np.random.permutation(len(data))
        else:
            shuffled_indices = range(len(data))

        self.data         = data[shuffled_indices]
        self.target       = target[shuffled_indices]
        self.batch_size   = batch_size 
        self.num_batches  = int(np.ceil(data.shape[0]/batch_size))
        self.counter      = 0


    def __iter__(self):
        return self

    def __next__(self):
        if self.counter<self.num_batches:
            batch_data = self.data[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            batch_target = self.target[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            self.counter+=1
            return batch_data,batch_target
        else:
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(self.target))
            else:
                shuffled_indices = range(len(self.target))

            self.data         = self.data[shuffled_indices]
            self.target       = self.target[shuffled_indices]

            self.counter = 0
            raise StopIteration
