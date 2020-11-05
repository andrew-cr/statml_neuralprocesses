import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

class Generator():

    def __init__(self, K, x=None, parameters=None):
        self.K = K
        if parameters is None:
            self.parameters = self._sample_parameters()
        else:
            self.parameters = parameters
        if x is None:
            self.x = self._sample_x()
        else:
            self.x = x

    def _sample_parameters(self):
        raise NotImplementedError

    def _sample_x(self):
        raise NotImplementedError

    def equally_spaced_samples(self, K=None):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def batch(self, x=None, force_new=False):
        '''Returns a batch of size K.

        It also changes the sape of `x` to add a batch dimension to it.

        Args:
            x: Batch data, if given `y` is generated based on this data.
                Usually it is None. If None `self.x` is used.
            force_new: Instead of using `x` argument the batch data is
                uniformly sampled.

        '''
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]


class Dataset():

    def __init__(self, K=20, size=None, generator_class=Generator, parameters_x_list=None):
        self.generator_class = generator_class
        self.K = K
        self.size = size

        if parameters_x_list is not None:
            if isinstance(parameters_x_list, str):
                with open(parameters_x_list, 'rb') as f:
                    parameters_x_list = pickle.load(f)
            if self.size is not None:
                assert len(parameters_x_list) == self.size
            else:
                self.size = len(parameters_x_list)
            self.generators = [self.generator_class(K=K, x=parameters_x['x'], parameters=parameters_x['parameters']) for
                               parameters_x in parameters_x_list]

        else:
            assert self.size
            self.generators = [self.generator_class(K=K) for _ in range(self.size)]

    def save(self, file_name, parameters=True, x=True):

        parameters_x_list = []

        for generator in self.generators:
            parameters_generator = generator.parameters if parameters else None
            x_generator = generator.x if x else None
            parameters_x_list.append({'parameters': parameters_generator, 'x': x_generator})

        with open(file_name, 'wb') as f:
            pickle.dump(parameters_x_list, f)

    def __len__(self):
        return len(self.generators)

    def sample(self):
        return random.sample(self.generators, len(self))

    def __getitem__(self, index):
        return self.generators[index]

class SinusoidGenerator(Generator):

    def _sample_parameters(self):
        parameters = {}
        parameters['amplitude'] = np.random.uniform(0.1, 5.0)
        parameters['phase'] = np.random.uniform(0, np.pi)
        return parameters

    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)

    def equally_spaced_samples(self, K=None):
        '''Returns `K` equally spaced samples.'''
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))

    def f(self, x):
        '''Sinewave function.'''
        return self.parameters['amplitude'] * np.sin(x - self.parameters['phase'])

class SinusoidDataset(Dataset):
    def __init__(self, **kwargs):
        assert 'generator' not in kwargs
        generator_class = SinusoidGenerator
        super().__init__(generator_class=generator_class, **kwargs)
