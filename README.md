# noisifier

Add label noise to your dataset

This work is inspired by the noise matrices for single label data that are introduced, at least to me, in [https://github.com/bhanML/Co-teaching](https://github.com/bhanML/Co-teaching).
In order to easily noisify my single label data, I created a python package that applies these noise matrices.
I also extend this package by defining and applying label noise to multi label data. 

## Prerequisites

python 3.6.9 

numpy 1.18.3

## Installation

```
pip install noisifier
```

### Link to the project 

https://pypi.org/project/noisifier/

## How to use

Create a noisifier instance

```
noisifier = Noisifier()
```

Noisify your data. Return is same type and same shape as y\_train.

### Single label noise

```
noised_y_train = noisifier.noisify(y_train, noise_type, noise_rate, NUM_OF_CLASSES, random_state_seed=0)
```

For the noise\_type, use symmetry or pair.

### Multi label noise

Selects random samples and random classes of the chosen samples by the given rates.

```
noised_y_train = noisifier.random_multi_label_noise(y_train, sample_rate, class_rate)
```

Selects random classes that are labeled with 0 by the given rate and flips them to 1.

```
noised_y_train = noisifier.add_missing_noise(y_train, rate)
```

Selects random classes that are labeled with 1 by the given rate and flips them to 0.

```
noised_y_train = noisifier.add_extra_noise(y_train, rate)
```

Selects random classes by the given rate and flips them to their opposites.

```
noised_y_train = noisifier.add_mix_noise(y_train, rate)
```

Selects random labels for each class in the data by the given rate and flips them.

```
noised_y_train = noisifier.add_classwise_noise(y_train, rate)
```

### Caveats

Provide y\_train in one-hot encoded form. If you use keras, you can do that by 

```
y_train = keras.utils.to_categorical(y_train, 10)
```

Use a float between 0 and 1 for the noise\_rate, sample\_rate, class\_rate and rate. Note that labels with more than 0.5 noise rate cannot be learned without additional assumptions.

## Licence

noisify is released under the MIT licence.

## Version 

0.4
