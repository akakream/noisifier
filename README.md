# noisifier

Add label noise to your dataset

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

Noisify your data

```
noised_y_train = noisifier.noisify(y_train, noise_type, noise_rate, NUM_OF_CLASSES, random_state_seed)
```

### Caveats

Provide y\_train in one-hot encoded form. If you use keras, you can do that by 

```
y_train = keras.utils.to_categorical(y_train, 10)
```

For the noise\_type, use symmetry or pair.

For the noise\_rate, use a float between 0 and 1. Note that labels with more than 0.5 noise rate cannot be learned without additional assumptions.

## Licence

noisify is released under the MIT licence.

## Version 

0.1
