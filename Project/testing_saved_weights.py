import cPickle
import numpy as np
import nengo
from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
from matplotlib import pylab
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- load the data
img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

temp = np.diag([1]*10)

ZERO = temp[0]
ONE =  temp[1]
TWO =  temp[2]
THREE= temp[3]
FOUR = temp[4]
FIVE = temp[5]
SIX =  temp[6]
SEVEN =temp[7]
EIGHT= temp[8]
NINE = temp[9]

labels =[ZERO,ONE,TWO,THREE,FOUR,FIVE,SIX,SEVEN,EIGHT,NINE]

dim =28
n_hid = 1000
rng = np.random.RandomState(9)

label_weights = cPickle.load(open("label_weights.p", "rb"))
activity_to_img_weights = cPickle.load(open("activity_to_img_weights.p", "rb"))
rotated_after_encoder_weights =  cPickle.load(open("rotated_after_encoder_weights.p", "r"))

model = nengo.Network(seed=3)
with model:
    ens_params = dict(
        eval_points=X_train,
        neuron_type=nengo.LIFRate(), #Why not use LIF?
        intercepts=nengo.dists.Choice([-0.5]),
        max_rates=nengo.dists.Choice([100]),
        )
        
    # linear filter used for edge detection as encoders, more plausible for human visual system
    encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
    encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)

    ens = nengo.Ensemble(n_hid, dim**2, seed=3, encoders=encoders, **ens_params)
    ens2 = nengo.Ensemble(n_hid, dim**2, seed=3, encoders=encoders, **ens_params)
    
sim = nengo.Simulator(model)

plt.subplot(121)
pylab.imshow(np.reshape(X_train[0],(dim, dim), 'F').T, cmap=plt.get_cmap('Greys_r'))

#Get activity of image
_, testing_act = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=X_train[0])

testing_rotate = np.dot(testing_act,activity_to_img_weights)

plt.subplot(122)
pylab.imshow(np.reshape(testing_rotate,(dim, dim), 'F').T, cmap=plt.get_cmap('Greys_r'))

plt.show()

