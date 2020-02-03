import matplotlib.pyplot as plt
#%matplotlib inline
import nengo
import numpy as np
import scipy.ndimage
import matplotlib.animation as animation
from matplotlib import pylab
from PIL import Image
import nengo.spa as spa
import cPickle

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask

#Encode categorical integer features using a one-hot aka one-of-K scheme.
def one_hot(labels, c=None):
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y


rng = np.random.RandomState(9)

# --- load the data
img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

train_targets = one_hot(y_train, 10)
test_targets = one_hot(y_test, 10)


# --- set up network parameters
#Want to encode and decode the image
n_vis = X_train.shape[1]
n_out =  X_train.shape[1]
#number of neurons/dimensions of semantic pointer
n_hid = 5000 #Try with more neurons for more accuracy
#n_hid = 1000

#Want the encoding/decoding done on the training images
ens_params = dict(
    eval_points=X_train,
    neuron_type=nengo.LIFRate(), #Why not use LIF?
    intercepts=nengo.dists.Choice([-0.5]),
    max_rates=nengo.dists.Choice([100]),
    )

#Least-squares solver with L2 regularization.
solver = nengo.solvers.LstsqL2(reg=0.01)
#solver = nengo.solvers.LstsqL2(reg=0.0001)
solver2 = nengo.solvers.LstsqL2(reg=0.01)

#network that 
with nengo.Network(seed=3) as model:
    a = nengo.Ensemble(n_hid, n_vis, seed=3, **ens_params)
    v = nengo.Node(size_in=n_out)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train, function=X_train,#want the same thing out
        solver=solver)
    
    v2 = nengo.Node(size_in=train_targets.shape[1])
    conn2 = nengo.Connection(
        a, v2, synapse=None,
        eval_points=X_train, function=train_targets, #Want to get the labels out
        solver=solver2)
    
    

def get_outs(sim, images):
    _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)
    return np.dot(acts, sim.data[conn2].weights.T)

def get_error(sim, images, labels):
    return np.argmax(get_outs(sim, images), axis=1) != labels

def get_labels(sim,images):
    return np.argmax(get_outs(sim, images), axis=1)

#Get the neuron activity of an image or group of images (this is the semantic pointer in this case)
def get_activities(sim, images):
    _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)
    return acts

def get_encoder_outputs(sim,images):
    outs = np.dot(images,sim.data[a].encoders.T) #before the neurons Why transpose?
    return outs

'''
#Images to train for rotation of 90 deg
orig_imgs = X_train[:10000].copy()

rotated_imgs =X_train[:10000].copy()
for img in rotated_imgs:
    img[:] = scipy.ndimage.interpolation.rotate(np.reshape(img,(28,28)),90,reshape=False).ravel()


test_imgs = X_test[:1000].copy()
'''  

#Images to train, starting at random orientation
orig_imgs = X_train[:100000].copy()
for img in orig_imgs:
    img[:] = scipy.ndimage.interpolation.rotate(np.reshape(img,(28,28)),(np.random.randint(360)),reshape=False,mode="nearest").ravel()

#Images rotated a fixed amount from the original random orientation
rotated_imgs =orig_imgs.copy()
for img in rotated_imgs:
    img[:] = scipy.ndimage.interpolation.rotate(np.reshape(img,(28,28)),6,reshape=False,mode="nearest").ravel()

    #^encoder outputs
    
#Images not used for training, but for testing (all at random orientations)
test_imgs = X_test[:1000].copy()
for img in test_imgs:
    img[:] = scipy.ndimage.interpolation.rotate(np.reshape(img,(28,28)),(np.random.randint(360)),reshape=False,mode="nearest").ravel()


#Check that rotation is done correctly
#plt.subplot(121)
#plt.imshow(orig_imgs[5].reshape(28,28),cmap='gray')
#plt.subplot(122)
#plt.imshow(rotated_imgs[5].reshape(28,28),cmap='gray')

# linear filter used for edge detection as encoders, more plausible for human visual system
encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)
#Set the ensembles encoders to this
a.encoders = encoders

#Check the encoders were correctly made
plt.imshow(encoders[0].reshape(28, 28), vmin=encoders[0].min(), vmax=encoders[0].max(), cmap='gray')


with nengo.Simulator(model) as sim:    
    
    #Neuron activities of different mnist images
    #The semantic pointers
    orig_acts = get_activities(sim,orig_imgs)
    #rotated_acts = get_activities(sim,rotated_imgs)
    #test_acts = get_activities(sim,test_imgs)
    
    #X_test_acts = get_activities(sim,X_test)
    #labels_out = get_outs(sim,X_test)
    
    rotated_after_encoders = get_encoder_outputs(sim,rotated_imgs)
    
    #solvers for a learning rule
    #solver_tranform = nengo.solvers.LstsqL2(reg=1e-8)
    #solver_word = nengo.solvers.LstsqL2(reg=1e-8)
    solver_rotate_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    
    
    #find weight matrix between neuron activity of the original image and the rotated image
    #weights returns a tuple including information about learning process, just want the weight matrix
    #weights,_ = solver_tranform(orig_acts, rotated_acts)
    
    #find weight matrix between labels and neuron activity
    #label_weights,_ = solver_word(labels_out,X_test_acts)
    
    
    rotated_after_encoder_weights,_ = solver_rotate_encoder(orig_acts,rotated_after_encoders)
    
    
cPickle.dump(rotated_after_encoder_weights, open( "rotated_after_encoder_weights.p", "wb" ) )