import nengo
import numpy as np
import cPickle
from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
nengo.log('debug')

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

label_weights = cPickle.load(open("label_weights1000.p", "rb"))
activity_to_img_weights = cPickle.load(open("activity_to_img_weights1000.p", "rb"))
rotated_after_encoder_weights =  cPickle.load(open("rotated_after_encoder_weights1000.p", "r"))
#rotated_after_encoder_weights_5000 =  cPickle.load(open("rotated_after_encoder_weights_5000.p", "r"))
rotation_weights = cPickle.load(open("rotation_weights1000.p","rb"))

input_shape = (1,28,28)

def display_func(t, x, input_shape=input_shape):
    import base64
    import PIL
    import cStringIO
    from PIL import Image

    values = x.reshape(input_shape)
    values = values.transpose((1, 2, 0))
    vmin, vmax = values.min(), values.max()
    values = (values - vmin) / (vmax - vmin + 1e-8) * 255.
    values = values.astype('uint8')

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    png = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    display_func._nengo_html_ = '''
        <svg width="100%%" height="100%%" viewbox="0 0 100 100">
        <image width="100%%" height="100%%"
               xlink:href="data:image/png;base64,%s"
               style="image-rendering: pixelated;">
        </svg>''' % (''.join(img_str))

rng = np.random.RandomState(9)
n_hid = 1000


model = nengo.Network(seed=3)
with model:
    stim = nengo.Node(lambda t: ONE if t < 0.1 else 0) #nengo.processes.PresentInput(labels,1))#
    
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
    
    #ens2 = nengo.Ensemble(n_hid, dim**2, seed=3, encoders=encoders, **ens_params)
    
    #nengo.Connection(ens.neurons, ens.neurons, transform = rotated_after_encoder_weights, synapse=0.1)      
    #nengo.Connection(ens.neurons, ens2.neurons, transform = rotated_after_encoder_weights, synapse=0.1)      
    
    display_node = nengo.Node(display_func, size_in=784)#ens.size_out)
    
    #node = nengo.Node(None, size_in=10)
    
    nengo.Connection(stim, ens, transform = np.dot(label_weights,activity_to_img_weights).T, synapse=0.1)
    
    nengo.Connection(ens, display_node, synapse=0.1)
    
    nengo.Connection(ens.neurons, ens.neurons, transform = rotated_after_encoder_weights.T, synapse=0.2)      
    #nengo.Connection(ens.neurons, ens2.neurons, transform = rotated_after_encoder_weights.T, synapse=0.1)      
    

