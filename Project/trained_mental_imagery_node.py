import nengo
import numpy as np
import cPickle


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

label_weights = cPickle.load(open("label_weights.p", "rb"))
activity_to_img_weights = cPickle.load(open("activity_to_img_weights.p", "rb"))

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

model = nengo.Network()
with model:
    stim = nengo.Node(nengo.processes.PresentInput(labels,1))

    activity_node = nengo.Node(None,size_in=1000)
    
    nengo.Connection(stim, activity_node, transform = label_weights.T)
    
    img_node = nengo.Node(None,size_in=dim**2)

    nengo.Connection(activity_node, img_node, transform = activity_to_img_weights.T)

    display_node = nengo.Node(display_func, size_in=img_node.size_out)
    nengo.Connection(img_node, display_node, synapse=None)
      
    