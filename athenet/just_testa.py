from athenet.algorithm.derest.network import get_derest_layer
from athenet.algorithm.derest.utils import change_order, add_tuples
from athenet.algorithm.numlike import Nplike, Interval
import theano
import theano.tensor as T

import numpy as np

def eval_tensor_on_layer(layer, tensor):
    layer.input = T.tensor4('tmp')
    evaluate = theano.function(
        [layer.input_layer.input],
        layer.output
    )
    return evaluate(tensor)


def eval_tensor_on_layers(layer1, layer2, tensor):
    layer1.input = T.tensor4()
    layer2.input_layer = layer1
    evaluate = theano.function(
        [layer1.input],
        layer2.output
    )
    return evaluate(tensor)


def testa():
    n = 3
    
    dummy_layer = ConvolutionalLayer((1,1,1),(n,n,1))
    input = np.ones((1, 1, n, n), dtype=theano.config.floatX)
    dummy_layer.input = input
    layer = InceptionLayer([1,2,3,4,5,6])
    layer.input_layer = dummy_layer

    print "Layer exists"
    eval_tensor_on_layers(dummy_layer, layer, input)

    print "Layer evaluated"
    print layer.output_shape
    print layer.input_shape
    derest_layer = get_derest_layer(layer)
    print "Derest layer exist"
    derest_layer.normalize = True
    act = Interval(np.zeros((1,n,n)), np.ones((1,n,n)))

    a = T.tensor3(name = "lollower", dtype=theano.config.floatX)
    b = T.tensor3(name="lolupper", dtype=theano.config.floatX)
    act_in_theory = Interval(a,b)

    derest_layer.activations = act_in_theory
    out_in_theory = derest_layer.count_activation(act_in_theory)
    print "run!"
    res = [r.eval({b: act.upper, a:act.lower})[0] for r in out_in_theory]
    print res
    shapes = [r.shape for r in res]
    print shapes

    da = T.tensor4(name="ochlower", dtype=theano.config.floatX)
    db = T.tensor4(name="ochupper", dtype=theano.config.floatX)
    print "and now test derivatives..."

    batches = 1
    input_shape = add_tuples(batches,
                             change_order(derest_layer.layer.input_shape))
    output_shape = add_tuples(batches,
                              change_order(derest_layer.layer.output_shape))
    der_in_theory = Interval(da,db)

    derest_layer.derivatives = der_in_theory
    derivatives_in_theory = derest_layer.count_derivatives(der_in_theory, input_shape)
    der_low = np.ones(output_shape)
    der_up = 2 * np.ones(output_shape)
    print output_shape

    res = derivatives_in_theory.eval({da:der_low, db:der_up, a:act.lower, b:act.upper})[0]
    print res
    print res.shape



def test_numlike():
    a = np.ones((3,4,5))
    b = 2 * a
    na = Nplike(a)
    nb = Nplike(b)
    print na.concat(nb).eval()
    ia = Interval(a, a * 3)
    ib = Interval(b, b * 3)

    print ia

    print ia.concat(ib).eval()



testa()
