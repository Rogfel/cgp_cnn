from pycgp import CGP
from feature_extractions import cnn


def build_cnn():
    return [
        CGP.CGPFunc(cnn.conv2D,'conv2D',1,2),
        CGP.CGPFunc(cnn.maxPool2D,'maxPool2D',1,2),
        CGP.CGPFunc(cnn.avgPool2D,'avgPool2D',1,2),
        CGP.CGPFunc(cnn.concatenate,'concatenate',2,1),
        CGP.CGPFunc(cnn.summation,'summation',2,1),
    ]