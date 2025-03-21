import numpy as np

from dsa2000_cal.model_utils import average_model


def test_average_model():
    D,T,B,C = 3,4,5,6
    Tm = 2
    Cm = 2
    assert np.shape(average_model(np.ones((D,T,B,C)), Tm=Tm, Cm=Cm)) == (D,Tm,B,Cm)

    Tm = 1
    Cm = 1
    assert np.shape(average_model(np.ones((D, T, B, C)), Tm=Tm, Cm=Cm)) == (D, Tm, B, Cm)
