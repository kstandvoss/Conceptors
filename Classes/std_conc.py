# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 17:08:43 2015

@author: User
"""

from reservoir_c import *
import pattCat
import time
from matplotlib.pyplot import *


t0 = time.clock()

patterns = []

for p in [53, 54, 10, 36]:
    patterns.append(pattCat.patts[p])

A = Reservoir()
A.load(patterns, gradient_c=False)
A.recall()

functions.plot_interpolate_1d(patterns, A.Y_recalls)


print time.clock()-t0