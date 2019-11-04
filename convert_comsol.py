# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:48:28 2018

@author: User
"""

from comsol_parser import comsol_to_python
import numpy as np

#fn = "Results/1d-2.txt"
#comsol_to_python(fn, 'k-device_gamma-1p5e-10_1d')
#
#fn = "Results/2d-2.txt"
#comsol_to_python(fn, 'k-device_gamma-1p5e-10_2d')
#
#fn = "Results/3d-2.txt"
#comsol_to_python(fn, 'k-device_gamma-1p5e-10_3d')


#for name in ['-1e-9', '-1e-10', '-1e-11']:
#    fn = f"Results/gamma_{name}.txt"
#    comsol_to_python(fn, f'k-device_gamma{name}')
    
#for name in ['C', 'logC']:
#    fn = f"Results/{name}.txt"
#    comsol_to_python(fn, f'k-device_gamma-1.5e-10_{name}')

#for name in ['10um', '20um', '50um', '100um']:
#    fn = f"Results/ramp_{name}.txt"
#    comsol_to_python(fn, f'k-device_gamma-1.5e-10_ramp_{name}')

#for i, name in enumerate(['0p1uM', '1uM', '10uM', '100uM', '1mM', '10mM']):
#    fn = f"Results/Cs_{i+1}.txt"
#    comsol_to_python(fn, f'k-device_gamma-1p5e-10_CsOut_{name}')
    
#for i, name in enumerate(np.fromstring('-2e-11 -5e-11 -1e-10 -2e-10 -5e-10', sep=' ')):
#    fn = f"Results/gamma_{i+1}.txt"
#    comsol_to_python(fn, f'k-device_gamma{name}')
    
#for i, name in enumerate(np.fromstring('1e-10 2e-10 5e-10 1e-9 2e-9 5e-9 1e-8', sep=' ')):
#    fn = f"Results/Cs_{i+1}.txt"
#    comsol_to_python(fn, f'k-device_Ds_{name}')
    
for i, name in enumerate(np.fromstring('5e-12 1e-11 2e-11 5e-11 1e-10 2e-10 5e-10', sep=' ')):
    fn = f"Results/Dp_{i+1}.txt"
    comsol_to_python(fn, f'k-device_Dp_{name}')


