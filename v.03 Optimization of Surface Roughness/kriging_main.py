from Classes import *
import numpy as np
from matplotlib import pyplot as plt


###Options###
#colum_defs
xi_col=0
yi_col=1
zi_col=4
#resolution
resolution=10
#kriging config
Params=[]

Variogram='exponential'
trendfunc='quadratic'
    #plotting
saveplot=True


loadeddata=np.loadtxt('data\surface_roughness.csv',delimiter=',',skiprows=1)
points=loadeddata[:,[xi_col,yi_col]]
zi=loadeddata[:,zi_col]

krige=UniversalKriging(points,zi,Variogram=Variogram,trendfunc=trendfunc)
krige.AutoKrige(step=resolution)
krige.Plot(f'Top SA_{Variogram}_Variogram_UK r2={krige.LOOr2}',xtitle='Power',ytitle='Speed',saveplot=saveplot,address=f'figs\Top_SA_{Variogram}_{trendfunc}_UK.png',extent=[0, 900, 0, 5000])
print(krige.LOOr2)
print(krige.params)

