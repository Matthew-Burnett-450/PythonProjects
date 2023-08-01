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
Variogram='power'
#plotting
saveplot=True


loadeddata=np.loadtxt('v.03 Optimization of Surface Roughness\data\surface_roughness.csv',delimiter=',',skiprows=1)
points=loadeddata[:,[xi_col,yi_col]]
zi=loadeddata[:,zi_col]

krige=UniversalKriging(points,zi,Variogram=Variogram)
krige.AutoKrige(step=resolution)
krige.Plot(f'Top SA_{Variogram}_Variogram_UK r2={krige.LOOr2}',xtitle='x',ytitle='y',saveplot=saveplot,address=f'v.03 Optimization of Surface Roughness\\figs\\Top_SA_{Variogram}_UK.png',extent=[0, 800, 0, 4000])
print(krige.LOOr2)
print(krige.params)
