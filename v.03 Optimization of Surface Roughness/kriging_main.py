from Classes import *
import numpy as np

###Options###
#colum_defs
xi_col=0
yi_col=1
zi_col=4
#resolution
resolution=25
#kriging config
Params=[]
Variogram='gaussian'
#plotting
saveplot=True
displayplot=True
#############

loadeddata=np.loadtxt('./data/surface_roughness.csv',delimiter=',',skiprows=1)
points=loadeddata[:,[xi_col,yi_col]]
zi=loadeddata[:,zi_col]

krige=OrdinaryKrigning(points,zi)