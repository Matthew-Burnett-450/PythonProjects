from Ok_Uk_Module import *
from scipy.interpolate import Rbf

class SpacialSensitivityAnalysis(OrdinaryKrigning):
    def __init__(self, Points, Zvals, Variogram='gaussian',DiverganceModel='KLD'):
        super().__init__(Points, Zvals, Variogram)
        self.DivModel=DiverganceModel
        if self.DivModel == 'KLD':
            def DivModel(Ztrue,Zest):
                    return np.abs(((Ztrue)**2)-((Zest)**2))**.5
        self.DivModel = np.vectorize(DivModel,otypes=[float])
    def DiverganceLOO(self):
        params=[self.C, self.a, self.nugget, self.anisotropy_factor]
        self.estimates=[]
        for i in range(len(self.points)):
            model = OrdinaryKrigning(np.delete(self.points, i, axis=0), np.delete(self.zvals, i),Variogram=self.variogram)
            model.ManualParamSet(*params)
            model.Matrixsetup()
            estimate = model.SinglePoint(*self.points[i])
            self.estimates.append(estimate)
        Divscores= self.DivModel(self.zvals_org,self.estimates)
        self.normdists=Divscores/np.linalg.norm(Divscores)
        return self.normdists, Divscores
    def plot_div(self,resolution=10,saveplot=False):
        
        grid_x, grid_y = np.mgrid[0:900:900j, 0:5000:5000j]

        rbf = Rbf(self.points[:,0], self.points[:,1], self.normdists, function='cubic')
        grid_z = rbf(grid_x, grid_y)
        
        plt.imshow(grid_z,cmap='magma',aspect='auto',origin='lower',extent=[0, 900, 0, 5000],)
        plt.scatter(self.points[:,0],self.points[:,1],cmap='magma',marker='^',s=30,c=self.normdists)
        plt.clim(0,np.max(grid_z))
        plt.colorbar()
        plt.title('Divergances')
        plt.xlabel('x')
        plt.ylabel('y')
        if saveplot==True:
            plt.savefig()
        plt.show()
        plt.close()

         



###Options###
#colum_defs
xi_col=0
yi_col=1
zi_col=4
#resolution
resolution=10
#kriging config
Params=[]

Variogram='gaussian'


loadeddata=np.loadtxt('data\surface_roughness.csv',delimiter=',',skiprows=1)
points=loadeddata[:,[xi_col,yi_col]]
zi=loadeddata[:,zi_col]

        
krige=SpacialSensitivityAnalysis(points,zi,Variogram=Variogram)
krige.AutoKrige(step=resolution)
distances=krige.DiverganceLOO()
krige.plot_div()
