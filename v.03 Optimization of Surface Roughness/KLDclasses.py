from Ok_Uk_Module import *
from scipy.interpolate import Rbf
from scipy.special import kl_div

class SpacialSensitivityAnalysis(OrdinaryKrigning):


    def __init__(self, Points, Zvals, Variogram='gaussian',DiverganceModel='KLD'):
        super().__init__(Points, Zvals, Variogram)
        self.DivModel=DiverganceModel
        if self.DivModel == 'KLD':
            def DivModel(p, q):
                m = 0.5 * (p + q)
                kld=np.mean(0.5 * np.sum(kl_div(p, m)) + 0.5 * np.sum(kl_div(q, m)))
                return kld
        self.DivModel = np.vectorize(DivModel,otypes=[np.float64])


    def DiverganceLOO(self):
        print(np.shape(np.asarray(self.points[:,0]).squeeze()))
        print(np.shape(np.asarray(self.points[:,1]).squeeze()))
        params=[self.C, self.a, self.nugget, self.anisotropy_factor]
        self.divscores=[]
        for i in range(len(self.points)):
            model = OrdinaryKrigning(np.delete(self.points, i, axis=0), np.delete(self.zvals, i),Variogram=self.variogram)
            estimate = model.AutoKrige(step=100,bounds=[np.max(self.points[:,0]),np.min(self.points[:,0]),np.max(self.points[:,1]),np.min(self.points[:,1])])
            self.divscores.append((np.mean(self.DivModel(np.abs(self.zarray/np.sum(self.zarray)),np.abs(estimate/np.sum(estimate))))))

        return self.divscores
    

    def plot_div(self,resolution=10,saveplot=False):
        
        np.save('div_scores',self.divscores)

        interpz = OrdinaryKrigning(points,np.asarray(self.divscores),Variogram='linear')
        grid_z=interpz.AutoKrige(step=10)
        plt.imshow(grid_z,cmap='magma',aspect='auto',origin='lower',extent=[0, 900, 0, 5000],)
        plt.scatter(self.points[:,0],self.points[:,1],cmap='magma',marker='^',s=30,c=self.divscores)
        plt.clim(np.min(grid_z),np.max(grid_z))
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
krige.AutoKrige(step=100)
distances=krige.DiverganceLOO()
krige.plot_div()
krige.Plot()
