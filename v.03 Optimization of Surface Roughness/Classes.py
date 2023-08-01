import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import linalg
import time



Points=np.loadtxt('surface_roughness.csv',dtype=float,skiprows=1,delimiter=',',usecols=[0,1])
Zpoints=np.loadtxt('surface_roughness.csv',dtype=float,skiprows=1,delimiter=',',usecols=[-2])




class OrdinaryKrigning:
    def __init__(self,Points,Zvals,Variogram='gaussian'):
        self.points=Points
        self.zvals=Zvals
        self.variogram=Variogram
    #variogram selector
        if self.variogram =='gaussian' or 'Gaussian' :
            def Variogram(h, a,C):
                # Gaussian model with sill C and range a
                return C * (1 - np.exp(-1*((h/a)**2)))
        if self.variogram =='spherical' or 'Spherical' : 
            def Variogram(h, a,C):
                # Spherical model with sill C and range a
                if h >= a:
                    return C  # Variogram value reaches the sill at the range
                else:
                    return C * (1 - ((3/2) * (h/a) - (1/2) * (h/a)**3))            
        self.vVariogram = np.vectorize(Variogram,otypes=[float])

    def ManualParamSet(self,C,a,nugget,anisotropy_factor):
        self.a=a
        self.anisotropy_factor=anisotropy_factor
        self.C=C
        self.nugget=nugget
    def Matrixsetup(self):
        # Compute the pairwise distance matrix
        distances = np.sqrt((self.points[:, None, 0] - self.points[None, :, 0]) ** 2 + 
                    (self.points[:, None, 1] - self.points[None, :, 1]) ** 2 / self.anisotropy_factor ** 2)

        result = self.vVariogram(distances, self.a,self.C)

        # Add a row of ones at the bottom and a column of ones at the right
        result = np.pad(result, ((0, 1), (0, 1)), mode='constant', constant_values=1)

        # Replace the bottom-right corner with 0
        result[-1, -1] = 0

        result = result + np.eye(result.shape[0]) * self.nugget

        self.result=result
    def SinglePoint(self,Xo,Yo,training_points=None):
        if training_points is None:
            training_points = self.points

        point0 = np.array([Xo, Yo])

        distances_to_point0 = np.sqrt((training_points[:, 0] - Xo) ** 2 + (training_points[:, 1] - Yo) ** 2 / self.anisotropy_factor ** 2)

        vectorb = self.vVariogram(distances_to_point0, self.a,self.C)

        vectorb = np.append(vectorb, 1)

        lamd=linalg.solve(self.result,vectorb,assume_a='sym')

        lamd=np.delete(lamd,-1)

        zout=np.dot(lamd,self.zvals.T) 
        return zout 
    def interpgrid(self, xmin, xmax, ymin, ymax, step):
        def __guess(point0):
            distances_to_point0 = np.sqrt(np.sum((self.points - point0) ** 2, axis=1))

            vectorb = self.vVariogram(distances_to_point0, self.a,self.C)

            vectorb = np.append(vectorb, 1)

            lamd = np.linalg.solve(self.result, vectorb)

            lamd = np.delete(lamd, -1)

            zout = np.dot(lamd, self.zvals.T) 
            return zout 
        
        SingleGuess = np.vectorize(__guess, signature='(n)->()')

        x_range = np.arange(xmin, xmax, step)
        y_range = np.arange(ymin, ymax, step)

        # Create a grid of points
        X, Y = np.meshgrid(x_range, y_range)

        # Stack the points into a 2D array
        points_to_estimate = np.column_stack((X.flatten(), Y.flatten()))

        z = SingleGuess(points_to_estimate)

        # Reshape the results to match the shape of the original grid
        z = z.reshape(X.shape)
        return z
    def AutoOptimize(self, InitialParams=None):
        if InitialParams is None:
            InitialParams = [np.var(self.zvals), np.max(np.sqrt(np.sum((self.points[:, None, :] - self.points[None, :, :]) ** 2, axis=-1)))/2, .001, 1]

        self.ManualParamSet(*InitialParams)
        self.params=InitialParams 
        def mseLOO(params):
            global itcounts
            C, a, nugget, anisotropy_factor = params
            errors = []
            for i in range(len(self.points)):
                left_out_point = self.points[i]
                left_out_value = self.zvals[i]
                # Perform Kriging with all points except the left-out one
                model = OrdinaryKrigning(np.delete(self.points, i, axis=0), np.delete(self.zvals, i))
                model.ManualParamSet(*params)
                model.Matrixsetup()
                estimate = model.SinglePoint(*left_out_point)
                errors.append((estimate - left_out_value)**2)
            print(params)
            return np.mean(errors)

        # Perform the optimization
        result = minimize(mseLOO, self.params,method='L-BFGS-B')
        C_opt, a_opt, nugget_opt, anisotropy_factor_opt = result.x
        self.a=a_opt
        self.anisotropy_factor=anisotropy_factor_opt
        self.C=C_opt
        self.nugget=nugget_opt     
    def AutoKrige(self,step=1):
        t0 = time.time()

        self.AutoOptimize()
        self.Matrixsetup()
        z=self.interpgrid(xmax=np.max(self.points[:,0]),xmin=np.min(self.points[:,0]),ymax=np.max(self.points[:,1]),ymin=np.min(self.points[:,1]),step=step)

        t1 = time.time()
        self.exetime = t1-t0
        return z
"""
test=OrdinaryKrigning(Points,Zpoints,Variogram='spherical')
Zarray=test.AutoKrige(step=50)
print(test.exetime)



plt.imshow(Zarray, origin='lower',extent=[0, 900, 0, 5000],aspect='auto',interpolation_stage='rgba')
plt.scatter(Points[:,0],Points[:,1],marker='3',s=100,c=Zpoints)
plt.title(f'{test.variogram} ')
plt.colorbar(label='Z value')
plt.clim(0,30)
plt.show()

"""
class UniversalKriging(OrdinaryKrigning):
    def __init__(self, Points, Zvals, Variogram='gaussian'):
        super().__init__(Points, Zvals, Variogram)
        
        # Define the trend function
        self.trend_function = lambda trend_params, x, y: trend_params[0] + trend_params[1]*x + trend_params[2]*y + trend_params[3]*x**2 + trend_params[4]*y**2

    def calc_trend_coefficients(self):
        # Define the residuals function
        residuals = lambda trend_params: self.trend_function(trend_params, self.points[:, 0], self.points[:, 1]) - self.zvals

        # Estimate the trend parameters
        self.trend_params, _, _, _ = np.linalg.lstsq(np.column_stack((np.ones(len(self.points)), self.points[:, 0], self.points[:, 1], self.points[:, 0]**2, self.points[:, 1]**2)), self.zvals, rcond=None)
        
        # Subtract the estimated trend from the data
        self.zvals -= self.trend_function(self.trend_params, self.points[:, 0], self.points[:, 1])

    def SinglePoint(self, Xo, Yo, training_points=None):
        # First, add back the trend to the actual observations
        zvals_original = self.zvals + self.trend_function(self.trend_params, self.points[:, 0], self.points[:, 1])

        # Perform kriging on detrended data and add the trend back
        return super().SinglePoint(Xo, Yo, training_points) + self.trend_function(self.trend_params, Xo, Yo)

    def interpgrid(self, xmin, xmax, ymin, ymax, step):
        # Perform kriging on detrended data
        z = super().interpgrid(xmin, xmax, ymin, ymax, step)

        # Add the trend back to the kriging estimates
        x_range = np.arange(xmin, xmax, step)
        y_range = np.arange(ymin, ymax, step)
        X, Y = np.meshgrid(x_range, y_range)
        z += self.trend_function(self.trend_params, X, Y)
        
        return z


test = UniversalKriging(Points, Zpoints, Variogram='spherical')
test.calc_trend_coefficients()  # calculate the trend coefficients
Zarray = test.AutoKrige(step=50)
print(test.exetime)



plt.imshow(Zarray, origin='lower',extent=[0, 900, 0, 5000],aspect='auto',interpolation_stage='rgba')
plt.scatter(Points[:,0],Points[:,1],marker='3',s=100,c=Zpoints)
plt.title(f'{test.variogram} ')
plt.colorbar(label='Z value')
plt.clim(0,30)
plt.show()
