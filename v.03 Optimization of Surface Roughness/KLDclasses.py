from Classes import *

class SpacialSensitivityAnalysis(OrdinaryKrigning):
    def __init__(self, Points, Zvals, Variogram='gaussian',DiverganceModel='KLD'):
        super().__init__(Points, Zvals, Variogram)