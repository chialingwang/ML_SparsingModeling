import numpy as np

class Choose_Target(object):
    
    def __init__(self , Xin , yin):
        self.X = Xin
        self.y = yin
        
    def select(self , i ):
        #self.X = self.Ｘ[(self.y == i)]  # select taget i and j
        #self.y = self.y[(self.y == i) ]

        
        
        self.y[(self.y == i)] = -1 # make all target "i" become "-1" to do binary classfication
        self.y[(self.y != -1)] = 1  