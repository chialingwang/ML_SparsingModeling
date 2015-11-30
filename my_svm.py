import numpy as np
import sys,os
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from enum import Enum

class stop(Enum):
    iter = 1
    opt = 2
    perfor =3

class my_svm(object):
#svm = my_svm(X1,y1,iter = 10 , step = 0.11 , stop = stop.perfor)    
    def __init__(self, **kwargs):
            self.h = 0.5
            self.C = 1
            self.e = 0.1e-3
            self.seed = 371986 
            self.iter = kwargs.get('iter', 1000)

            
    def compute_yt(self , w):
        return self.y*np.dot(self.X,w)
    
    def compute_yx(self):
        list = []
        for i in range(len(self.y)):
            list.append(self.y[i]*self.X[i])
        return list
    
    def huber_hinge(self ,yt):
        if(yt < 1-self.h ):
            return 1-yt
        if (yt >= 1-self.h and yt <= 1+self.h ):
            return ( pow((1+self.h-yt), 2) )/(4*self.h)
        if (yt > 1+self.h):
            return 0

        
    def compute_obj(self ,w): 
        result = 0
#        print(self.C)
        Allyt = self.compute_yt(w)
        for yt in Allyt:
            result += self.huber_hinge(yt)
        return pow(np.linalg.norm(w),2) + (self.C/self.n)*result


    def gradient_huber_hinge(self ,yt , yx):
        if(yt < 1-self.h ):
            return (-yx)
        if (yt >= 1-self.h and yt <= 1+self.h ):
            return(-( 1+self.h-yt )/(2*self.h))*yx
        if (yt > 1+self.h):
            return(0)


    def compute_grad(self ,w , i = -1):
        grad_list = []
        Allyt = self.compute_yt(w)  # t = w^T dot x  return value of each data
        Allyx = self.compute_yx()   # return vector of each data
        if(i == -1):
            for index in range(0,len(self.X)):
                yt = Allyt[index]
                yx = Allyx[index]
                grad_list.append(self.gradient_huber_hinge(yt , yx))
        else :
                yt = Allyt[i]
                yx = Allyx[i]
                grad_list.append(self.gradient_huber_hinge(yt , yx))
        return 2*w+(self.C/self.n)*np.sum(c for c in grad_list)

    

    def grad_checker(self,  X, y , w):
        self.X =X
        self.y =y
        self.n = len(X)
        numresult = self.getNumDiffResult(w)
        gradresult = self.compute_grad(w)
        print("Numerical result : ")
        print(numresult)
        print("My Gradient result : ")
        print(gradresult)
        return sum( gradresult - numresult )/sum(numresult) * 100
    def getNumDiffResult(self, w):
        result = []
        board = np.zeros((len(w) , len(w)))
        np.fill_diagonal(board , self.e)
        wp = np.add(w,board)
        wn = np.subtract(w,board)
        def df(wp,wn):
            return ( self.compute_obj(wp) - self.compute_obj(wn) )/(2*self.e)
        
        for i in range(len(w)):
            result.append(df(wp[i],wn[i]))
        
        return result


    def validation_data(self,data,target, num):
        rng = np.random.RandomState(self.seed)
        i = 0
        data_list = []
        target_list = []
        while i < num:
            index = rng.randint(0, len(data))
            pick_data = data[index]
            pick_target = target[index]
            data_list.append(pick_data)
            target_list.append(pick_target)
            i += 1
        return data_list,target_list
    
    def step_size(self, w , obj , grad , step , iter =10):  #backtracking line search c in (0,1) ta in (0,1)
        c = 1
        ta = 0.9
        m = np.dot(grad,grad)
        t = -c*m
        j = 0
        while(j < iter):
            new_obj = self.compute_obj(w+step*grad)
            if((obj-new_obj >= t*step) ):
                break
            step = ta*step
            j += 1
        return step
    
    def my_gradient_descent(self, X, y,**kwargs):
        self.X =X
        self.y =y
        self.C = kwargs.get('C', 1)
        self.n = len(X)
        obj_result = []
        w_result = []
        step_result= []
        back_track = False
        step = kwargs.get('step', -1)
        self.w = np.zeros(self.X.shape[1])
        self.stop = kwargs.get('stop', stop.iter)
        window = kwargs.get('window', 10)
        i = 0
        if(step == -1):
            back_track = True
            step = 1.1
        if(self.stop.value == 3) :
            valid_len = 10
            pickX , picky = self.validation_data(X,y,valid_len)
        min_err = sys.maxsize
        while i < self.iter:
            gradF = self.compute_grad(self.w)
            obj = self.compute_obj(self.w)
            if(back_track):
                step = self.step_size(self.w , obj , gradF , step )
            obj_result.append(obj)
            step_result.append((step))
            self.w -= np.multiply(step,gradF)
            w_result.append(list(self.w))
            count = 0
            i += 1
            if(self.stop.value == 2 ): # stop when opt
                new_obj = self.compute_obj(self.w)
                if(np.linalg.norm(new_obj-obj)<self.e):
                    print("break because stop criteria II at iteration :" , i )
                    break
                
            if(self.stop.value == 3) : # stop when good perform

                tol = 0.9
                error = self.predict(pickX , picky)
                if(error <= tol*min_err or count < window):
                    min_err = error
                else :
                    print("break because stop criteria III at iteration :" , i )
                    break
            
        return w_result , obj_result , step_result
    
    
    def my_sgd(self, X, y,**kwargs):
        self.X =X
        self.y =y
        self.C = kwargs.get('C', 1)
        self.n = len(X)
        obj_result = []
        w_result = []
        step_result= []
        back_track = False
        self.w = np.zeros(self.X.shape[1])
        self.stop = kwargs.get('stop', stop.iter)
        step = kwargs.get('step', 0.11)
        t0 = kwargs.get('t0', 0)
        seed = kwargs.get('seed', 371986)
        window = kwargs.get('window', 10)
        i = 0
        min_err = sys.maxsize
        rng = np.random.RandomState(seed)
        stop_sign = False
        if(self.stop.value == 3) :
            valid_len = 10
            pickX , picky = self.validation_data(X,y,valid_len)
        while i < self.iter: 
            obj = self.compute_obj(self.w)
            obj_result.append(obj)
            step = (step)/(i+t0)
            step_result.append((step))
            w_iter = []
            count = 0 
            index = i % self.n

#                print(index)
            gradF  = self.compute_grad(self.w , index)
#                print(gradF)
                
            self.w -= np.multiply(step,gradF)
            w_iter.append(self.w)
            if(self.stop.value == 3) : # stop when good perform
 
                tol = 0.9
                error = self.predict(pickX , picky)
                if(error <= tol*min_err or count < window):
                    min_err = error
                    count += 1
                else :
                    
                    break
            if index == 0:
                permutation  = rng.permutation(self.n)
                self.X, self.y = self.X[permutation], self.y[permutation]
                self.w = np.mean(w_iter,0)
                w_result.append(self.w)
           
            i += 1

        return w_result , obj_result , step_result
    
 
    
   
    def predict(self , X , y,**kwargs) :
        w = kwargs.get('w', self.w)
        TotalError = 0
        for i in range(len(X)):
            TotalError += (self.produce(X[i] , w) != y[i])
        return ( TotalError  / len(X) )*100     

    def produce(self , X , w):
        #return np.sign(np.dot(X, self.w)+self.b)
        return np.sign(np.dot(X, w))