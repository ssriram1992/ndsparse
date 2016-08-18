# ###############################################################################
# Author: Sriram Sankaranarayanan
# File: sparse.py
# Institution: Johns Hopkins University
# Contact: ssankar5@jhu.edu
#
# All rights reserved.
# You are free to distribute this code for non-profit purposes
# as long as this header is kept intact
# ###############################################################################

import numpy as np
import warnings

class coo_array(object):
    """
    Generalization of coo_matrix for n-dimensional array.
    Not possible to change the shape of the array
    """
    format = 'coo'
    def __init__(self, arg1, arg2=None, arg3 = None, dtype=None):
        if isinstance(arg1, tuple):
            self.shape = arg1
            self.positions = np.empty((0,len(self.shape)))
            self.values = np.empty((0,),dtype=dtype)
        elif isinstance(arg1, np.ndarray) and arg2 is None:
            self.shape = arg1.shape
            temp = np.where(arg1)
            l = temp[0].size
            self.positions = temp[0].copy().reshape(1,l)
            for i in np.arange(1,len(temp)):
                self.positions = np.concatenate((self.positions, temp[i].reshape(1,l)),0)
            self.positions = self.positions.T
            self.values = arg1[temp]
        elif isinstance(arg1, np.ndarray) :
            if arg3 is None:
                self.shape = ()
                for i in np.arange(arg1.shape[1]):
                    self.shape += (np.max(arg1[:,i])+1,)
            else:
                self.shape = arg3
            self.positions = arg1
            self.values = arg2
        elif isinstance(arg1, coo_array):
            self.shape = arg1.shape.copy()
            self.positions = arg1.positions.copy()
        else:
            raise ValueError("Improper invocation of coo_array")
    def __str__(self):
        ans = ""
        for i,v in zip(self.positions,self.values):
            ans += i.__str__()
            ans += " ---> "
            ans += str(v)
            ans += '\n'
        return ans
    # Removes any "0" elements from storage
    def flush(self,tol = 1e-15):
        temp = np.where(np.abs(self.values)<=tol)[0]
        if not temp.size==0:
            self.positions = np.delete(self.positions, temp,0)
            self.values = np.delete(self.values, temp)
    def size(self):
        return self.values.size
    def add_entry(self,posn,val): # Be careful as this can cause duplication. But it is fast.
        if len(posn.shape)!=1:
            if posn.shape[0]!=val.size:
                raise ValueError("Inconsistent shapes")
        else:
            if (not np.isscalar(val)):
                if val.size!=1:
                    raise ValueError("Inconsistent shapes")
        self.positions = np.concatenate((self.positions,posn),0)
        self.values = np.append(self.values, val)
    def set_entry(self,posn,val): # Makes sure duplication doesn't occur. But this is slow.
        if len(posn.shape)==1:
            if (not np.isscalar(val)):
                if val.size!=1:
                    raise ValueError("Inconsistent shapes")
            temp = np.where(np.all(self.positions==posn,1))[0]
            if temp.size==0:
                self.positions = np.concatenate((self.positions,posn.reshape(1,self.positions.shape[1])),0)
                self.values = np.append(self.values, val)
            elif temp.size==1:
                self.values[temp] = val
            else:
                raise ValueError("Multiple assignments to single entry")
                self.values[temp[0]] = val
                self.values[temp[1:]] = 0
        else:
            if posn.shape[0]!=val.size:
                raise ValueError("Inconsistent shapes")
            for i,v in zip(posn,val):
                self.set_entry(i, v)
    def get_entry(self,posn): # Returns the values at the said positions
        if len(posn.shape)==1:
            temp = np.where(np.all(self.positions==posn,1))[0]
            if temp.size==0:
                ans = 0
            elif temp.size==1:
                ans = self.values[temp]
            else:
                raise ValueError("Multiple assignments to single entry")
                ans = np.sum(self.values[temp])
        else:
            ans = np.zeros((0,))
            for i in posn:
                ans = np.append(ans, self.get_entry(i))
                # temp = np.where(np.all(self.positions==i))[0]
                # if temp.size==0:
                #     ans = np.append(ans, 0)
                # elif temp.size==1:
                #     ans = np.append(ans, self.values[temp])
                # else:
                #     warnings.warn("Multiple assignments to single entry")
                #     ans = np.append(ans, np.sum(self.values[temp]))
        return ans
    def remove_duplicate_at(self,posn,func = 0):
        temp = np.where(np.all(self.positions==posn,1))[0]
        if temp.size>1:
            if np.isscalar(func):
                self.values[temp]=0
                self.values[temp[0]]=func
            else:
                temp2 = self.values[temp]
                self.values[temp]=0
                self.values[temp[0]]=func(temp2)
            self.flush()
    def swapaxes(self,axis1,axis2):
        self.positions = np.swapaxes(self.positions, axis1, axis2)
    def todense(self):
        ans = np.zeros(self.shape)
        for i,v in self.iterate():
            ans[tuple(i)] += v
        return ans
    def iterate(self):
        return zip(self.positions,self.values)