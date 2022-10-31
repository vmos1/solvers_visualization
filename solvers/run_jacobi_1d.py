import numpy as np
import matplotlib.pyplot as plt
import time

if __name__=='__main__':
    
    ## Input parameters
    N=64; max_iters=100;m=0.4;
    
    diag=2.0+m**2.0 ## Diagonal value
    b=np.zeros(N,dtype=np.float64)
    ## Source term
    b[N//2-1]=1.0

    def f_jacobi(x,b,diag,max_iters,op_file):
        ''' Performing Jacobi Iterations for Laplace operator'''
        x_temp = x.copy()  # Temp array required for Jacobi

        open(op_file,'w').close() # Create empty file
        with open(op_file,'a') as f:
            for k in range(1,max_iters):
                for i in range(N):
                    # Store in temp array until all are updated.
                    x_temp[i]=(b[i]+ x[(i-1+N)%N] + x[(i+1)%N])* (1.0/diag)
                    
                x=x_temp.copy()
                f.write(str(k)+","+",".join([str(i) for i in x])+'\n')

    
    op_file='op2.out'
    x=np.ones(N,dtype=np.float64)*0.0
    f_jacobi(x,b,diag,max_iters,op_file)