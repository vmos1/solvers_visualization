import numpy as np
#import matplotlib.pyplot as plt
import time
import argparse

def f_get_residue(b,x,diag):
    ''' Compute residue '''
    res=np.zeros_like(b)
    for i in range(N):
        x1=i%L
        y1=i//L
        res[i]=b[i]- x[i] * diag + x[(x1-1+L)%L+y1*L] + x[(x1+1)%L+y1*L] + x[x1+((y1-1+L)%L)*L] + x[x1+((y1+1)%L)*L]

    return res

def f_jacobi(x,b,diag,max_iters,op_file):
    ''' Performing Jacobi Iterations for Laplace operator'''
    x_temp = x.copy()  # Temp array required for Jacobi
    res=f_get_residue(b,x,diag)

    open(op_file,'w').close() # Create empty file
    with open(op_file,'a') as f:
        for k in range(1,max_iters):
            for i in range(N):
                # Store in temp array until all are updated.
                x_temp[i]=(b[i]+ x[(i-1+N)%N] + x[(i+1)%N])* (1.0/diag)

                x1=i%L
                y1=i//L
                x_temp[i]=(b[i]+ x[(x1-1+L)%L+y1*L] + x[(x1+1)%L+y1*L] + x[x1+((y1-1+L)%L)*L] + x[x1+((y1+1)%L)*L])* (1.0/diag)

            x=x_temp.copy()
            f.write(str(k)+","+",".join([str(i) for i in x])+'\n')

            res=f_get_residue(b,x,diag)
            if np.allclose(res,np.zeros(N),rtol=1e-5,atol=1e-12):
                print("Quitting after {0} iterations".format(k))
                break
            if (k==max_iters-1): print("No convergence after % iterations"%(max_iters))


def f_gauss_seidel(x,b,diag,max_iters,op_file):
    ''' Performing Gauss_seidel Iterations for Laplace operator'''
    
    res=f_get_residue(b,x,diag)

    open(op_file,'w').close() # Create empty file
    with open(op_file,'a') as f:
        for k in range(1,max_iters):
            for i in range(N):
                # Store in temp array until all are updated.
                x[i]=(b[i]+ x[(i-1+N)%N] + x[(i+1)%N])* (1.0/diag)

                x1=i%L
                y1=i//L
                x[i]=(b[i]+ x[(x1-1+L)%L+y1*L] + x[(x1+1)%L+y1*L] + x[x1+((y1-1+L)%L)*L] + x[x1+((y1+1)%L)*L])* (1.0/diag)

            f.write(str(k)+","+",".join([str(i) for i in x])+'\n')

            res=f_get_residue(b,x,diag)
            if np.allclose(res,np.zeros(N),rtol=1e-5,atol=1e-12):
                print("Quitting after {0} iterations".format(k))
                break
            if (k==max_iters-1): print("No convergence after % iterations"%(max_iters))

            
def f_cg(x,b,diag,max_iters,op_file):

    r_old=f_get_residue(b,x,diag)
    p=r_old.copy()
    
    Ap=np.zeros_like(b)
    
    open(op_file,'w').close() # Create empty file
    with open(op_file,'a') as f:
        for k in range(1,max_iters):
            for i in range(N):
                # Compute A . p 
                x1=i%L
                y1=i//L            
                Ap[i]=(p[x1+y1*L]*diag - p[(x1-1+L)%L+y1*L] - p[(x1+1)%L+y1*L] - p[x1+((y1-1+L)%L)*L] - p[x1+((y1+1)%L)*L])

            rsquare=np.dot(r_old,r_old)
            alpha=rsquare/np.dot(p,Ap)
            x+=alpha*p  # Update solution
            r_new=r_old-alpha*Ap  # Update residue

            beta=np.dot(r_new,r_new)/rsquare

            p= r_new + beta * p
            r_old=r_new.copy()
        
            f.write(str(k)+","+",".join([str(i) for i in x])+'\n')

            res=f_get_residue(b,x,diag)
            if np.allclose(r_new,np.zeros(N),rtol=1e-5,atol=1e-12):
                print("Quitting after {0} iterations".format(k))
                break
            if (k==max_iters-1): print("No convergence after % iterations"%(max_iters))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Solvers for 2D Laplace operator")
    add_arg = parser.add_argument
    
    add_arg('--L','-L', type=int, default='16',help='Length of 2D lattice.')
    add_arg('--iters','-itr', type=int, default='2000',help='Max number of iterations')
    add_arg('--m','-m', type=float, default='0.5',help='Mass')
    add_arg('--solver', type=str, choices=['Jacobi','GS','CG'], default='Jacobi', help='Which solver to use: Jacobi, Gauss-Seidel or Conjugate Gradient.')

    return parser.parse_args()


if __name__=='__main__':
    
    args=parse_args()
    solver=args.solver
    
    ## Input parameters
    L=64; max_iters=2000;m=0.4;
    L,max_iters,m=args.L,args.iters,args.m
    
    print("Running %s with m %.2f, L %d for max %d iters"%(solver,m,L,max_iters))
    N=L**2
    diag=4.0+m**2.0 ## Diagonal value
    b=np.zeros(N,dtype=np.float64)
    ## Source term
    b[L//2-1 + L * (L//2-1)]=1.0
    
    op_file='op2.out'
    x=np.ones(N,dtype=np.float64)*0.1
    
    if solver=='Jacobi': 
        f_jacobi(x,b,diag,max_iters,op_file)
    elif solver=='GS':
        f_gauss_seidel(x,b,diag,max_iters,op_file)
    elif solver=='CG':
        f_cg(x,b,diag,max_iters,op_file)