# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:00:56 2018

@author: Debasish Jyotishi
"""

import numpy as np

def VMD(signal, alpha=120, tau=0.8, K=5, DC=0, init=1, tol=10**(-7)):
    
    
    #---------- Preparations
    T=signal.size
    fs=1/T
    
    # extend the signal by mirroring
    x=np.zeros(2*T)
    x[0:int(np.ceil(T/2))]=signal[int(np.ceil(T/2)-1)::-1]
    x[int(np.ceil(T/2)):(int(np.ceil(T/2))+T)]=signal
    x[int(np.ceil(T/2)+T):]=signal[T:int(np.ceil(T/2)-1):-1]
    T=x.size
    t=np.arange(1,T+1)/T
    freq = t-0.5-1/T
    
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N=500
    Alpha=alpha*np.ones((1,K))
    
    #fourier transform of the signal
    f=np.fft.fftshift(np.fft.fft(x))
    f[0:int(T/2)]=0
    
    # intialisation of matrix of modes of all the iterations.
    u_hati=np.zeros((N,freq.size,K)).astype(complex)
    
    #initialisation of mode centers
    #init    - 0 = all omegas start at 0
    #               1 = all omegas start uniformly distributed
    #                2 = all omegas initialized randomly
    omegai=np.zeros((N,K))
    if init==1:
        for i in range(0,K):
            omegai[0,i] = (0.5/K)*i
    elif init==2:
        omegai[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
        
    if DC:
        omegai[0,0]=0
    
    eps=2.2204*np.exp(-16)
    udiff=tol+eps
    n=0
    sum_uk=0
    lambda_hat=(np.zeros((N,freq.size))).astype(complex)
    
    # Main loop for iterative update.
    
    while ((udiff>=eps) & (n<(N-1))):
        
        # updating the first mode
        
        sum_uk= sum_uk+u_hati[n,:,K-1]-u_hati[n,:,0]
        u_hati[n+1,:,0]=np.divide(f-sum_uk+(lambda_hat[n,:]/2),(1+Alpha[0,0]*np.power((freq-omegai[n,0]),2)))
        if ~DC:
            omegai[n+1,0]=np.divide(np.sum(np.multiply(freq[int(T/2):T],np.power(np.absolute(u_hati[n+1,int(T/2):T,0]),2))),np.sum(np.power(np.absolute(u_hati[n+1,int(T/2):T,0]),2)))
        
        for k in range(1,K):
            sum_uk= sum_uk+u_hati[n+1,:,k-1]-u_hati[n,:,k]
            u_hati[n+1,:,k]=np.divide(f-sum_uk+(lambda_hat[n,:]/2),(1+Alpha[0,0]*np.power((freq-omegai[n,k]),2)))
            omegai[n+1,k]=np.divide(np.sum(np.multiply(freq[int(T/2):T],np.power(np.absolute(u_hati[n+1,int(T/2):T,k]),2))),np.sum(np.power(np.absolute(u_hati[n+1,int(T/2):T,k]),2)))
        
        lambda_hat[n+1,:]=lambda_hat[n,:]+tau*(f-np.sum(u_hati[n+1,:,:],1))
        n=n+1
        
        udiff=eps
        for i in range(0,K):
            udiff=udiff+(1/T)*np.matmul((u_hati[n,:,i]-u_hati[n-1,:,i]),np.conj(u_hati[n,:,i]-u_hati[n-1,:,i]))
        
        udiff=np.absolute(udiff)
        
    N=min(n,N)
    print(N)
    omega=omegai[N,:]
    
    u_hat=(np.zeros((T,K))).astype(complex)
    u_hat[int(T/2):,:]=u_hati[N,int(T/2):,:]
    u_hat[1:(int(T/2)+1),:]=np.conjugate(u_hati[N,T:(int(T/2)-1):-1,:])
    u_hat[0,:]=np.conjugate(u_hati[N,(T-1),:])
    
    ui=np.zeros((K,T))
    for i in range(0,K):
        ui[i,:]=np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,i])))
        
    u=ui[:,int(np.ceil(T/4)):(int(np.ceil(T/4))+int(T/2))]
    
    u_hat_spectrum=(np.zeros((K,int(T/2)))).astype(complex)
    for i in range(0,K):
        u_hat_spectrum[i,:]=np.fft.fftshift(np.fft.fft(u[i,:]))
        
    return(u.T,u_hat_spectrum.T,omega)
    
    
        
            
    

