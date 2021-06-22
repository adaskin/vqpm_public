#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:49:36 2021
includes some utility functions for vqpm
@author: adaskin
"""
import numpy as np
def print_debug(msg, psi, n):
    ''' prints qubit probabilities of 1...n qubit in given quantum state
    '''
    print(msg, np.linalg.norm(psi)**2)
    for q in range(1, n+1):
        print("q:", q, calculate_prob(psi, q))


############################################
#################################################################
def calculate_block_prob(psi,qubits):
    '''computes probabilities for the states of a given qubit group.
      psi: a quantum state.
      qubits:the set of qubits such as [1 3 5]
    **the order of the qubits |1,2,3,4,..n>;
    '''
    
    n = len(psi)

    logn = int( np.log2(n));   
    nq = len(qubits)
    f = np.zeros(2**nq);
    
    for j in range(n):
        jbits = bin(j)[2:].zfill(logn)
        ind = 0
        for q in range(nq):
            if jbits[qubits[q]-1] == '1':
                ind += 2**(nq-q-1)
      

        f[ind] += np.real(np.abs(psi[j])**2)
    return f; 
############################################
#################################################################
def calculate_prob(psi,qubit):
    '''computes probabilities for the states of a given qubit.
     psi: a quantum state.
     qubit: index of the qubit, the order of the qubits |1,2,3,4,..n>;
     '''
    n = len(psi)

    logn = int( np.log2(n));   

    f = np.zeros(2);
    qmask = (1<<logn-qubit)
    #for k in range(0, len(psi)) #the columns of the input
    for j in range(n):
        
        ind = 0;
        index = ((j & qmask) != 0);
        ind = ind + index;   
       
        f[ind] = f[ind]+np.real(np.abs(psi[j])**2);
    return f; 
    
############################################
#################################################################
def prepareNewState(outVec, n,qStates):
    '''        
     based on probabilities of the qubits generates new state
     measures qubits 1 to n
     prepares a state by combining them
    ----------
    outVec: a quantum state
    n: #qubits
    qStates: given qubit states
 
    Returns a state, and new qStates
    
    '''
    p = 4      #precision in probs
    p2 = 0.001 #diff in probs to assume qubit is 1 or 0
  
    state = [1]
    
    for q in range(1,n+1):
        if q in qStates:
            stateQ = qStates[q]
        else:
            stateQ = calculate_prob(outVec, q)
           
            stateQ = np.round(stateQ,p);
        
            if(stateQ[0]>stateQ[1]+p2):
                stateQ[1]= 0
                stateQ[0] = 1
                qStates[q]=stateQ
                
            elif stateQ[0]+p2<stateQ[1]:
                stateQ[0]=0
                stateQ[1] = 1
                
                qStates[q]=stateQ
            else:
                stateQ = np.sqrt(stateQ)
      
        print("q:", q, stateQ)
        state = np.kron(state, stateQ)    
    return state,qStates  
############################################