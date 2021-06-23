'''
QPM converges the eigenvector of I+U with the maximum magnitude 
that is max |1+e^{i\lambda}|, where \lambda is the minimum eigenvalue of H.
The file includes the implementations of variational quantum power method (vqpm) applied to random QUBO, 
vqpmForQUBO: it finds the minimum eigenphase of u 

@author: adaskin, the last updated on June 23, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
from vqpmUtilities import   prepareNewState

def vqpmForQubo(u, n, maxiter, iexpected,  pdiff, precision):
    '''
    measures the state at each iteration
    '''
    Pimin = np.zeros((maxiter,), float);
    N = 2 ** n;
    s2 = 1 / np.sqrt(2); 
    inVec = np.ones((N, ), dtype=complex) / np.sqrt(N); 
    
    psi1 = np.zeros((N, ), dtype=complex);#first qubit in \ket{1}
    numOfIter = maxiter;

    qStates={}    
    for j in range(0, maxiter):
        
        #inVec is the state after R_z gates
        # Apply 1st Hadamard to the first qubit
        psi0 = inVec * s2; #first qubit is 0
        psi1 = inVec * s2; #first qubit is 1
        
        # print(np.abs(u))
        # print(np.abs(psi1))
        
        # apply CU
        for k in range(0, N):
            psi1[k] = u[k] * psi1[k];
        
        
        # Apply 2nd Hadamard to the first qubit
        psi0final = np.add(s2 * psi0, psi1 * s2); #we use I+U
        # psi1final = snp.add(s2 * psi0, -s2*psi1) 
        
        p0 = np.linalg.norm(psi0final);
        # p1 = np.linalg.norm(psi1final);
        
        psi0final = psi0final/p0
        
        Pimin[j] = np.abs(psi0final[iexpected]) ** 2;
        print("iter: %d probO Of 1st qubit: %f prob Of Expected State: %lf" % (j, p0, Pimin[j]));
        if (Pimin[j] >= 0.5):
            numOfIter = j + 1;
            break;
            
        inVec,qStates = prepareNewState(psi0final,n,qStates,  pdiff, precision)
   
        

    final_probs =  np.power(np.abs(psi0final), 2);
    stateProb = np.max(final_probs)
    foundState =np.argmax(final_probs)  
    
    
    return   foundState, stateProb, qStates, p0, numOfIter,  Pimin[0:numOfIter]
  

def unitaryForQubo(n, Q):
    '''
    creates a vector which represents the diagonal of U
    '''    
    
    # x {0, +1}
    # for given coefficients c and number of parameters n, it creates a diagonal
    # unitary whose elements encode the solution space. it returns a vector.
    N = 2 ** n;
    u = np.ones((N,), complex)
    # \sum qiixi +   # \sum qij xixj , i<= j
  
    for k in range(0, N):
        b1 = bin(k)[2:].zfill(n);
        for i in range(0, n):
            for j in range(i, n ):
                qij = np.exp(1j * Q[i][j] );
                if  (b1[i] == '1') and (b1[j] == '1'):
                    u[k] = u[k] * qij;

    return u

def testunitaryForQubo(n, Q):
    '''
    creates a vector which represents the diagonal of U
    '''    
    
    # x {0, +1}
    # for given coefficients c and number of parameters n, it creates a diagonal
    # unitary whose elements encode the solution space. it returns a vector.
    N = 2 ** n;
    u = np.ones((N,), complex)
    utest = np.zeros((N,), float)
    # \sum qiixi +   # \sum qij xixj , i<= j
  
    for k in range(0, N):
        b1 = bin(k)[2:].zfill(n);
        for i in range(0, n):
            for j in range(i, n ):
                qij = np.exp(1j * Q[i][j] );
                if  (b1[i] == '1') and (b1[j] == '1'):
                    u[k] = u[k] * qij;
                    utest[k] += Q[i][j]
    return u, utest
def randomQ(n):
    '''for agiven n parameters it generates a random coefficient matrix Q
    its diagonal should be considered as c
    '''
    Q = np.random.randn(n, n); Q = Q + np.transpose(Q);
    return Q;

##############################################################################
##############################################################################
def uForMinAbsolutePhase(Q):
    '''prepares u for min absolute value 
    '''
    maxQ = np.sum(np.sum(np.abs(Q)));
    Q1 = Q / (maxQ); #sum in [-1 1]
    
    Q1 = Q1*np.pi/4; #the sum in [-pi/4, pi/4]
   

    u = unitaryForQubo(n, Q1);
    u = np.exp(1j * np.pi/4) * u;  # the sum in [0, pi/2] always positive
    lu = np.real(np.log(u) / (1j));
    expectedState = np.argmin(lu);
    return Q1, u, lu, expectedState

def uForMinPhase(Q):
    '''
    this prepares u for min negative value
    '''
    
    maxQ = np.sum(np.sum(np.abs(Q)));
    Q1 = Q / (maxQ); # sum in [-1, 1]
    
    Q1 = Q1*np.pi/4; #the sum in [-pi/4, pi/4]
    
    negativesum = 0  #possible min negative value
    for row in Q1:
        for e in row:
            if e<0:
                negativesum += e
                

    u = unitaryForQubo(n,  Q1);
    lu = np.real(np.log(u) / (1j));

    u = np.exp(-1j * negativesum) * u;  # the sum in [negativesum, pi/4+|negativesum|]
    lu = np.real(np.log(u) / (1j));

    expectedState = np.argmin(lu);
    return Q1, u, lu, expectedState
    
if __name__ == '__main__':
    ###PARAMETERS
    pdiff = 0.001 #necessary prob diff to assume a qubit 1 or 0
    precision = 4 #precision of measurement outcome
    ########################################################
    ##RANDOM Q
    n = 4;  
    Q = np.array([[-1.33402856, -0.95754382,  0.04671655, -0.11152588],
       [-0.95754382, -2.81090125,  1.6279527 ,  1.28859629],
       [ 0.04671655,  1.6279527 ,  0.21939734, -1.84276014],
       [-0.11152588,  1.28859629, -1.84276014, -4.14428733]])

    #Q = randomQ(n) 
    Qscaled, u, lu, expectedState = uForMinAbsolutePhase(Q)
    ########################################################

    Qscaled, u, lu, expectedState = uForMinPhase(Q)
    #####################################################

    
    ############################################
    ###RUN VQPM
    foundState, stateProb, qStates, p0, numOfIter,  Pimin = vqpmForQubo(u, n, 50, expectedState,  pdiff, precision);
    
    ####################################################################
    ### print/plot results
    print("-------------------------------------")
    print("Eigenvalues in [%f, %f]" % (np.min(lu), np.max(lu)))
    print("Expected and Found Eigenvalues")
    
    print("expected  value:%f, state:%d" % (lu[expectedState], expectedState))
    print("found     value:%f, state:%d with prob:%f" % (lu[foundState], foundState, stateProb))
    sortedlu = np.sort(np.abs(lu));
    print("eigengap: ", sortedlu[1]-sortedlu[0])
    print(sortedlu)
    
    fig = plt.figure()
    plt.plot(range(numOfIter), Pimin,'b.')
    plt.ylabel('iteration')
    
    plt.xlabel('the success probability (the eigengap: %f, 1/2^n: %f)' 
               %(sortedlu[1]-sortedlu[0], 1/2**n))
    plt.show()
    #fig.savefig('destination_path.eps', format='eps', dpi=1000)

 
