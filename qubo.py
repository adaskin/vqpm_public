'''
you can run with qubo(n) n is the number of parameters
@author: adaskin, 2020
'''
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    pass


def calculate_block_prob(psi,qubit):
    '''
    computes probabilities for the states of a given qubit group.
    psi is a quantum state.
    qubit is the index of the qubit
    the order of the qubits |1,2,3,4,..n>;
    '''
    
    n = len(psi)

    logn = int( np.log2(n));   

    f = np.zeros(2);

    for j in range(n):
        
        ind = 0;
        index = ((j&(1<<logn-qubit)) != 0);
        ind = ind + index;   

        f[ind] = f[ind]+np.real(np.abs(psi[j])**2);
    return f; 

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
    print('qubit states:')
    for q in range(1,n+1):
        if q in qStates:
            stateQ = qStates[q]
        else:
            stateQ = calculate_block_prob(outVec.T[0], q)
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
      
        print(stateQ)
        state = np.kron(state, stateQ)    
    return state,qStates  
    
          

def qpmForQubo2(u, n, m, imin):
    '''
    variational quantum poer method for diagonal matrix (measures the state at each iteration)
    u: vector with the diagonal elements of U
    n: number of qubits
    m: number of max iterations
    imin: index of the minimum eigenphase (to follow probability change in each iteration)
    -----------------
    returns final state (probs etc) and final states of the qubits
    '''
    Pimin = np.zeros((m,1), float);
    N = 2 ** n;
    s2 = 1 / np.sqrt(2); 
    inVec = np.ones((N, 1), dtype=complex) / np.sqrt(N); 
    psi2 = np.zeros((N, 1), dtype=complex);
    nm = m;

    qStates={}    
    for j in range(0, m):
        
       
        # print('norm:',np.linalg.norm(inVec))
        # HoI*initial
        psi1 = inVec * s2;
        # print(np.abs(u))
        # print(np.abs(psi1))
        # apply CU
        for k in range(0, N):
            psi2[k] = u[k] * psi1[k];
        
        # print('psi2:') 
        # print(np.abs(psi2))
        # apply Hadamard
        psi11 = s2 * psi1 + psi2 * s2;
        # psi21 = s2 * psi1 - s2 * psi2;
        
        p0 = np.linalg.norm(psi11);
        # p1 = np.linalg.norm(psi21);
        
        inVec = psi11 / p0;        
        
        Pimin[j] = np.abs(inVec[imin][0]) ** 2;
        print("iter: %d probOof1stqubit: %f probofMinInCollapsedState: %lf" % (j, p0, Pimin[j]));
        if (Pimin[j] >= 0.5):
            nm = j + 1;
            break;
        inVec,qStates = prepareNewState(inVec,n,qStates)
        inVec = np.array([inVec]).T     
       
    psi21 = s2 * psi1 - s2 * psi2;
    final_probs = np.power(np.abs(psi11), 2) + np.power(np.abs(psi21), 2);
    y = [np.argmax(final_probs), np.max(final_probs), p0, nm, Pimin];
    
    return y,qStates;


def unitaryForQubo(n, c, Q):
    '''
    creates a vector which represents the diagonal of U
    '''    

    N = 2 ** n;
    u = [1];
    # \sum cixi
    for i in range(0, n):
        ui = [1, np.exp(1j * c[i] )];
        u = np.kron(u, ui);
    
    # \sum hij xixj
    for i in range(0, n - 1):
        for j in range(i + 1, n - 1):
            qij = np.exp(1j * Q[i][j] );
            
            for k in range(0, N - 1):
                b1 = bin(k)[2:];
                b1 = b1.zfill(n);
                if  (b1[i] == 1) and (b1[j] == 1):
                    u[k] = u[k] * qij;
    return u


def randomQ(n):
    '''
    for agiven n parameters it generates a random coefficient matrix Q
    its diagonal should be considered as c
    '''
    Q = np.random.randn(n, n); Q = Q + np.transpose(Q);
    maxQ = np.sum(np.sum(np.abs(np.tril(Q, 0))));
    Q = Q / (maxQ); 
    return Q;


def qubo(n): 
    '''
    generates random instance of qubo and applies vqpm
    '''
    Q = randomQ(n)*np.pi/4; #the sum in [-pi/4, pi/4]
    c = np.diagonal(Q);
    u = unitaryForQubo(n, c, Q);
    u = np.exp(1j * np.pi/4) * u;  # the sum in [0, pi/2] always positive
    
    
    
    lu = np.real(np.log(u) / (1j));
      
    minlu = np.min(lu);
    iminlu = np.argmin(lu);
    
    nm,qStates = qpmForQubo2(u, n, 50, iminlu);
    print("-------------------------------------")
    print("Eigenvalues in [%f, %f]" % (np.min(lu), np.max(lu)))
    print("Expected and Found Eigenvalues")
    
    print("expected:%f state:%d" % (minlu, iminlu))
    print("found state:%d with prob:%f" % (nm[0], nm[1]))
    sortedlu = np.sort(np.abs(lu));
    print("eigengap: ", sortedlu[1]-sortedlu[0])
    #print(sortedlu)
    
    fig = plt.figure()
    plt.plot(nm[4][0:nm[3]],'b.')
    plt.ylabel('iteration')
    
    plt.xlabel('the success probability (the eigengap: %f, 1/2^n: %f)' 
               %(sortedlu[1]-sortedlu[0], 1/2**n))
    plt.show()
    #fig.savefig('fig.eps', format='eps', dpi=1000)
