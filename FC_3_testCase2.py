import numpy as np


def cholesky_decomposition(c):
    '''
        Implement the cholesky decomposition method to get a matrix.
            Parameters:
                c: 2d matrix
            Output:
                a: 2d matrix
    '''
    # Initialize
    rowCnt = c.shape[0]
    a = np.zeros((rowCnt,rowCnt))
    
    # Step1
    a[0,0] = c[0,0]**0.5
    a[0, 1:] = c[0, 1:]/a[0,0]
    
    # Step2 
    for i in range(1,rowCnt-1):
        a[i,i] = (c[i,i] - np.sum(np.power(a[:i,i],2)))**0.5
        for j in range(i+1, rowCnt):
            a[i, j] = a[i, i]**(-1) * (c[i, j] - np.sum(a[:i,i]*a[:i,j]))
    # Step4
    a[rowCnt-1, rowCnt-1] = (c[rowCnt-1, rowCnt-1] - np.sum(np.power(a[:rowCnt-1, rowCnt-1], 2)))**0.5

    return a


def maximum_rainbow_option_calcultaion(
    K, r, T, simulationCnt, repetitionCnt, 
    n, sPriceAtTime0, qList, sigmaList, AMatrix):
    meanPayoffList = []

    np.random.seed(0)
    for i in range(repetitionCnt):
        randomMatrix = np.random.normal(loc=0, scale=1, size=(simulationCnt, n))    # simulationCnt*n
        rMatrix = np.matmul(randomMatrix, AMatrix)                                  # simulationCnt*n

        sPriceAtTimeT = np.exp(np.add(rMatrix, np.log(sPriceAtTime0) + (np.subtract(r, qList)-(sigmaList**2)/2)*T))
        payoffList = np.maximum(np.max(sPriceAtTimeT, axis=1) - K, 0)
        meanPayoffList.append(np.mean(payoffList)*np.exp(-r*T))
    print('Maximum rainbow option calcultaion')
    print(f'  upper bound: {np.round(np.mean(meanPayoffList) + 2*np.std(meanPayoffList), 4)}')
    print(f'  lower bound: {np.round(np.mean(meanPayoffList) - 2*np.std(meanPayoffList), 4)}')

def maximum_rainbow_option_calcultaion_with_AVA_and_MMM(
    K, r, T, simulationCnt, repetitionCnt, 
    n, sPriceAtTime0, qList, sigmaList, AMatrix):
    '''
    Combine the antithetic variate approach and moment matching method to price the above rainbow option.
    '''
    def generate_randomNum_by_AVA_and_MMM(simulationCnt, n):
        # AVA
        randomUpperMatrix = np.random.normal(loc=0, scale=1, size=(int(simulationCnt/2), n))    # simulationCnt/2 * n
        randomLowerMatrix = np.multiply(randomUpperMatrix, -1)
        randomMatrix = np.vstack((randomUpperMatrix, randomLowerMatrix))
        # MMM
        stdev = np.std(randomMatrix, axis=0)  # 1*n  
        randomMatrix = np.divide(randomMatrix, stdev)
        #print(np.mean(randomMatrix, axis=0), np.std(randomMatrix, axis=0))
        return randomMatrix

    meanPayoffList = []
    np.random.seed(0)
    for i in range(repetitionCnt):
        randomMatrix = generate_randomNum_by_AVA_and_MMM(simulationCnt, n)          # simulationCnt*n
        rMatrix = np.matmul(randomMatrix, AMatrix)                                  # simulationCnt*n

        sPriceAtTimeT = np.exp(np.add(rMatrix, np.log(sPriceAtTime0) + (np.subtract(r, qList)-(sigmaList**2)/2)*T))
        payoffList = np.maximum(np.max(sPriceAtTimeT, axis=1) - K, 0)
        meanPayoffList.append(np.mean(payoffList)*np.exp(-r*T))
    print('\nMaximum rainbow option calcultaion by AVA and MMM')
    print(f'  upper bound: {np.round(np.mean(meanPayoffList) + 2*np.std(meanPayoffList), 4)}')
    print(f'  lower bound: {np.round(np.mean(meanPayoffList) - 2*np.std(meanPayoffList), 4)}')

if __name__ == '__main__':
    K, r, T, simulationCnt, repetitionCnt, n = 100, 0.1, 0.5, 10000, 20, 2

    sPriceAtTime0 = [95]*2
    qList = [0.05]*2
    sigmaList = [0.5]*2

    sPriceAtTime0 = np.array(sPriceAtTime0)
    qList = np.array(qList)
    sigmaList = np.array(sigmaList)

    coefficientMatrix = [[1, -1],
                        [-1, 1]]
    coefficientMatrix = np.array(coefficientMatrix)

    # Construct c matrix
    c = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            c[i,j] = sigmaList[i]*sigmaList[j]*T*coefficientMatrix[i,j]

    AMatrix = cholesky_decomposition(c)


    maximum_rainbow_option_calcultaion(
    K, r, T, simulationCnt, repetitionCnt, 
    n, sPriceAtTime0, qList, sigmaList, AMatrix)

    maximum_rainbow_option_calcultaion_with_AVA_and_MMM(
    K, r, T, simulationCnt, repetitionCnt, 
    n, sPriceAtTime0, qList, sigmaList, AMatrix)

