import numpy as np
from scipy.stats import norm
from math import factorial


def blackScholes_calc(paramsList):
    '''
    Calculate call and put pirce by Black-Scholes formulas.
        Parameters:
            paramsList: list -> with length of 5
        Outputs:
            c: float -> call price
            p: float -> put price
    '''
    # Parameters we nned to calculate Black Scholes call & put price
    S0, K, r, q, sigma, T, simCnt, repCnt, n = paramsList
    d1 = (np.log(S0/K) + (r-q+sigma**2/2)*T) / (sigma*(T**0.5))
    d2 = d1 - sigma*(T**0.5)

    # Calcuation of call and put price
    c = S0*(np.exp(-q*T))*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    p = K*np.exp(-r*T)*norm.cdf(-d2) - S0*(np.exp(-q*T))*norm.cdf(-d1)
    print('Black-Scholes formula')
    print('  Call = ', np.round(c, 4))
    print('  Put = ', np.round(p, 4))

def monteCarlo_calc(paramsList):
    '''
    Calculate call and put pirce by Monte Carlo simulation.
        Parameters:
            paramsList: list -> with length of 5
        Outputs:
            c: float -> call price
            p: float -> put price
    '''
    S0, K, r, q, sigma, T, simCnt, repCnt, n = paramsList

    def drawSample(S0, K, r, q, sigma, T):
        mean = np.log(S0) + ((r-q-sigma**2/2)*T)
        std = sigma*(T**0.5)
        lnST_list = np.random.normal(loc=mean, scale=std, size=int(simCnt))
        lnST_list = np.exp(lnST_list) - K

        call_list = np.maximum(lnST_list, 0)*np.exp(-r*T)
        put_list = np.minimum(lnST_list, 0)*(-1)*np.exp(-r*T)
        return np.mean(call_list), np.mean(put_list)

    callResultList = []
    putResultList = []
    for i in range(int(repCnt)):
        c, p = drawSample(S0, K, r, q, sigma, T)
        callResultList.append(c)
        putResultList.append(p)
    print('\nMonte Carlo simulation')
    print(f'  Call upper bound: {np.round(np.mean(callResultList) + 2*np.std(callResultList), 4)}')
    print(f'  Call lower bound: {np.round(np.mean(callResultList) - 2*np.std(callResultList), 4)}')
    print(f'  Put upper bound: {np.round(np.mean(putResultList) + 2*np.std(putResultList), 4)}')
    print(f'  Put lower bound: {np.round(np.mean(putResultList) - 2*np.std(putResultList), 4)}')


def crrBinomialTree_2dTable_calc(paramsList):
    '''
    Calculate call and put pirce by CRR binomial tree method with 2d table.
    '''
    S0, K, r, q, sigma, T, simCnt, repCnt, n = paramsList
    simCnt = int(simCnt)
    repCnt = int(repCnt)
    n = int(n)
    u = np.exp(sigma*((T/n)**0.5))
    d = np.exp(-sigma*((T/n)**0.5))
    p = (np.exp((r-q)*T/n)-d) / (u-d)
    
    uArr = np.ones(n+1)*u**np.arange(n,-1,-1)
    dArr = np.ones(n+1)*d**np.arange(0,n+1)
    sPriceAtTn = S0*uArr*dArr

    # European Call price table calculation
    priceTable = np.zeros((n+1,n+1))
    priceTable[:, n] = np.maximum(sPriceAtTn - K, 0)
    for i in range(n-1, -1, -1):
        previousColSum = np.exp(-r*(T/n))*(p*priceTable[:i+1, i+1] + (1-p)*priceTable[1:i+2, i+1])
        priceTable[:i+1, i] = previousColSum
    europeanCallPrice = priceTable[0,0]
    
    # European Put price table calculation
    priceTable = np.zeros((n+1,n+1))
    priceTable[:, n] = (np.maximum( K - sPriceAtTn, 0))
    for i in range(n-1, -1, -1):
        previousColSum = np.exp(-r*(T/n))*(p*priceTable[:i+1, i+1] + (1-p)*priceTable[1:i+2, i+1])
        priceTable[:i+1, i] = previousColSum
    europeanPutPrice = priceTable[0,0]

    # American Call price table calculation
    priceTable = np.zeros((n+1,n+1))
    priceTable[:, n] = np.maximum(sPriceAtTn - K, 0)
    for i in range(n-1, -1, -1):
        uArr = np.ones(i+1)*u**np.arange(i,-1,-1)
        dArr = np.ones(i+1)*d**np.arange(0,i+1)
        sPriceAtTi = S0*uArr*dArr
        previousColSum = np.maximum(np.exp(-r*(T/n))*(p*priceTable[:i+1, i+1] + (1-p)*priceTable[1:i+2, i+1]), sPriceAtTi-K)
        priceTable[:i+1, i] = previousColSum
    americanCallPrice = priceTable[0,0]

    # American Put price table calculation
    priceTable = np.zeros((n+1,n+1))
    priceTable[:, n] = np.maximum(K-sPriceAtTn, 0)
    for i in range(n-1, -1, -1):
        uArr = np.ones(i+1)*u**np.arange(i,-1,-1)
        dArr = np.ones(i+1)*d**np.arange(0,i+1)
        sPriceAtTi = S0*uArr*dArr
        previousColSum = np.maximum(np.exp(-r*(T/n))*(p*priceTable[:i+1, i+1] + (1-p)*priceTable[1:i+2, i+1]), K-sPriceAtTi)
        priceTable[:i+1, i] = previousColSum
    americanPutPrice = priceTable[0,0]
    print('\nCRR binomial tree model using 2d matrix')
    print('  European Call Price: ', np.round(europeanCallPrice, 4))
    print('  European Put Price: ', np.round(europeanPutPrice, 4))
    print('  American Call Price: ', np.round(americanCallPrice, 4))
    print('  American Put Price: ', np.round(americanPutPrice, 4))



def crrBinomialTree_1dTable_calc(paramsList):
    '''
    Calculate european and american of call and put pirce by CRR binomial tree method with 1d table.
    '''
    print('\nCRR binomial tree model using one Column vector')

    S0, K, r, q, sigma, T, simCnt, repCnt, n = paramsList
    n = int(input('new n = '))
    simCnt = int(simCnt)
    repCnt = int(repCnt)
    n = int(n)
    u = np.exp(sigma*((T/n)**0.5))
    d = np.exp(-sigma*((T/n)**0.5))
    p = (np.exp((r-q)*T/n)-d) / (u-d)
    
    uArr = np.ones(n+1)*u**np.arange(n,-1,-1)
    dArr = np.ones(n+1)*d**np.arange(0,n+1)
    sPriceAtTn = S0*uArr*dArr

    # European Call price table calculation
    priceTable = np.maximum(sPriceAtTn - K, 0)
    for i in range(n-1, -1, -1):
        previousColSum = np.exp(-r*(T/n))*(p*priceTable[:i+1] + (1-p)*priceTable[1:i+2])
        priceTable[:i+1] = previousColSum
    europeanCallPrice = priceTable[0]
    
    # European Put price table calculation
    priceTable = np.maximum(K - sPriceAtTn, 0)
    for i in range(n-1, -1, -1):
        previousColSum = np.exp(-r*(T/n))*(p*priceTable[:i+1] + (1-p)*priceTable[1:i+2])
        priceTable[:i+1] = previousColSum
    europeanPutPrice = priceTable[0]

    # American Call price table calculation
    priceTable = np.maximum(sPriceAtTn - K, 0)
    for i in range(n-1, -1, -1):
        uArr = np.ones(i+1)*u**np.arange(i,-1,-1)
        dArr = np.ones(i+1)*d**np.arange(0,i+1)
        sPriceAtTi = S0*uArr*dArr
        previousColSum = np.maximum(np.exp(-r*(T/n))*(p*priceTable[:i+1] + (1-p)*priceTable[1:i+2]), sPriceAtTi-K)
        priceTable[:i+1] = previousColSum
    americanCallPrice = priceTable[0]

    # American Put price table calculation
    priceTable = np.maximum(K-sPriceAtTn, 0)
    for i in range(n-1, -1, -1):
        uArr = np.ones(i+1)*u**np.arange(i,-1,-1)
        dArr = np.ones(i+1)*d**np.arange(0,i+1)
        sPriceAtTi = S0*uArr*dArr
        previousColSum = np.maximum(np.exp(-r*(T/n))*(p*priceTable[:i+1] + (1-p)*priceTable[1:i+2]), K-sPriceAtTi)
        priceTable[:i+1] = previousColSum
    americanPutPrice = priceTable[0]
    
    print('  European Call Price: ', np.round(europeanCallPrice, 4))
    print('  European Put Price: ', np.round(europeanPutPrice, 4))
    print('  American Call Price: ', np.round(americanCallPrice, 4))
    print('  American Put Price: ', np.round(americanPutPrice, 4))

def combinatorial_calc(paramsList):
    print('\nCombinatorial method')
    def nCr(n,r,p):
        ans = np.sum(np.log(np.arange(n, r, -1)))\
              - np.sum(np.log(np.arange(1,n-r+1)))\
              + (n-r)*np.log(p) + r*np.log(1-p)
        return np.exp(ans)

    S0, K, r, q, sigma, T, simCnt, repCnt, n = paramsList
    n = int(input('new n = '))
    simCnt = int(simCnt)
    repCnt = int(repCnt)
    n = int(n)
    u = np.exp(sigma*((T/n)**0.5))
    d = np.exp(-sigma*((T/n)**0.5))
    p = (np.exp((r-q)*T/n)-d) / (u-d)

    # European Call price
    # calcTable is (n+1) * 1 array
    # nCr
    calcTable = np.array(list(map(nCr, np.repeat(n, n+1), np.arange(0, n+1), np.repeat(p, n+1))))
    calcTable *= np.maximum(S0 * (u**np.arange(n, -1, -1)) * (d**np.arange(0, n+1)) - K, 0)  #((p**np.arange(n, -1, -1)) * ((1-p)**np.arange(0, n+1)) \
    europeanCallPrice = np.sum(calcTable) *np.exp(-r*T)
    
    # European Put price
    calcTable = np.array(list(map(nCr, np.repeat(n, n+1), np.arange(0, n+1), np.repeat(p, n+1))))
    calcTable *= np.maximum(K - S0 * (u**np.arange(n, -1, -1)) * (d**np.arange(0, n+1)), 0)
    europeanPutPrice = np.sum(calcTable) *np.exp(-r*T)
    
    print('  European Call Price: ', np.round(europeanCallPrice, 4))
    print('  European Put Price: ', np.round(europeanPutPrice, 4))


if __name__ == '__main__':
    # S0, K, r, q, σ, T, simCnt, repCnt, n
    paramsList = [float(i) for i in input('S0, K, r, q, σ, T, simCnt, repCnt, n = ').split()]
    blackScholes_calc(paramsList)
    monteCarlo_calc(paramsList)
    try:
        # May crash when n is too large
        crrBinomialTree_2dTable_calc(paramsList)
    except:
        pass
    
    crrBinomialTree_1dTable_calc(paramsList)

    combinatorial_calc(paramsList)


import numpy as np

a = np.array([[1,-2,3]])
np.maximum(a,0)
