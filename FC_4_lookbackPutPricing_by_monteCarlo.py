import numpy as np


def monteCarlo_calc():
    '''
    Calculate call and put pirce by Monte Carlo simulation.
        Parameters:
            paramsList: list -> with length of 5
        Outputs:
            c: float -> call price
            p: float -> put price
    '''
    # St, r, q, σ, t, T, Smaxt, n, simCnt, repCnt
    paramsList = [float(i) for i in input('# St, r, q, σ, t, T, Smaxt, n, simCnt, repCnt = ').split()]
    St, r, q, sigma, t, T, Smaxt, n, simCnt, repCnt = paramsList
    n, simCnt, repCnt = int(n), int(simCnt), int(repCnt)

    def drawSample(St, r, q, sigma, t, T, Smaxt, n, simCnt):
        mean = np.log(St) + ((r-q-sigma**2/2)*((T-t)/n))
        std = sigma*(((T-t)/n)**0.5)
        lnSt_list = np.random.normal(loc=mean, scale=std, size=int(simCnt))

        lnSmax_list = np.maximum(lnSt_list, np.log(Smaxt))
        for i in range(n-1):
            mean = lnSt_list + ((r-q-sigma**2/2)*((T-t)/n))
            lnSt_list = np.random.normal(loc=mean, scale=std)
            lnSmax_list = np.maximum(lnSt_list, lnSmax_list)
        
        Smax_list = np.exp(lnSmax_list)
        St_list = np.exp(lnSt_list)
        payoff_list = np.maximum(Smax_list - St_list, 0)*np.exp(-r*(T-t))
        return np.mean(payoff_list)

    putResultList = []
    for i in range(int(repCnt)):
        p = drawSample(St, r, q, sigma, t, T, Smaxt, n, simCnt)
        putResultList.append(p)
    print('\nMonte Carlo simulation')
    print(f'  Lookback put price upper bound: {np.round(np.mean(putResultList) + 2*np.std(putResultList), 4)}')
    print(f'  Lookback put priceclower bound: {np.round(np.mean(putResultList) - 2*np.std(putResultList), 4)}')


if __name__ == "__main__":
    monteCarlo_calc()
