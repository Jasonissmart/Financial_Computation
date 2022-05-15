import numpy as np
from scipy.stats import norm

def closed_form_sol(S0, r, q, sigma, T, K1, K2, K3, K4)-> float: 
    '''
    Given 9 parameters, calculate and return the closed form option value
        Input:
            S0: stock price in time 0
            r: annual risk free rate
            q: annual divend rate
            sigma: expected annual standard deviation
            T: time to maturity
            K1, K2, K3, K4: strike prices
        Output:
            option value -> float
    '''
    # parameters preprocess
    ln_S0 = np.log(S0)
    ln_K1 = np.log(K1)
    ln_K2 = np.log(K2)
    ln_K3 = np.log(K3)
    ln_K4 = np.log(K4)
    residual_underR = (r - q + sigma**2 * 0.5) * T
    residual_underQ = (r - q - sigma**2 * 0.5) * T
    d_11 = (ln_S0 - ln_K1 + residual_underR) / (sigma * (T**0.5))
    d_12 = (ln_S0 - ln_K2 + residual_underR) / (sigma * (T**0.5))
    d_21 = (ln_S0 - ln_K1 + residual_underQ) / (sigma * (T**0.5))
    d_22 = (ln_S0 - ln_K2 + residual_underQ) / (sigma * (T**0.5))
    d_31 = (ln_S0 - ln_K2 + residual_underQ) / (sigma * (T**0.5))
    d_32 = (ln_S0 - ln_K3 + residual_underQ) / (sigma * (T**0.5))
    d_41 = (ln_S0 - ln_K3 + residual_underQ) / (sigma * (T**0.5))
    d_42 = (ln_S0 - ln_K4 + residual_underQ) / (sigma * (T**0.5))
    d_51 = (ln_S0 - ln_K3 + residual_underR) / (sigma * (T**0.5))
    d_52 = (ln_S0 - ln_K4 + residual_underR) / (sigma * (T**0.5))


    row_1 = S0 * np.exp((r-q) * T) * (norm.cdf(d_11) - norm.cdf(d_12))
    row_2 = K1 * (norm.cdf(d_21) - norm.cdf(d_22))
    row_3 = (K2 - K1) * (norm.cdf(d_31) - norm.cdf(d_32))
    row_4 = (K2 - K1) / (K4 - K3) * (K4 * (norm.cdf(d_41) - norm.cdf(d_42)))
    row_5 = (K2 - K1) / (K4 - K3) * (S0 * np.exp((r-q) * T) * ( norm.cdf(d_51) - norm.cdf(d_52)))
    optionValue = np.exp(-r * T) * (row_1 - row_2 + row_3 + row_4 -row_5)
    print(f'Option value: {optionValue}') 

def monteCarlo(S0, r, q, sigma, T, K1, K2, K3, K4):
    '''
    Given 9 parameters, using monte carlo to simulate stock prices for 10,000 times,
    then calculate mean and repeat above actions for 20 times, then it will return the upper bound and lower bound.
        Input:
            S0: stock price in time 0
            r: annual risk free rate
            q: annual divend rate
            sigma: expected annual standard deviation
            T: time to maturity
            K1, K2, K3, K4: strike prices
        Output:
            upper bound-> float
            lower bound-> float
    '''
    def drawSample(S0, r, q, sigma, T, K1, K2, K3, K4):
        mean = np.log(S0) + ((r - q - sigma**2/2) * T)
        std = sigma * (T**0.5)
        lnST_list = np.random.normal(loc=mean, scale=std, size=10000)
        lnST_list = np.exp(lnST_list)
        lnST_list[(lnST_list <= K1) | (lnST_list >= K4)] = 0
        lnST_list[(lnST_list > K1) & (lnST_list <= K2)] -= K1
        lnST_list[(lnST_list > K2) & (lnST_list <= K3)] = K2 - K1
        lnST_list[(lnST_list > K3) & (lnST_list < K4)] = (K2-K1)/(K4-K3)*(K4 - lnST_list[(lnST_list > K3) & (lnST_list < K4)])
        return np.mean(lnST_list)

    resultList = []
    for i in range(20):
        resultList.append(drawSample(S0, r, q, sigma, T, K1, K2, K3, K4))
    #print(f'mean: {np.mean(resultList)}')
    print(f'upper bound: {np.mean(resultList) + 2*np.std(resultList)}')
    print(f'lower bound: {np.mean(resultList) - 2*np.std(resultList)}')


if __name__ == "__main__":
    S0, r, q, sigma, T, K1, K2, K3, K4 = [float(i) for i in input().split()]
    closed_form_sol(S0, r, q, sigma, T, K1, K2, K3, K4)
    monteCarlo(S0, r, q, sigma, T, K1, K2, K3, K4)