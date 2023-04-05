import numpy as np

# Build the tree
class BinomialTree:
    def __init__(self, paramsList):
        self.St, self.r, self.q, self.sigma, self.t, self.T, self.n = paramsList
        self.n = int(self.n)
        self.deltaT = self.T - self.t

        self.u = np.exp(self.sigma * ((self.deltaT / self.n) ** 0.5))
        self.d = np.exp(-self.sigma * ((self.deltaT / self.n) ** 0.5))
        self.miu = np.exp((self.r - self.q) * (self.deltaT / self.n))
        self.p = (self.miu * self.u - 1) / (self.miu * (self.u - self.d))
        self.tree = np.zeros((self.n + 1, self.n + 1))

    def calc_stockPice(self):
        u = self.u
        d = self.d

        # calculate stock price
        for ithCol in range(0, self.n + 1):
            for jthRow in range(ithCol + 1):
                self.tree[self.n - jthRow, ithCol] = u ** (jthRow)

    def calc_terminal_node_Payoff(self):
        lastColIndex = self.n
        self.tree[:, lastColIndex] = np.maximum(self.tree[:, lastColIndex] - 1, 0)

    def backward_induction(self, type='European'):
        pu = self.p
        pd = 1 - pu
        for ithCol in range(self.n - 1, -1, -1):
            for jthRow in range(ithCol + 1):
                if (jthRow == 0):
                    payoff = pu * self.tree[self.n - jthRow, ithCol + 1] + pd * self.tree[self.n - jthRow - 1, ithCol + 1] 
                    self.tree[self.n - jthRow, ithCol] = payoff * np.exp(-self.r * self.deltaT / self.n) * self.miu
                else:
                    payoff = pu * self.tree[self.n - jthRow + 1, ithCol + 1] + pd * self.tree[self.n - jthRow - 1, ithCol + 1]
                    self.tree[self.n - jthRow, ithCol] = payoff * np.exp(-self.r * self.deltaT / self.n) * self.miu
                if (type != 'European'):
                    self.tree[self.n - jthRow, ithCol] = np.maximum(self.tree[self.n - jthRow, ithCol], self.u ** (jthRow) - 1)
        

    def getPrice(self):
        return np.round(self.St * self.tree[self.n, 0], 4)

def calc_lookbackPutPrice():
    # St, r, q, σ, t, T, Smaxt, n
    paramsList = [float(i) for i in input('# St, r, q, σ, t, T, n = ').split()]

    binomialTree = BinomialTree(paramsList)
    binomialTree.calc_stockPice()
    binomialTree.calc_terminal_node_Payoff()
    binomialTree.backward_induction()
    print('European Lookback Put price: ', binomialTree.getPrice())

    binomialTree = BinomialTree(paramsList)
    binomialTree.calc_stockPice()
    binomialTree.calc_terminal_node_Payoff()
    binomialTree.backward_induction('American')
    print('American Lookback Put price: ', binomialTree.getPrice())


if __name__ == "__main__":
    calc_lookbackPutPrice()