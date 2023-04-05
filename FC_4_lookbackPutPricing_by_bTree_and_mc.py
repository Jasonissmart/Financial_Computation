import numpy as np


class Node:
    def __init__(self, sPrice):
        self.sPrice = sPrice
        self.possibleSmaxList = []
        self.possiblePutList = []


# Build the tree
class BinomialTree:
    '''
    Build binomial tree to calculate european and american lookback put option price.

    '''
    def __init__(self, paramsList):
        '''
        Initialize parameters and create an empty tree which will store node information.
        '''
        self.St, self.r, self.q, self.sigma, self.t, self.T, self.Smaxt, self.n = paramsList
        self.n = int(self.n)
        self.deltaT = self.T - self.t

        self.u = np.exp(self.sigma * ((self.deltaT / self.n) ** 0.5))
        self.d = np.exp(-self.sigma * ((self.deltaT / self.n) ** 0.5))
        self.p = (np.exp((self.r - self.q) * self.deltaT / self.n) - self.d) / (self.u - self.d)

        self.tree = np.zeros((self.n + 1, self.n + 1), dtype=object)

    def calc_stockPice(self):
        '''
        Calculate stock price for every node on the tree.
        '''
        u = self.u
        d = self.d

        # calculate stock price
        for ithCol in range(0, self.n + 1):
            for jthRow in range(ithCol + 1):
                self.tree[jthRow, ithCol] = Node(round(self.St * (u ** (ithCol - jthRow)) * (d ** jthRow), 4))

    def record_possible_Smax(self):
        self.tree[0, 0].possibleSmaxList.append(self.Smaxt)
        for ithCol in range(1, self.n + 1):
            for jthRow in range(ithCol + 1):
                currentNode = self.tree[jthRow, ithCol]
                # currentNode.possibleSmaxList.append(self.Smaxt)
                # The first node in that col has only one parent and its smax will be its sPrice
                if(jthRow == 0):
                    if(currentNode.sPrice > self.tree[jthRow, ithCol - 1].possibleSmaxList[0]):
                        currentNode.possibleSmaxList.append(currentNode.sPrice)
                    else:
                        currentNode.possibleSmaxList.append(self.tree[jthRow, ithCol - 1].possibleSmaxList[0])

                # The last node in that col has only one parent and its smax will be St
                elif(jthRow == ithCol):
                    if(currentNode.sPrice > self.tree[jthRow - 1, ithCol - 1].possibleSmaxList[0]):
                        currentNode.possibleSmaxList.append(currentNode.sPrice)
                    else:
                        currentNode.possibleSmaxList.append(self.tree[jthRow-1, ithCol-1].possibleSmaxList[0])

                else:
                    # The upper parent
                    for parentSmax in self.tree[jthRow-1, ithCol-1].possibleSmaxList:
                        if(currentNode.sPrice >= parentSmax):
                            currentNode.possibleSmaxList.append(currentNode.sPrice)
                        elif(currentNode.sPrice < parentSmax):
                            currentNode.possibleSmaxList.append(parentSmax)
                    # The lower parent
                    for parentSmax in self.tree[jthRow, ithCol - 1].possibleSmaxList:
                        if(currentNode.sPrice >= parentSmax):
                            currentNode.possibleSmaxList.append(currentNode.sPrice)
                        elif(currentNode.sPrice < parentSmax):
                            currentNode.possibleSmaxList.append(parentSmax)
                # Remove duplicates in  possibleSmaxList
                currentNode.possibleSmaxList = sorted(list(set(currentNode.possibleSmaxList)), reverse=True)

    def calc_terminal_node_Payoff(self):
        lastColIndex = self.n
        for jthRow in range(lastColIndex + 1):
            # Map each element in possibleSmaxList to possiblePutList
            currentNode = self.tree[jthRow, lastColIndex]
            for i in range(len(currentNode.possibleSmaxList)):
                currentNode.possiblePutList.append(currentNode.possibleSmaxList[i] - currentNode.sPrice)

    def backward_induction(self, type='European'):
        p = self.p

        for ithCol in range(self.n - 1, -1, -1):
            for jthRow in range(ithCol + 1):
                # Process each node
                currentNode = self.tree[jthRow, ithCol]
                put_upperChildNode = -1
                put_lowerChildNode = -1

                upperChildNode = self.tree[jthRow, ithCol + 1]
                lowerChildNode = self.tree[jthRow + 1, ithCol + 1]
                # Every Smax
                for i in range(len(currentNode.possibleSmaxList)):
                    # Match every element
                    # The upper child
                    match = 0
                    for j in range(len(upperChildNode.possibleSmaxList)):
                        if(currentNode.possibleSmaxList[i] == upperChildNode.possibleSmaxList[j]):
                            put_upperChildNode = upperChildNode.possiblePutList[j]
                            match = 1
                            break
                    # If it doesnt match
                    if(match == 0):
                        for j in range(len(upperChildNode.possibleSmaxList)):
                            if(upperChildNode.possibleSmaxList[j] - round(currentNode.sPrice*self.u, 4) < 0.001):
                                put_upperChildNode = upperChildNode.possiblePutList[j]
                                match = 1
                                break

                    # The lower child
                    for j in range(len(lowerChildNode.possibleSmaxList)):
                        if(currentNode.possibleSmaxList[i] == lowerChildNode.possibleSmaxList[j]):
                            put_lowerChildNode = lowerChildNode.possiblePutList[j]
                            break

                    currentNodePut = np.exp((-1) * self.r * (self.deltaT / self.n)) * (p * put_upperChildNode + (1 - p) * put_lowerChildNode)

                    if(type == 'American'):
                        # Early exercise or not
                        if(currentNodePut < (currentNode.possibleSmaxList[i] - currentNode.sPrice)):
                            currentNodePut = (currentNode.possibleSmaxList[i] - currentNode.sPrice)
                    currentNode.possiblePutList.append(currentNodePut)

    def getPrice(self):
        return np.round(self.tree[0, 0].possiblePutList[0], 4)


def monteCarlo_calc():
    '''
    Calculate call and put pirce by Monte Carlo simulation.
        Parameters:
            paramsList: list
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
    print('Monte Carlo simulation')
    print(f'  Lookback put price upper bound: {np.round(np.mean(putResultList) + 2*np.std(putResultList), 4)}')
    print(f'  Lookback put priceclower bound: {np.round(np.mean(putResultList) - 2*np.std(putResultList), 4)}', '\n')


def calc_lookbackPutPrice():
    '''
    Calculate european and american lookback put option price and print them out.
    For each kind of put, it will 
    1. initialize a BinomialTree class
    2. calculate stock price for every node on the tree
    3. record possible Smax for every node on the tree
    4. calculate the terminal nodes' payoff
    5. implement backward induction for every node one the tree
    6. print the payoff at tree[0, 0]
        Parameters:
            St, r, q, σ, t, T, Smaxt, n
    '''
    # St, r, q, σ, t, T, Smaxt, n
    paramsList = [float(i) for i in input('# St, r, q, σ, t, T, Smaxt, n = ').split()]

    binomialTree = BinomialTree(paramsList)
    binomialTree.calc_stockPice()
    binomialTree.record_possible_Smax()
    binomialTree.calc_terminal_node_Payoff()
    binomialTree.backward_induction()
    print('European Lookback Put price: ', binomialTree.getPrice())

    binomialTree = BinomialTree(paramsList)
    binomialTree.calc_stockPice()
    binomialTree.record_possible_Smax()
    binomialTree.calc_terminal_node_Payoff()
    binomialTree.backward_induction('American')
    print('American Lookback Put price: ', binomialTree.getPrice())
    

if __name__ == "__main__":
    # test for Smax,t = 50, n = 100
    calc_lookbackPutPrice()
    # test for Smax,t = 50, n = 300
    calc_lookbackPutPrice()
    monteCarlo_calc()

    # test for Smax,t = 60, n = 100
    calc_lookbackPutPrice()
    # test for Smax,t = 60, n = 300
    calc_lookbackPutPrice()
    monteCarlo_calc()

    # test for Smax,t = 70, n = 100
    calc_lookbackPutPrice()
    # test for Smax,t = 70, n = 300
    calc_lookbackPutPrice()
    monteCarlo_calc()

