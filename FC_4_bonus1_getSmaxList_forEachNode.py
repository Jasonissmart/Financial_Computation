import numpy as np


class Node:
    def __init__(self, sPrice):
        self.sPrice = sPrice
        self.possibleSmaxList = []
        self.possiblePutList = []


# Build the tree
class BinomialTree:
    def __init__(self, paramsList):
        self.St, self.r, self.q, self.sigma, self.t, self.T, self.Smaxt, self.n = paramsList
        self.n = int(self.n)
        self.deltaT = self.T - self.t

        self.u = np.exp(self.sigma * ((self.deltaT / self.n) ** 0.5))
        self.d = np.exp(-self.sigma * ((self.deltaT / self.n) ** 0.5))
        self.p = (np.exp((self.r - self.q) * self.deltaT / self.n) - self.d) / (self.u - self.d)

        self.tree = np.zeros((self.n + 1, self.n + 1), dtype=object)

    def calc_stockPice(self):
        u = self.u
        d = self.d

        # calculate stock price
        for ithCol in range(0, self.n + 1):
            for jthRow in range(ithCol + 1):
                self.tree[jthRow, ithCol] = Node(round(self.St * (u ** (ithCol - jthRow)) * (d ** jthRow), 4))

    def get_SmaxList_perNode(self, jthRow:int, ithCol:int):
        SmaxList = []
        SmaxAtTimet = max(self.St, self.Smaxt)
        for k in range(jthRow + 1):
            Smax = np.round(self.St * (self.u ** (ithCol - jthRow - k)), 4)
            if (Smax > SmaxAtTimet):
                SmaxList.append(Smax)
            else:
                SmaxList.append(SmaxAtTimet)
                break
        self.tree[jthRow, ithCol].possibleSmaxList = SmaxList
    
    def get_SmaxList_tree(self):
        for ithCol in range(0, self.n + 1):
            for jthRow in range(ithCol + 1):
                self.get_SmaxList_perNode(jthRow, ithCol)

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

def calc_lookbackPutPrice():
    # St, r, q, σ, t, T, Smaxt, n
    paramsList = [float(i) for i in input('# St, r, q, σ, t, T, Smaxt, n = ').split()]

    binomialTree = BinomialTree(paramsList)
    binomialTree.calc_stockPice()
    binomialTree.get_SmaxList_tree()
    binomialTree.calc_terminal_node_Payoff()
    binomialTree.backward_induction()
    print('European Lookback Put price: ', binomialTree.getPrice())

    binomialTree = BinomialTree(paramsList)
    binomialTree.calc_stockPice()
    binomialTree.get_SmaxList_tree()
    binomialTree.calc_terminal_node_Payoff()
    binomialTree.backward_induction('American')
    print('American Lookback Put price: ', binomialTree.getPrice())


if __name__ == "__main__":
    # test for Smax,t = 50, n = 100
    calc_lookbackPutPrice()
    # test for Smax,t = 60, n = 100
    calc_lookbackPutPrice()
    # test for Smax,t = 70, n = 100
    calc_lookbackPutPrice()