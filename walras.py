"""
 Supply functions are a way of representing agents like AMMs when what we
  want is to find the optimal price.

 Given a price vector, they describe the maximum trade they would clear 
 at that price.  
  "It takes positive values at coordinate `i`  when the agent is selling 
   the `i`-th asset and negative values when buying"

"""

from collections import defaultdict
from math import sqrt
import numpy as np

class ConstantProductAMM:
    # Token A is the numeraire
    def __init__(self, poolA, poolB, A=None, B=None):
        self.poolA = poolA
        self.poolB = poolB
        self.A = A
        self.B = B

    # Returns relative price in units of A per unit of B
    def price(self):
        return self.poolA / self.poolB

    # Returns the overall amount of token B given a positive amtA
    def trade(self, amtA):
        assert self.poolA+amtA >= 0
        amtB = amtA*self.poolB/(amtA + self.poolA)
        return amtB

    # Supply function: given a price dict, returns a trade dict
    # After the resulting trade, the price of the pool will match `price`
    def supply(self, price):
        assert self.A is not None
        assert self.B is not None

        # Look up the relative price `p` for our token pair
        p = price[self.B] / price[self.A]

        # How much will i buy or sell until (poolA-amtA)*(poolB-amtB)
        #   and (poolA-amtA)/(poolB-amtB)=p?
        amtA = self.poolA - sqrt(self.poolA * self.poolB * p)
        amtB = self.poolB - sqrt(self.poolA * self.poolB / p)

        # print('supplyamm():', self.poolA,self.poolB,amtA,amtB)
        assert self.poolA-amtA > 0
        assert self.poolB-amtB > 0
        assert np.isclose((self.poolA-amtA)/(self.poolB-amtB), p)
        assert p >= 0
        assert amtA * amtB <= 0, "either buy or sell, not both"

        # For the supply_function, we want to imagine a virtual agent
        # that takes all the surplus. This is because supply functions
        # are supposed to be net-zero at each price.
        # But since the price has just now reached p, the overall price offered
        # by the AMM in this trade was positive.
        #   This is the step that's necessary to convert between these representations.

        if amtA >= 0: # AMM sells A, receives B
            assert -amtB*p <= amtA or np.isclose(-amtB*p,amtA)
            amtA = -amtB*p
        else: # AMM sells B, receives A
            assert -amtA/p <= amtB or np.isclose(-amtA/p,amtB)
            amtB = -amtA/p
            
        # We use defaultdict so that the amount is zero for every other token
        d = defaultdict(int)
        d[self.A] = amtA
        d[self.B] = amtB
        return d


# This is a one-sided limit order. The convention is a little tricky.
# It only has a reserve of token B, so that's the only amount it sells
#  (therefore the only positive value in the supply function).
# The convention is A is the numeraire, so we're selling Token B
# for a minimum price of _price
class LimitOrder(object):
    def __init__(self, poolB, price, A=None, B=None):
        self.poolB = poolB
        self._price = price
        self.A = A
        self.B = B

    # The limit order accepts any amount it can if the price is better
    def price(self):
        if self.poolB > 0: return self._price

    # Returns the overall amount of token B given a positive amtA
    def trade(self, amtA):
        return max(0, min(amtA / self._price, self.poolB))

    # Supply function: given a price dict, returns a trade dict
    # The resulting trade is at this price overall, so net neutral
    def supply(self, price):
        assert self.A is not None
        assert self.B is not None

        # Look up the relative price `p` for our token pair
        p = price[self.B] / price[self.A]

        # Determine the trade amounts based on the limit order's price
        """there are prices 0 < r1 < r2 for which g(r) = 0 if r ≤ r1
         To approximate a traditional limit sell order, we have to choose 
        r1 close to r2, and take any increasing interpolation (e.g. 
        linear interpolation) between r1 and r2."""
        r1 = self._price
        r2 = self._price * (1+1e-4) # scale by a fraction of a bp
        if p > r2:
            amtB = self.poolB
        elif p > r1:
            amtB = (p - r1) / (r2 - r1) * self.poolB
        else:
            amtB = 0
        amtA = -amtB * p            

        # Return the trade amounts for each token
        d = defaultdict(int)
        d[self.A] = amtA
        d[self.B] = amtB
        return d



###
# Optional tests below
###

def test_constant_product():
    p1 = ConstantProductAMM(1000,1000,'A','B')

    # Trade until price is zero
    price = {'A': 1.0, 'B': 1.0}
    assert p1.supply(price)['A'] == 0

    price['B'] += 0.1
    x = p1.supply(price)
    # print(p1.trade(x['A']), price['A'], x['B'])
    assert p1.trade(x['A']) <= x['A']/price['B']

    price['B'] -= 0.2
    x = p1.supply(price)
    assert p1.trade(x['A']) >= -x['A']/price['B']
    
test_constant_product()


def test_limit_order():
    p2 = LimitOrder(100, 1.0, 'A', 'B')
    
    # Case 1: Market price equal to limit order price
    price = {'A': 1.0, 'B': 1.0}
    x = p2.supply(price)
    assert np.isclose(p2.trade(x['A']), -x['B'])

    # Case 2: Market price above limit order price
    price['B'] = 1.1  # Market price is more favorable
    x = p2.supply(price)
    assert np.isclose(p2.trade(x['B']), -x['A']/1.1)
    
    # Case 3: Market price below limit order price
    price['B'] = 0.9  # Market price is less favorable
    x = p2.supply(price)
    assert x['A'] == 0 and x['B'] == 0  # No trade should occur

test_limit_order()
