import matplotlib.pyplot as plt
import numpy as np
import walras
from importlib import reload
reload(walras)
from walras import ConstantProductAMM, LimitOrder
from functools import reduce
import math
from scipy.optimize import minimize
from collections import defaultdict

# Here's a scenario with just 2 tokens, a stack of limit orders, and 1 AMM
lim1 = LimitOrder(100, 1.01, 'A', 'B')
lim2 = LimitOrder(100, 1.02, 'A', 'B')
lim3 = LimitOrder(100, 1.03, 'A', 'B')
lim4 = LimitOrder(100, 1.01, 'B', 'A')
lim5 = LimitOrder(100, 1.02, 'B', 'A')
lim6 = LimitOrder(100, 1.03, 'B', 'A')
# limit_orders = [lim1, lim2, lim3, lim4, lim5, lim6]
limit_orders = [lim1,lim2,lim3,lim4,lim5,lim6]
price_vector = 1.0

# Here's the AMM, configured by an initial pool amount
poolA=1000
poolB=1100
pool1 = ConstantProductAMM(poolA, poolB, 'A', 'B')

# We can represent the scenario just by the list of supply functions
supply_functions = [lim.supply for lim in limit_orders] + [pool1.supply]

def net_demand(supply_functions, price_vector):
    # Initialize a dictionary to hold the net demand for each token
    net_demand = defaultdict(lambda: 0)

    # For each supply function, calculate the supply and add it to the net demand
    for supply_func in supply_functions:
        supply = supply_func(price_vector)
        for token, amount in supply.items():
            # print('token', token, price_vector[token])
            net_demand[token] += amount

    # The net demand can be negative (net supply) or positive (net demand)
    # print('net_demand', net_demand)
    return net_demand


def find_market_clearing_price_scipy(supply_functions, initial_price_vector):
    # The error vector is the L2 sum of net demand of each asset...
    #    should it matter if this is scaled by relative price? how to normalize?
    def error_vector(price_vector):
        price_dict = {token: price for token, price in zip(initial_price_vector.keys(), price_vector)}
        net_demands = net_demand(supply_functions, price_dict)
        return sum([v*v for v in net_demands.values()])

    # Convert initial price vector to a list for fsolve
    initial_prices = list(initial_price_vector.values())

    # Use scipy to find the price vector that makes net demand zero
    clearing_prices = minimize(error_vector,
                               x0=initial_prices,
                               bounds=2*[(1e-15,None)]) # Prices must be positive
    print('clearing prices:', clearing_prices)
    print('error_vec initial:', error_vector(initial_prices))
    print('error_vec with solution', error_vector(clearing_prices.x))

    # Convert back to a dictionary
    return {token: price for token, price in zip(initial_price_vector.keys(), clearing_prices.x)}


initial_price_vector = {'A': 1.0, 'B': 1.0}
market_clearing_price = find_market_clearing_price_scipy(supply_functions, initial_price_vector)
# market_clearing_price = initial_price_vector
rel_price = market_clearing_price['B'] / market_clearing_price['A']
print("Market Clearing Price:", rel_price)
print('Residual net demand:', net_demand(supply_functions, market_clearing_price))

# Sum a list of supply functions
def sum_supply(supply_funcs):
    def _supply(price_dict):
        d = defaultdict(lambda: 0)
        for f in supply_funcs:
            for k,v in f(price_dict).items():
                d[k] += v
        return d
    return _supply

# We should define with test functions what it means for a supply
# function to be admissible, including continuity, monotone, ..
def test_supply():
    pass

def plot_supply(supply_funcs, my_price=None, title=None):
    plt.figure(3)
    plt.clf()

    A = 'A'
    B = 'B'
    
    # Relative price
    p_values = np.linspace(0.8,1.2,400)

    supply_values = np.zeros((len(supply_funcs),len(p_values)))

    tokens = ['A','B']
    def pricevector(rel_price):
        return {A: 1.0, B: rel_price}
    def eval_supply(supply_func, rel_price):
        
        As = [supply_func(pricevector(p))['A'] for p in rel_price]
        Bs = [supply_func(pricevector(p))['B'] for p in rel_price]
        for a,b,p in zip(As,Bs,rel_price):
            assert np.isclose(a + b*p, 0)
        return As

    for f in supply_funcs:
        ys = eval_supply(f, p_values)

        if 'AMM' in str(type(f)):
            label = 'AMM'
        elif 'LimitOrder' in str(type(f)):
            label = 'Limit Order'
        else: label = None
        plt.plot(p_values, ys, label=label)

    # Price vector as a vertical line
    plt.axvline(x=my_price, color='blue', linestyle='--', label=f'Price: {rel_price}')

    summed = sum_supply(supply_funcs)
    ys = eval_supply(summed, p_values)
    plt.plot(p_values, ys, 'k--', label='Sum')
    plt.ylabel('Amount sold of asset A')
    plt.xlabel('Price')
    plt.title(title)
    plt.legend()
    plt.draw()
    
plot_supply([x.supply for x in [lim1,lim2,lim3,lim4,lim5,lim6,pool1]],
            rel_price, 'Supply functions')


###
# Illustrations
###

"""
These are illustrations of how the individual components, namely limit 
order books and constant product AMMs respond to the proposed clearing price
"""
def plot_depth_chart(limit_orders, rel_price, A='A',B='B'):
    for order in limit_orders:
        assert order.A in (A,B)
        assert order.B in (A,B)
        assert order.A != order.B
    
    # Sort buy orders (buying B with A) and sell orders (selling B for A) based on price
    buy_orders = sorted([order for order in limit_orders if order.A == B], key=lambda x: x.price())
    sell_orders = sorted([order for order in limit_orders if order.A == A], key=lambda x: x.price())

    # Cumulative amounts for buy and sell orders
    cumulative_buy_amount = [0]
    buy_prices = []
    for order in buy_orders:
        amtB = order.poolB*order.price()
        price = 1./order.price()
        cumulative_buy_amount.append(cumulative_buy_amount[-1] + amtB)
        cumulative_buy_amount.append(cumulative_buy_amount[-1])
        buy_prices.append(price)
        buy_prices.append(price)

    cumulative_sell_amount = [0]
    sell_prices = []
    for order in sell_orders:
        amtB = order.poolB
        cumulative_sell_amount.append(cumulative_sell_amount[-1] + amtB)
        cumulative_sell_amount.append(cumulative_sell_amount[-1])
        sell_prices.append(order.price())
        sell_prices.append(order.price())

    cumulative_sell_amount.pop()
    cumulative_buy_amount.pop()

    # Asset nets
    all_prices = []
    asset_nets = []
    for price in sorted(buy_prices + sell_prices):
        all_prices.append(price)
        net = 0
        # Just need the amount of B bought/sold
        for order in buy_orders + sell_orders:
            d = order.supply(dict(A=1.0,B=price))
            net += price * d[A]
        asset_nets.append(net)
    
    # Plotting
    plt.figure(1, figsize=(10, 6))
    plt.clf()

    # Plot for buy orders
    plt.fill_between(buy_prices, 0, cumulative_buy_amount, color='green', alpha=0.5, label='Buys')
    
    # Plot for sell orders
    plt.fill_between(sell_prices, 0, cumulative_sell_amount, color='red', alpha=0.5, label='Sells')

    # Price vector as a vertical line
    plt.axvline(x=rel_price, color='blue', linestyle='--', label=f'Price: {rel_price}')

    # Plot supply function end points
    plt.plot(all_prices, asset_nets, linestyle='--',color='black', label='supply(p)')

    plt.xlabel('Price')
    plt.ylabel('Cumulative Amount')
    plt.title('Depth Chart of Limit Orders')
    plt.legend()
    plt.grid(True)
    plt.draw()

def plot_amm(poolA, poolB, rel_price):

    # Trade vector from the supply function at this price
    pool = ConstantProductAMM(poolA, poolB,'A','B')
    print('ok')
    trade_vec = pool.supply(dict(A=1.,B=rel_price))
    if trade_vec['A'] > 0:
        # The virtual agent sold us some amtA. But we overpaid in trade_vec B.
        # We need to solve backward by price.
        # print('case 1', trade_vec)
        amtB = -trade_vec['B']
        trade_amtA = -amtB*poolA/(amtB + poolB)
    else:
        # print('case 2')
        trade_amtA = -trade_vec['A']
    trade_amtB = -pool.trade(trade_amtA)
    # print('trade_amtA:', trade_amtA)
    # print('trade_amtB:', trade_amtB)
    end_point_A = poolA + trade_amtA
    end_point_B = poolB + trade_amtB
    assert np.isclose((end_point_A)/(end_point_B), rel_price)

    # Generate points for the constant product curve
    # Avoid zero to prevent division by zero
    a_values = np.linspace(0.98 * min(poolA,end_point_A), 1.02*max(poolA,end_point_A), 200)
    b_values = poolA * poolB / a_values

    # Tangent goes through ending point
    #print('rel_price', rel_price)
    #print('end point:', end_point_A / end_point_B)
    tangent_line = end_point_B + (a_values - end_point_A) * -1./rel_price

    # Plotting
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.plot(a_values, b_values, label='Constant Product Curve')
    plt.plot([poolA, end_point_A], [poolB, end_point_B], 'ro-',
             label='Trade vector to reach price')
    plt.plot(a_values, tangent_line, 'b--', label=f'Price {rel_price}')

    plt.xlabel('Amount of Token A')
    plt.ylabel('Amount of Token B')
    plt.title('Automated Market Maker (AMM) Behavior')
    plt.legend()
    plt.grid(True)
    plt.draw()

plot_depth_chart(limit_orders, rel_price)
plot_amm(poolA=poolA, poolB=poolB, rel_price=rel_price)

plt.show()
