# MEV Reductions

A python notebook guided tour through MEV.

- 1. Uniswap
- 2. 

### Requires python3, pyplot
```
pip install -r requirements.txt
python 01-uniswap.py
python 02-batchauction.py
```

### Interactive mode

```
ipython --pylab
run 01-uniswap.py
```

### To render the notebook

```
p2j -o 01-uniswap.py
p2j -o 02-batchauction.py
for x in *.ipynb; do jupyter nbconvert --execute --inplace $x; done
```