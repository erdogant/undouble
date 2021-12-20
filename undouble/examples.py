# https://content-blockchain.org/research/testing-different-image-hash-functions/

# %%
import undouble
print(dir(undouble))
print(undouble.__version__)

# %%
from undouble import Undouble
model = Undouble(targetdir='D://magweg1//2020', grayscale=True)
model.preprocessing()

# %%
model.fit(method='phash')
model.find(score=0)
model.plot()

# %%
model.move()

# %%
model.results
