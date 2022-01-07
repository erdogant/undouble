# https://content-blockchain.org/research/testing-different-image-hash-functions/

# %%
import undouble
print(dir(undouble))
print(undouble.__version__)

# %%
from undouble import Undouble

model = Undouble(targetdir='//NAS_SYNOLOGY//Photo//2016')
# model = Undouble(targetdir='D://magweg1//2003')
model.preprocessing()

# %%
model.fit(method='phash')
model.find(score=5)

# %%
model.plot()

# %%
model.move()

# %%
model.results

# %%

