# https://content-blockchain.org/research/testing-different-image-hash-functions/

# %%
from undouble import Undouble
# import undouble
# print(dir(undouble))
# print(undouble.__version__)

# %%

targetdir='//NAS_SYNOLOGY//Photo//2016'
targetdir='D://magweg//101_ObjectCategories//bonsai'

# %%
model = Undouble(method='phash', grayscale=True)
model.preprocessing(targetdir)
model.fit()

# %%
model.find(score=5)

# %%
model.plot()

# %%
model.move()

# %%
model.results

# %%
