# %%
# import undouble
# print(dir(undouble))
# print(undouble.__version__)

# %%
from undouble import Undouble
model = Undouble(targetdir='D://magweg1//2020', grayscale=True)
model.preprocessing()

# %%
model.fit(method='phash')
model.find(score=10)
model.plot()
