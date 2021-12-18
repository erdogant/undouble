# %%
import undouble
print(dir(undouble))
print(undouble.__version__)

# %%
from undouble import Undouble
model = Undouble(verbose=20)
model.fit_transform()

# %%
