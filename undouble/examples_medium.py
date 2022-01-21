# from clustimage import Clustimage
# from undouble import Undouble
# import numpy as np

# %%
# Import library
from undouble import Undouble
# Initialize model
model = Undouble(method='phash', hash_size=8)
# Import example data
targetdir = 'D://magweg/101_ObjectCategories'
# targetdir = model.import_example(data='flowers')
# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)
# Compute image-hash
model.fit_transform()

model.group(threshold=0)


# %%

# Photo by Louis-Philippe Poitras on Unsplash
targetdir='//NAS_SYNOLOGY//Mijn Documenten//Erdogan//Medium_blogs//undouble//figs'
model = Undouble(method='ahash', hash_size=8)
model.import_data(targetdir)
# model.fit_transform()


# %% Compare jfif vs jpeg
source_jfif='//NAS_SYNOLOGY/Mijn Documenten/Erdogan/Medium_blogs/undouble/figs/cat_and_dog_source.jfif'
jfif = cl.imread(source_jfif, dim=(128, 128), colorscale=1, flatten=False)
plt.imshow(jfif[..., ::-1])

source_jpg='//NAS_SYNOLOGY/Mijn Documenten/Erdogan/Medium_blogs/undouble/figs/cat_and_dog_source.jpg'
jpg = cl.imread(source_jpg, dim=(128, 128), colorscale=1, flatten=False)
plt.imshow(jpg[..., ::-1])

diff = jpg -jfif
plt.figure()
plt.imshow(diff[..., ::-1])

# %% Compare source (jpg) vs whatsapp (jpg)

source='//NAS_SYNOLOGY/Mijn Documenten/Erdogan/Medium_blogs/undouble/figs/cat_and_dog_source.jpg'
img_source = cl.imread(source, dim=(1024, 1024), colorscale=1, flatten=False)
plt.imshow(img_source[..., ::-1])

whatsapp='//NAS_SYNOLOGY/Mijn Documenten/Erdogan/Medium_blogs/undouble/figs/cat_and_dog_whatsapp_sent.jpg'
img_whatsapp = cl.imread(whatsapp, dim=(1024, 1024), colorscale=1, flatten=False)
plt.imshow(img_whatsapp[..., ::-1])

diff = img_whatsapp -img_source
plt.figure()
plt.imshow(diff[..., ::-1])
