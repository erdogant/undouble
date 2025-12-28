from undouble import Undouble
import itertools as it
import numpy as np
import unittest
from tqdm import tqdm
import cv2
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from undouble import Undouble

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

# Run in terminal: pytest -v

class TestUNDOUBLE(unittest.TestCase):

    def test_blog_example_1(self):
        
        methods = ['ahash', 'dhash', 'whash-haar']
        
        for method in methods:
            # Average Hash
            model = Undouble(method=method, hash_size=8)
            # Import example data
            targetdir = model.import_example(data='cat_and_dog')
            # Grayscaling and scaling
            model.import_data(targetdir)
            # Compute image for only the first image.
            hashs = model.compute_imghash(model.results['img'][0], to_array=False)
            # Compute the image-hash
            image_hash = ''.join(hashs[0].astype(int).astype(str).ravel())
            print(f'{method } Hash:')
            print(f"Binary image hash: {image_hash}")
            print(f"Hex image hash: {(hex(int(image_hash, 2)))}")
        
            # Import image for plotting purposes
            img_g = cv2.imread(model.results['pathnames'][0], cv2.IMREAD_GRAYSCALE)
            img_r = cv2.resize(img_g, (8, 8), interpolation=cv2.INTER_AREA)
        
            # Make the figure
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            ax[0][0].imshow(model.results['img'][0][..., ::-1])
            ax[0][0].axis('off')
            ax[0][0].set_title('Source image')
            ax[0][1].imshow(img_g, cmap='gray')
            ax[0][1].axis('off')
            ax[0][1].set_title('grayscale image')
            ax[1][0].imshow(img_r, cmap='gray')
            ax[1][0].axis('off')
            ax[1][0].set_title('grayscale image, size %.0dx%.0d' %(8, 8))
            ax[1][1].imshow(hashs[0], cmap='gray')
            ax[1][1].axis('off')
            ax[1][1].set_title(method + ' function')
            
            
    def test_import_data(self):
        model = Undouble()
        # Import flowers example
        X = model.import_example(data='flowers')

        # Check numpy array imports
        model.import_data(X)
        # assert model.results['img'].shape==(214, 128, 128, 4)
        assert set(model.results.keys())==set(['img', 'pathnames', 'url', 'filenames', 'img_hash_bin', 'img_hash_hex', 'adjmat', 'select_pathnames', 'select_scores', 'select_idx', 'stats'])

    def test_compute_imghash(self):
        model = Undouble()
        # Import flowers example
        X = model.import_example(data='flowers')
        imgs = model.import_data(X, return_results=True)

        hash_sizes=[4,8,16]
        for hash_size in hash_sizes:
            hashs = model.compute_imghash(imgs['img'][0], to_array=True, hash_size=hash_size)
            assert len(hashs[0])==(hash_size*hash_size)

        hashs = model.compute_imghash(imgs['img'][0:5], to_array=True, hash_size=8)
        assert len(hashs)==5
        assert hashs[0].shape==(64,)
        hashs = model.compute_imghash(imgs['img'][0:5], to_array=False, hash_size=8)
        assert len(hashs)==5
        assert hashs[0].shape==(8,8)

        hashs = model.compute_imghash(imgs['img'][0], to_array=True, hash_size=8)
        assert len(hashs)==1
        assert hashs[0].shape==(64,)
        hashs = model.compute_imghash(imgs['img'][0], to_array=False, hash_size=8)
        assert len(hashs)==1
        assert hashs[0].shape==(8,8)

    def test_compute_hash(self):
        model = Undouble(method='phash')
        # Import flowers example
        X = model.import_example(data='flowers')
        # Import data
        model.import_data(X, return_results=False)
        # Compute Hash
        model.compute_hash()
        assert set(model.results.keys())==set(['img', 'url', 'pathnames', 'filenames', 'img_hash_bin', 'img_hash_hex', 'adjmat'])

        param_grid = {
        	'method': ['ahash','phash','dhash','whash-haar','crop-resistant-hash'],
        	'grayscale':[True, False],
        	'hash_size' : [4, 8, 16],
            'dim' : [(64,64), (128,128), (256,256)]
        	}

        allNames = param_grid.keys()
        combinations = it.product(*(param_grid[Name] for Name in allNames))
        combinations=list(combinations)

        for combination in combinations:
            model = Undouble(method=combination[0], grayscale=combination[1], hash_size=combination[2], dim=combination[3], verbose=40)
            # Import data
            model.import_data(X, return_results=False)
            # Compute Hash
            assert model.compute_hash(return_dict=True)

