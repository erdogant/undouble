from undouble import Undouble
import itertools as it
import numpy as np
import unittest
from tqdm import tqdm

class TestUNDOUBLE(unittest.TestCase):

    def test_import_data(self):
        model = Undouble()
        # Import flowers example
        X = model.import_example(data='flowers')

        # Check numpy array imports
        model.import_data(X)
        assert model.results['img'].shape==(214, 128, 128, 4)
        assert len(model.results['pathnames'])==214
        assert len(model.results['filenames'])==214
        assert set(model.results.keys())==set(['img', 'feat', 'pathnames', 'filenames', 'url'])

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

        for combination in tqdm(combinations):
            model = Undouble(method=combination[0], grayscale=combination[1], hash_size=combination[2], dim=combination[3], verbose=40)
            # Import data
            model.import_data(X, return_results=False)
            # Compute Hash
            assert model.compute_hash(return_dict=True)

