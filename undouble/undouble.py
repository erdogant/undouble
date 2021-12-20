# --------------------------------------------------
# Name        : undouble.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/undouble
# Licence     : See licences
# --------------------------------------------------

import os
import pandas as pd
import requests
from urllib.parse import urlparse
import logging
import numpy as np
from tqdm import tqdm
import zipfile
from clustimage import Clustimage
import clustimage.clustimage as cl

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
# formatter = logging.Formatter('[%(asctime)s] [undouble]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('[undouble] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Undouble():
    """Undouble your photo collection.

    Description
    -----------
    The aim of this library is to undouble your photo collection.
    The following steps are taken:
        1. Read recursively all images from directory with the specified extensions.
        2. 

    Parameters
    ----------
    method : str, (default: 'pca')
        Method to extract features from images.
            hashmethod : str (default: 'ahash')
            * 'ahash': Average hash
            * 'phash': Perceptual hash
            * 'dhash': Difference hash
            * 'whash-haar': Haar wavelet hash
            * 'whash-db4': Daubechies wavelet hash
            * 'colorhash': HSV color hash
            * 'crop-resistant': Crop-resistant hash
    targetdir : str, (default: None)
        Directory to read the images.
    ext : list, (default: ['png','tiff','jpg'])
        Images with the file extentions are used.
    grayscale : Bool, (default: False)
        Colorscaling the image to gray. This can be usefull when clustering e.g., faces.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    verbose : int, (default: 20)
        Print progress to screen. The default is 20.
        10:Debug, 20:Info, 30:Warn 40:Error, 60:None, 

    Returns
    -------
    Object.
    dict containing keys:
        pathnames : list of str.
            Full path to images that are used in the model.
        filenames : list of str.
            Filename of the input images.

    Example
    -------
    >>> from undouble import Undouble
    >>>
    >>> # Init with default settings
    >>> model = Undouble(method='ahash')
    >>>
    >>> # load example with faces
    >>> X = cl.import_example(data='mnist')
    >>>
    >>> # Cluster digits
    >>> results = cl.fit_transform(X)
    >>>
    >>> # Find images
    >>> results_find = cl.find(X[0,:], k=None, alpha=0.05)
    >>> cl.plot_find()
    >>> cl.scatter()
    >>>
    
    References
    ----------
    * https://content-blockchain.org/research/testing-different-image-hash-functions/

    """
    def __init__(self, method='phash', targetdir='', grayscale=False, dim=(128,128), ext=['png','tiff','jpg'], verbose=20):
        """Initialize undouble with user-defined parameters."""
        # Clean readily fitted models to ensure correct results
        self.clean()
        # Check existence targetdir
        if not os.path.isdir(targetdir): raise Exception(logger.error('Input parameter <targetdir> does not contain a valid directory: [%s]' %(targetdir)) )
        if verbose<=0: verbose=60
        # Store user setting in params
        self.params = {'method':method, 'targetdir':targetdir, 'grayscale':grayscale, 'dim':dim, 'ext':ext, 'verbose':verbose}
        # Initialize the clustimage library
        self.clustimage = Clustimage(method=self.params['method'], grayscale=self.params['grayscale'], ext=self.params['ext'], dim=self.params['dim'], verbose=self.params['verbose'])
        # Set the logger
        set_logger(verbose=verbose)

    def preprocessing(self):
        """All in one function."""
        logger.info("Retrieving files from: [%s]" %(self.params['targetdir']))
        # Preprocessing the images the get them in the right scale and size etc
        self.clustimage.import_data(self.params['targetdir'])

    def fit(self, method='phash'):
        self.clustimage.params['method'] = method
        self.params['method'] = method
        self.clustimage.params_hash = cl.hash_method(method, {})
        # Extract features using method
        self.clustimage.extract_feat(self.clustimage.results)

    def find(self, score=0):
        # Make sets of images that are similar based on the minimum defined score.
        pathnames=[]
        out = []
        scores = []
        # Extract similar images with minimum score
        for i in tqdm(np.arange(0, self.clustimage.results['feat'].shape[0]), disable=disable_tqdm()):
            idx = np.where(self.clustimage.results['feat'][i,:]<=score)[0]
            if len(idx)>1:
                if len(out)==0:
                    out.append(idx)
                    pathnames.append(self.clustimage.results['pathnames'][idx])
                    scores.append(self.clustimage.results['feat'][i,idx])
                elif ~np.any(list(map(lambda x: np.all(np.isin(x, idx)), out))):
                    out.append(idx)
                    pathnames.append(self.clustimage.results['pathnames'][idx])
                    scores.append(self.clustimage.results['feat'][i,idx])

        self.results = {'pathnames':pathnames, 'scores':scores}
        logger.info('Number of groups with similar images detected: %d' %(len(self.results['pathnames'])))

    def clean(self):
        """Clean or removing previous results and models to ensure correct working."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if hasattr(self, 'results'): del self.results
            if hasattr(self, 'params'): del self.params
        # Store results
        # self.results = {'img':None, 'feat':None, 'xycoord':None, 'pathnames':None, 'labels': None}

    def import_example(self, data='titanic', url=None, sep=','):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
        url : str
            url link to to dataset.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep)


    def plot(self, ncols=None, cmap=None, figsize=(15,10)):
        """Plot the results.

        Parameters
        ----------
        labels : list, (default: None)
            Cluster label to plot. In case of None, all cluster labels are plotted.
        ncols : int, (default: None)
            Number of columns to use in the subplot. The number of rows are estimated based on the columns.
        Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        show_hog : bool, (default: False)
            Plot the hog features next to the input image.
        min_clust : int, (default: 1)
            Plots are created for clusters with > min_clust samples
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        # Do some checks and set defaults
        self._check_status()
        cmap = cl._set_cmap(cmap, self.params['grayscale'])

        # Plot the clustered images
        if (self.results.get('pathnames', None) is not None):
            # Set logger to error only
            # verbose = logger.getEffectiveLevel()
            # set_logger(verbose=50)

            # Run over all labels.
            for i, pathnames in tqdm(enumerate(self.results['pathnames']), disable=disable_tqdm()):
                # Get the images that cluster together
                imgs = list(map(lambda x: self.clustimage.imread(x, colorscale=1, dim=self.params['dim'], flatten=False), pathnames))
                # Make subplots
                # Setup rows and columns
                _, ncol = self.clustimage._get_rows_cols(len(imgs), ncols=ncols)
                labels=list(map(lambda x: 'score: ' + x, list(self.results['scores'][i].astype(str))))
                self.clustimage._make_subplots(imgs, ncol, cmap, figsize, title=("Number of similar images %s" %(len(pathnames))), labels=labels)

                # Restore verbose status
                # set_logger(verbose=verbose)

    def _check_status(self):
        if not hasattr(self, 'results'):
            raise Exception(logger.error('Results in missing! Hint: try to first fit_transform() your data!'))


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=','):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
	verbose : int, (default: 20)
		Print progress to screen. The default is 3.
		60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        logger.info('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        logger.info('Downloading [%s] dataset from github source..' %(data))
        wget(url, PATH_TO_DATA)

    # Import local dataset
    logger.info('Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Download files from github source
def wget(url, writepath):
    """ Retrieve file from url.

    Parameters
    ----------
    url : str.
        Internet source.
    writepath : str.
        Directory to write the file.

    Returns
    -------
    None.

    Example
    -------
    >>> import clustimage as cl
    >>> images = cl.wget('https://erdogant.github.io/datasets/flower_images.zip', 'c://temp//flower_images.zip')

    """
    r = requests.get(url, stream=True)
    with open(writepath, "wb") as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)


# %% Import example dataset from github.
def load_example(data='breast'):
    """Import example dataset from sklearn.

    Parameters
    ----------
    'breast' : str, two-class
    'titanic': str, two-class
    'iris' : str, multi-class

    Returns
    -------
    tuple containing dataset and response variable (X,y).

    """

    try:
        from sklearn import datasets
    except:
        print('This requires: <pip install sklearn>')
        return None, None

    if data=='iris':
        X, y = datasets.load_iris(return_X_y=True)
    elif data=='breast':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    elif data=='titanic':
        X, y = datasets.fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    return X, y

# %% unzip
def unzip(path_to_zip):
    """Unzip files.

    Parameters
    ----------
    path_to_zip : str
        Path of the zip file.

    Returns
    -------
    getpath : str
        Path containing the unzipped files.

    Example
    -------
    >>> import clustimage as cl
    >>> dirpath = cl.unzip('c://temp//flower_images.zip')

    """
    getpath = None
    if path_to_zip[-4:]=='.zip':
        if not os.path.isdir(path_to_zip):
            logger.info('Extracting files..')
            pathname, _ = os.path.split(path_to_zip)
            # Unzip
            zip_ref = zipfile.ZipFile(path_to_zip, 'r')
            zip_ref.extractall(pathname)
            zip_ref.close()
            getpath = path_to_zip.replace('.zip', '')
            if not os.path.isdir(getpath):
                logger.error('Extraction failed.')
                getpath = None
    else:
        logger.warning('Input is not a zip file: [%s]', path_to_zip)
    # Return
    return getpath



# %%
def set_logger(verbose=20):
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


# %% Main
# if __name__ == "__main__":
#     import undouble as undouble
#     df = undouble.import_example()
#     out = undouble.fit(df)
#     fig,ax = undouble.plot(out)
