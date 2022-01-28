"""Python package undouble is to detect (near-)identical images."""
# --------------------------------------------------
# Name        : undouble.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/undouble
# Licence     : See licences
# --------------------------------------------------

import os
import requests
import logging
import numpy as np
from tqdm import tqdm
import zipfile
from clustimage import Clustimage
import clustimage.clustimage as cl
import shutil
import cv2
import copy
from ismember import ismember

logger = logging.getLogger('')
for handler in logger.handlers[:]:  # get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
# formatter = logging.Formatter('[%(asctime)s] [undouble]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('[undouble] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Undouble():
    """Detect duplicate images.

    Description
    -----------
    Python package undouble is to detect (near-)identical images.

    The following steps are taken:
        1. Read recursively all images from directory with the specified extensions.
        2. Compute image hash.
        3. Group similar images.
        4. Move if desired.

    Parameters
    ----------
    method : str, (default: 'phash')
        Image hash method.
        * 'ahash': Average hash
        * 'phash': Perceptual hash
        * 'dhash': Difference hash
        * 'whash-haar': Haar wavelet hash
    targetdir : str, (default: None)
        Directory to read the images.
    hash_size : integer (default: 8)
        The hash_size will be used to scale down the image and create a hash-image of length: hash_size*hash_size.
    ext : list, (default: ['png','tiff','jpg'])
        Images with the file extentions are used.
    grayscale : Bool, (default: True)
        Colorscaling the image to gray.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    verbose : int, (default: 20)
        Print progress to screen. The default is 20.
        10:Debug, 20:Info, 30:Warn 40:Error, 60:None

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
    >>> # Import library
    >>> from undouble import Undouble
    >>>
    >>> # Init with default settings
    >>> model = Undouble(method='phash', hash_size=8)
    >>>
    >>> # Import example data
    >>> targetdir = model.import_example(data='flowers')
    >>>
    >>> # Importing the files files from disk, cleaning and pre-processing
    >>> model.import_data(targetdir)
    >>>
    >>> # Compute image-hash
    >>> model.compute_hash()
    >>>
    >>> # Find images with image-hash <= threshold
    >>> model.group(threshold=0)
    >>>
    >>> # Plot the images
    >>> model.plot()
    >>>
    >>> # Move the images
    >>> model.move()

    References
    ----------
    * https://content-blockchain.org/research/testing-different-image-hash-functions/

    """

    def __init__(self, method='phash', targetdir='', grayscale=False, dim=(128, 128), hash_size=8, ext=['png', 'tiff', 'jpg', 'jfif'], verbose=20):
        """Initialize undouble with user-defined parameters."""
        if isinstance(ext, str): ext = [ext]
        # Clean readily fitted models to ensure correct results
        self.clean()
        if verbose<=0: verbose=60
        # Store user setting in params
        self.params = {'method': method, 'grayscale': grayscale, 'dim': dim, 'ext': ext, 'hash_size': hash_size, 'verbose': verbose}
        # Initialize the clustimage library
        self.clustimage = Clustimage(method=self.params['method'], grayscale=self.params['grayscale'], ext=self.params['ext'], dim=self.params['dim'], params_hash={'hash_size': hash_size}, verbose=self.params['verbose'])
        # Set the logger
        set_logger(verbose=verbose)

    def import_data(self, targetdir, black_list=['undouble'], return_results=False):
        """Preprocessing.

        Parameters
        ----------
        black_list : list, (default: ['undouble'])
            Exclude directory with all subdirectories from processing.

        Returns
        -------
        None.

        """
        if isinstance(black_list, str): black_list = [black_list]
        # Set targetdir
        self.params['targetdir'] = targetdir
        # logger.info("Retrieving files from: [%s]" %(self.params['targetdir']))
        # Preprocessing the images the get them in the right scale and size etc
        self.results = self.clustimage.import_data(self.params['targetdir'], black_list=black_list)
        # Remove keys that are not used.
        if 'labels' in self.results: self.results.pop('labels')
        if 'xycoord' in self.results: self.results.pop('xycoord')
        # Return
        if return_results:
            return self.results

    def compute_hash(self, method=None, hash_size=None, return_dict=False):
        """Compute the hash for each image.

        Parameters
        ----------
        method : str, (default: 'phash')
            Image hash method.
            * 'ahash': Average hash
            * 'phash': Perceptual hash
            * 'dhash': Difference hash
            * 'whash-haar': Haar wavelet hash
        hash_size : integer (default: 8)
            The hash_size will be used to scale down the image and create a hash-image of length: hash_size*hash_size.

        Returns
        -------
        None.

        """
        # Set parameters
        if method is not None:
            self.params['method'] = method
            self.clustimage.params['method'] = method
        if hash_size is not None:
            self.params['hash_size'] = hash_size
            self.clustimage.params_hash['hash_size'] = hash_size
        # Set hash parameters
        self.clustimage.params_hash = cl.get_params_hash(self.params['method'], self.clustimage.params_hash)

        # Compute image-hash features
        self.results['img_hash_bin'] = np.array(list(map(self.clustimage.compute_hash, tqdm(self.results['img'], disable=disable_tqdm()))))
        self.results['img_hash_hex'] = self.bin2hex()

        # Build adjacency matrix for the image-hash based on nr. of differences
        logger.info('Compute adjacency matrix [%gx%g] with absolute differences based on the image-hash of [%s].' %(self.results['img_hash_bin'].shape[0], self.results['img_hash_bin'].shape[0], self.params['method']))
        self.results['adjmat'] = (self.results['img_hash_bin'][:, None, :] != self.results['img_hash_bin']).sum(2)

        # Remove keys that are not used.
        if 'labels' in self.results: self.results.pop('labels')
        if 'xycoord' in self.results: self.results.pop('xycoord')
        if 'feat' in self.results: self.results.pop('feat')
        if return_dict:
            return self.results

    def group(self, threshold=0, return_dict=False):
        """Find similar images using the hash signatures.

        Parameters
        ----------
        threshold : float, (default: 0)
            Threshold on the hash value to determine similarity.

        Returns
        -------
        None.

        """
        if self.results['img_hash_bin'] is None:
            logger.warning('Can not group similar images because no features are present. Tip: Use the compute_hash() function first.')
            return None

        # Make sets of images that are similar based on the minimum defined threshold.
        pathnames, indexes, thresholds = [], [], []

        # Only the files that exists (or are not moved in an previous action)
        _, idx = get_existing_pathnames(self.results['pathnames'])
        if np.sum(idx==False)>0:
            logger.warning('Files that are moved in a previous action are kept untouched.')
        paths = self.results['pathnames'][idx]
        feat = copy.deepcopy(self.results['adjmat'])
        feat = feat[idx, :]
        feat = feat[:, idx]

        # Extract similar images with minimum threshold
        for i in tqdm(np.arange(0, feat.shape[0]), disable=disable_tqdm()):
            idx = np.where(feat[i, :]<=threshold)[0]
            if len(idx)>1:
                # if ~np.any(list(map(lambda x: np.all(np.isin(x, idx)), indexes))):
                if ~np.any(list(map(lambda x: np.any(np.isin(x, idx)), indexes))):
                    indexes.append(idx)
                    pathnames.append(paths[idx])
                    thresholds.append(feat[i, idx])

        # Sort on threshold
        for i, _ in enumerate(thresholds):
            # indexes[i] = indexes[i][isort]
            isort=np.argsort(thresholds[i])
            pathnames[i] = pathnames[i][isort]
            thresholds[i] = thresholds[i][isort]

        # Sort on directory
        idx = np.argsort(list(map(lambda x: os.path.split(x[0])[0], pathnames)))
        self.results['select_pathnames'] = np.array(pathnames)[idx].tolist()
        self.results['select_scores'] = np.array(thresholds)[idx].tolist()
        logger.info('[%d] groups with similar image-hash.' %(len(self.results['select_pathnames'])))

        totfiles = np.sum(list(map(len, self.results['select_pathnames'])))
        totgroup = len(self.results['select_pathnames'])
        logger.info('[%d] groups are detected for [%d] images.' %(totgroup, totfiles))
        self.results['stats'] = {'groups': totgroup, 'files': totfiles}

        if return_dict:
            return self.results

    def move(self, filters=None, targetdir=None):
        """Move images.

        Description
        -----------
        Files are moved that are listed by the group() functionality.

        Parameters
        ----------
        filters : list, (Default: ['location'])
            'location' : Only move images that are seen in the same directory.
        targetdir : str (default: None)
            Moving similar files to this directory.
            None: A subdir, named "undouble" is created within each directory.

        Returns
        -------
        None.

        """
        # Do some checks and set defaults
        self._check_status()
        if targetdir is not None:
            if not os.path.isdir(targetdir): raise Exception(logger.error(''))
        # logger.info('Detected images: [%d] across of [%d] groups.' %(totfiles, totgroup))

        totfiles = np.sum(list(map(len, self.results['select_pathnames'])))
        totgroup = len(self.results['select_pathnames'])
        tdir = 'undouble' if targetdir is None else targetdir
        answer = input('\n-------------------------------------------------\n>You are at the point of physically moving files.\n-------------------------------------------------\n>[%d] similar images are detected over [%d] groups.\n>[%d] images will be moved to the [%s] subdirectory.\n>[%d] images will be copied to the [%s] subdirectory.\n\n>[C]ontinue moving all files.\n>[W]ait in each directory.\n>[Q]uit\n>Answer: ' %(totfiles, totgroup, totfiles - totgroup, tdir, totgroup, tdir))
        answer = str.lower(answer)
        if answer == 'q':
            return

        # For each group, check the resolution and location.
        pathmem=''
        for pathnames in self.results['select_pathnames']:
            curdir=os.path.split(pathnames[0])[0]
            if pathmem!=curdir:
                pathmem=curdir
                logger.info('Working in dir: [%s]' %(curdir))
                if answer!='c':
                    answer = input('><enter> to proceed to the next directory.\n>[C]ontinue to move all files.\n>[Q]uit\nAnswer: ')
                    answer = str.lower(answer)
                    if answer == 'q': return
            # Check file exists
            pathnames = np.array(pathnames)
            pathnames = pathnames[list(map(os.path.isfile, pathnames))]
            # Check whether move is allowed
            filterOK = filter_checks(pathnames, filters)
            # Move to targetdir
            if filterOK:
                # Sort images on resolution and least amount of blur (best first)
                pathnames = sort_images(pathnames)['pathnames']
                # Move to dir
                self._move_to_dir(pathnames, targetdir, make_moved_filename_consistent=True)

    def _move_to_dir(self, pathnames, targetdir, make_moved_filename_consistent=True):
        """Move to target directory.

        Description
        -----------
        The first pathname is copied, the other are moved.

        Parameters
        ----------
        pathnames : list of str
        targetdir : target directory to copy and move the files

        """
        # Create targetdir
        movedir, dirname, filename, ext = create_targetdir(pathnames[0], targetdir)
        # 1. Copy first file to targetdir and add "_COPY"
        shutil.copy(pathnames[0], os.path.join(movedir, filename + '_COPY' + ext))
        # 2. Move all others
        for i, file in enumerate(pathnames[1:]):
            logger.debug(file)
            if make_moved_filename_consistent:
                ext = os.path.split(file)[1][-4:].lower()
                shutil.move(file, os.path.join(movedir, filename + '_' + str(i) + ext))
            else:
                shutil.move(file, os.path.join(movedir, os.path.split(file)[1]))

    def clean(self, params=True, results=True):
        """Clean or removing previous results and models to ensure correct working."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if results and hasattr(self, 'results'): del self.results
            if params and hasattr(self, 'params'): del self.params
            if params and hasattr(self, 'clustimage'): del self.clustimage
        # Store results
        # self.results = {'img':None, 'feat':None, 'xycoord':None, 'pathnames':None, 'labels': None}

    def import_example(self, data='flowers', url=None):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'flowers', 'mnist', 'cat_and_dog'
        url : str
            url link to to dataset.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url)

    def plot(self, cmap=None, figsize=(15, 10)):
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
        ncols=None
        cmap = cl._set_cmap(cmap, self.params['grayscale'])
        colorscale = 0 if self.params['grayscale'] else 1

        # Plot the clustered images
        if (self.results.get('select_pathnames', None) is not None):
            # Set logger to error only
            # verbose = logger.getEffectiveLevel()
            # set_logger(verbose=50)

            # Run over all labels.
            for i, pathnames in tqdm(enumerate(self.results['select_pathnames']), disable=disable_tqdm()):
                # pathnames = self.results['pathnames'][idx]
                # Check whether file exists.
                pathnames = np.array(pathnames)[list(map(os.path.isfile, pathnames))]
                # Only groups with > 1 images needs to be moved.
                if len(pathnames)>1:
                    # Sort images
                    imgscores = sort_images(pathnames, hash_scores=self.results['select_scores'][i])
                    # Get the images that cluster together
                    imgs = list(map(lambda x: self.clustimage.imread(x, colorscale=colorscale, dim=self.params['dim'], flatten=False), imgscores['pathnames']))
                    # Setup rows and columns
                    _, ncol = self.clustimage._get_rows_cols(len(imgs), ncols=ncols)
                    labels = list(map(lambda x, y, z: 'score: ' + str(int(x)) + ' and blur: ' + str(int(y)) + '\nresolution: ' + str(int(z)), imgscores['hash_scores'], imgscores['blur'], imgscores['resolution']))
                    # Make subplots
                    self.clustimage._make_subplots(imgs, ncol, cmap, figsize, title=("[%s] groups with similar image-hash" %(len(imgscores['pathnames']))), labels=labels)

                # Restore verbose status
                # set_logger(verbose=verbose)

    def _check_status(self):
        if not hasattr(self, 'results'):
            raise Exception(logger.error('Results missing! Hint: try to first use the model.group() functionality'))

    def compute_imghash(self, img, hash_size=None, to_array=False):
        """Compute hash.

        Parameters
        ----------
        img : Object or RGB-image.
            Image.
        hash_size : integer (default: None)
            The hash_size will be used to scale down the image and create a hash-image of length: hash_size*hash_size.
        to_array : Bool (default: False)
            True: Return the hash-array in the same size as the scaled image.
            False: Return the hash-image vector.

        Returns
        -------
        imghash : numpy-array
            Hash.

        """
        if hash_size is None: hash_size=self.params['hash_size']
        if hasattr(img, 'results'):
            img = self.results['img']
        elif len(img.shape)<=3:
            img = [img]

        # Compute hash
        hashes = list(map(lambda x: self.clustimage.compute_hash(x, hash_size=hash_size), img))

        # Convert to image-hash
        if to_array:
            hashes = np.array(list(map(lambda x: x.ravel().astype(int), hashes)))
            hashes = np.c_[hashes]
        else:
            hashes = list(map(lambda x: x.reshape(hash_size,hash_size), hashes))

        return hashes

    def bin2hex(self):
        """Binary to hex.

        Returns
        -------
        str
            Hex of image hash.

        """
        if hasattr(self, 'results'):
            return np.array(list(map(lambda x: hex(int(''.join(x.astype(int).astype(str)), 2)), self.results['img_hash_bin'])))
        else:
            logger.warning('Results missing! Hint: try to first use compute_hash()')


# %% Import example dataset from github.
def import_example(data='flowers', url=None):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'flowers', 'faces', 'mnist', 'cat_and_dog'
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
    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if data=='cat_and_dog':
        df = cl.import_example(data=None, url='https://erdogant.github.io/datasets/cat_and_dog.zip', curpath=curpath)
    else:
        df = cl.import_example(data=data, curpath=curpath)
    # Return
    return df

    # %% Set target directory


def create_targetdir(pathname, targetdir):
    """Create directory.

    Parameters
    ----------
    pathname : str
        Absolute path location of the image of interest.
    targetdir : str
        Target directory.

    Returns
    -------
    movedir : str
        Absolute path to directory.
    dirname : str
        Absolute path to directory.
    filename : str
        Name of the file.
    ext : str
        Extension.

    """
    dirname, filename, ext = seperate_path(pathname)
    # Set the targetdir
    if targetdir is None:
        movedir = os.path.join(dirname, 'undouble')
    else:
        movedir = targetdir

    if not os.path.isdir(movedir):
        logger.debug('Create dir: <%s>' %(movedir))
        os.makedirs(movedir, exist_ok=True)
    # Return
    return movedir, dirname, filename, ext


# %%
def _compute_rank(scores, higher_is_better=True):
    rankscore = np.argsort(scores)[::-1]
    uires = np.unique(scores)
    uirankscore = np.argsort(uires)
    if higher_is_better: uirankscore = uirankscore[::-1]
    rank_res = uirankscore[ismember(scores, uires)[1]]
    return rank_res, rankscore


# %% Sort images.
def sort_images(pathnames, hash_scores=None, sort_first_img=False):
    """Sort images.

    Description
    -----------
    Sort images on the following conditions:
        1. Resolution
        2. Amount of blur

    Parameters
    ----------
    pathnames : list of str.
        Absolute locations to image path.

    Returns
    -------
    list of str
        images sorted on conditions.

    """
    if not sort_first_img:
        # Remove the first image from the list.
        pathname1 = pathnames[0]
        pathnames = pathnames[1:]

    # Compute resolution (higher is better)
    scor_res = np.array(list(map(lambda x: np.prod(cl._imread(x).shape[0:2]), pathnames)))
    ranks_sim, rank_exact = _compute_rank(scor_res, higher_is_better=True)

    # Compute amount of blurr (higher is better)
    scor_blr = np.ceil(np.array(list(map(compute_blur, pathnames))))

    # Within the ordering of the resolution, prefer the least blurry images.
    for r in ranks_sim:
        idx = np.where(r==ranks_sim)[0]
        if len(idx)>1:
            rank_exact[idx] = idx[np.argsort(scor_blr[idx])[::-1]]

    for i in rank_exact:
        logger.debug('%g - %g' %(scor_res[i], scor_blr[i]))

    results = {'pathnames': pathnames[rank_exact], 'resolution': scor_res[rank_exact], 'blur': scor_blr[rank_exact], 'idx': rank_exact}
    # Stack together
    if not sort_first_img:
        scor_res1 = np.prod(cl._imread(pathname1).shape[0:2])
        scor_blr1 = np.ceil(compute_blur(pathname1))
        results['pathnames'] = np.array([pathname1] + list(results['pathnames']))
        results['resolution'] = np.array([scor_res1] + list(results['resolution']))
        results['blur'] = np.array([scor_blr1] + list(results['blur']))
        results['idx'] = [0] + list(results['idx'] + 1)

    hash_scores = np.array(hash_scores)[results['idx']] if hash_scores is not None else np.array([0] *len(results['idx']))
    results['hash_scores'] = hash_scores
    # Return
    return results


# %%
def filter_checks(pathnames, filters):
    """Filter checks.

    Parameters
    ----------
    pathnames : list of str
        pathnames to the images.
    filters : list, (Default: ['location'])
        'location' : Only move images that are seen in the same directory.

    Returns
    -------
    bool
        When all filters are true.

    """
    resOK, locOK = True, True
    # Check nr. of files
    fileOK = True if len(pathnames)>1 else False
    # Get resolution
    if (filters is not None) and np.isin('resolution', filters):
        res = np.array(list(map(lambda x: cl._imread(x).shape[0:2], pathnames)))
        resOK = np.all(np.isin(res[0], res))
    # Get location
    if (filters is not None) and np.isin('location', filters):
        loc = list(map(lambda x: os.path.split(x)[0], pathnames))
        locOK = np.all(np.isin(loc[0], loc))

    return np.all([fileOK, resOK, locOK])


# %% Download files from github source
def wget(url, writepath):
    """Retrieve file from url.

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
def compute_blur(pathname):
    """Compute amount of blur in image.

    Description
    -----------
    load the image, convert it to grayscale, and compute the focus measure of
    the image using the Variance of Laplacian method. The returned scores <100
    are generally more blurry.

    Parameters
    ----------
    pathname : str
        Absolute path location to image.

    Returns
    -------
    fm_score : float
        Score the depicts the amount of blur. Scores <100 are generally more blurry.

    """
    # method
    img = cv2.imread(pathname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the Laplacian of the image and then return the focus measure, which is simply the variance of the Laplacian
    fm_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # if the focus measure is less than the supplied threshold, then the image should be considered "blurry"
    if fm_score < 100:
        logger.debug('Blurry image> %s' %(pathname))
    # show the image
    # cv2.putText(img, "{}: {:.2f}".format('amount of blur:', fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    return fm_score


# %%
def get_existing_pathnames(pathnames):
    """Get existing pathnames.

    Parameters
    ----------
    pathnames : list of str
        pathnames to the images.

    """
    Iloc = np.array(list(map(os.path.isfile, pathnames)))
    return pathnames[Iloc], np.array(Iloc)


# %%
def set_logger(verbose=20):
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


# %%
def seperate_path(pathname):
    """Seperate path.

    Parameters
    ----------
    pathnames : list of str
        pathnames to the images.

    Returns
    -------
    dirname : str
        directory path.
    filename : str
        filename.
    ext
        Extension.

    """
    dirname, filename = os.path.split(pathname)
    filename, ext = os.path.splitext(filename)
    return dirname, filename, ext.lower()

# %%

# %% Main
# if __name__ == "__main__":
#     import undouble as undouble
#     df = undouble.import_example()
#     out = undouble.compute_hash(df)
#     fig,ax = undouble.plot(out)
