
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import urllib
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import torchvision
from torchvision import transforms

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a"}

# ===========================================================
# simple baseline examples
def load_libsvm(name, datadir='datasets/'):
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(datadir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    X = csr_matrix(X).todense()
    labels = np.unique(y)
    y[y==labels[0]] = 0
    y[y==labels[1]] = 1
    return X, y

# ===========================================================
# matrix matrix_factorization
def generate_synthetic_mfac(xdim=6, ydim=10, nsamples=1000, A_condition_number=1e-10):
	"""
    Generate a synthetic matrix factorization dataset as suggested by Ben Recht.
		See: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
	"""
	Atrue = np.linspace(1, A_condition_number, ydim
					   ).reshape(-1, 1) * np.random.rand(ydim, xdim)
	# the inputs
	X = np.random.randn(xdim, nsamples)
	# the y's to fit
	Ytrue = Atrue.dot(X)
	data = (X.T, Ytrue.T)
	return data

# ===========================================================
# load cifar 100
def load_mnist(datadir):
    dataset = torchvision.datasets.MNIST(datadir, train=None,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ]))
    X, y = dataset.data.numpy(), dataset.targets.numpy()
    return X, y

# ===========================================================
# load cifar 10
def load_cifar10(datadir):
    transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    dataset = torchvision.datasets.CIFAR10(
        root=datadir,
        train=None,
        download=True,
        transform=transform_function)
    X, y = np.array(dataset.data), np.array(dataset.targets)
    return X, y

# ===========================================================
# matrix matrix_factorization
def load_cifar100(datadir):
    transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=None,
            download=True,
            transform=transform_function)
    X, y = np.array(dataset.data), np.array(dataset.targets)
    return X, y
