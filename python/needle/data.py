import gzip
from re import S
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    images, labels = None, None

    with gzip.open(image_filesname) as f:
        pixels = np.frombuffer(f.read(), 'B', offset=16)
        images = pixels.reshape(-1, 784).astype('float32') / 255
    with gzip.open(label_filename) as f:
        labels = np.frombuffer(f.read(), 'B', offset=8)
    
    return images, labels   




class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return img[:,::-1,:]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)

        height, width, _ = img.shape
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        img = img[
            shift_x + self.padding:shift_x + self.padding + height,
            shift_y + self.padding:shift_y + self.padding + width,
        ]
        return img


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ordering = None
        self.current_batch = 0
        # self._init_ordering()

    def _init_ordering(self):
        self.current_batch = 0
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        self.ordering = np.array_split(idxs, 
                                        range(self.batch_size, len(self.dataset), self.batch_size))

    def __iter__(self):
        self._init_ordering()
        return self

    def __next__(self):
        if self.current_batch >= len(self.ordering):
            raise StopIteration
        batch = [self.dataset[i] for i in self.ordering[self.current_batch]]
        self.current_batch += 1
        return tuple(Tensor.make_const(np.stack(b)) for b in zip(*batch))


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.x, self.y = parse_mnist(image_filename, label_filename)
        self.height = 28
        self.width = 28
        self.num_channels = 1

    def __getitem__(self, index) -> object:
        imgs, labels = self.x[index], self.y[index]
        is_single_item = imgs.ndim < 2

        if is_single_item:
            imgs = imgs[None, :]
            labels = np.array([labels])
        
        imgs = imgs.reshape(imgs.shape[0], self.height, self.width, self.num_channels)
        imgs = np.array([self.apply_transforms(img) for img in imgs])

        if is_single_item:
            imgs = imgs[0]
            labels = labels[0]

        return imgs, labels

    def __len__(self) -> int:
        return len(self.y)

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
