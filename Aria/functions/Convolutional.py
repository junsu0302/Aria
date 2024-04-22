import numpy as np

from Aria.core.Function import Function
from Aria.core.Utils import as_variable
from Aria.functions.utils.Convolutional import pair, get_conv_outsize, im2col
from Aria.functions.Tensor import linear
from Aria.functions.utils.Convolutional import im2col_array, col2im_array, get_deconv_outsize

class Conv2d(Function):
  def __init__(self, stride=1, pad=0):
    super().__init__()
    self.stride = pair(stride)
    self.pad = pair(pad)

  def forward(self, x, W, b):
    KH, KW = W.shape[2:]
    col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

    y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
    if b is not None:
      y += b
    y = np.rollaxis(y, 3, 1)
    return y

  def backward(self, gy):
    x, W, b = self.inputs
    gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))
    gW = Conv2DGradW(self)(x, gy)
    gb = None
    if b.data is not None:
      gb = gy.sum(axis=(0, 2, 3))
    return gx, gW, gb

def conv2d(x, W, b=None, stride=1, pad=0):
  return Conv2d(stride, pad)(x, W, b)

class Deconv2d(Function):
  def __init__(self, stride=1, pad=0, outsize=None):
    super().__init__()
    self.stride = pair(stride)
    self.pad = pair(pad)
    self.outsize = outsize

  def forward(self, x, W, b):
    Weight = W
    SH, SW = self.stride
    PH, PW = self.pad
    C, OC, KH, KW = Weight.shape
    N, C, H, W = x.shape
    if self.outsize is None:
      out_h = get_deconv_outsize(H, KH, SH, PH)
      out_w = get_deconv_outsize(W, KW, SW, PW)
    else:
      out_h, out_w = pair(self.outsize)
    img_shape = (N, OC, out_h, out_w)

    gcol = np.tensordot(Weight, x, (0, 1))
    gcol = np.rollaxis(gcol, 3)
    y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
    if b is not None:
      self.no_bias = True
      y += b.reshape((1, b.size, 1, 1))
    return y

  def backward(self, gy):
    x, W, b = self.inputs

    gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
    f = Conv2DGradW(self)
    gW = f(gy, x)
    gb = None
    if b.data is not None:
      gb = gy.sum(axis=(0, 2, 3))
    return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
  return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
  def __init__(self, conv2d):
    W = conv2d.inputs[1]
    kh, kw = W.shape[2:]
    self.kernel_size = (kh, kw)
    self.stride = conv2d.stride
    self.pad = conv2d.pad

  def forward(self, x, gy):
    col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
    gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
    return gW

  def backward(self, gys):
    x, gy = self.inputs
    gW, = self.outputs

    xh, xw = x.shape[2:]
    gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
    ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
    return gx, ggy

class Pooling(Function):
  def __init__(self, kernel_size, stride=1, pad=0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad

  def forward(self, x):
    col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

    N, C, KH, KW, OH, OW = col.shape
    col = col.reshape(N, C, KH * KW, OH, OW)
    self.indexes = col.argmax(axis=2)
    return col.max(axis=2)

  def backward(self, gy):
    return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
  def __init__(self, mpool2d):
    self.mpool2d = mpool2d
    self.kernel_size = mpool2d.kernel_size
    self.stride = mpool2d.stride
    self.pad = mpool2d.pad
    self.input_shape = mpool2d.inputs[0].shape
    self.dtype = mpool2d.inputs[0].dtype
    self.indexes = mpool2d.indexes

  def forward(self, gy):
    N, C, OH, OW = gy.shape
    N, C, H, W = self.input_shape
    KH, KW = pair(self.kernel_size)

    gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

    indexes = (self.indexes.ravel() + np.arange(0, self.indexes.size * KH * KW, KH * KW))
        
    gcol[indexes] = gy.ravel()
    gcol = gcol.reshape(N, C, OH, OW, KH, KW)
    gcol = np.swapaxes(gcol, 2, 4)
    gcol = np.swapaxes(gcol, 3, 5)

    return col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False)

  def backward(self, ggx):
    f = Pooling2DWithIndexes(self.mpool2d)
    return f(ggx)


class Pooling2DWithIndexes(Function):
  def __init__(self, mpool2d):
    self.kernel_size = mpool2d.kernel_size
    self.stride = mpool2d.stride
    self.pad = mpool2d.pad
    self.input_shpae = mpool2d.inputs[0].shape
    self.dtype = mpool2d.inputs[0].dtype
    self.indexes = mpool2d.indexes

  def forward(self, x):
    col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
    N, C, KH, KW, OH, OW = col.shape
    col = col.reshape(N, C, KH * KW, OH, OW)
    col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
    indexes = self.indexes.ravel()
    col = col[np.arange(len(indexes)), indexes]
    return col.reshape(N, C, OH, OW)

def pooling(x, kernel_size, stride=1, pad=0):
  return Pooling(kernel_size, stride, pad)(x)