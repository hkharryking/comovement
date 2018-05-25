from torch import nn
import torch
from wavelets_pytorch.transform import WaveletTransformTorch
import numpy as np
from torch.autograd import Variable
from pycwt.helpers import rect, fft, fft_kwargs
from scipy.signal import convolve2d

class Attention_WCC(nn.Module):
    """
    Attention model with sliding window Wavelet coherence
    """

    def get_code(self):
        return 'WCC'

    def __init__(self,window,dt=0.1,dj=0.125,f0=6):
        super(Attention_WCC, self).__init__()
        self.window_size=window*3-2 # practical window size, keep in accordance with the other models
        self.wavelet=WaveletTransformTorch(dt,dj,cuda=True)
        self._set_f0(f0)
        self.dt=dt
        self.dj=dj

    def _set_f0(self, f0):
        # Sets the Morlet wave number, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta},
        # \gamma, \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.f0 = f0  # Wave number
        self.dofmin = 2  # Minimum degrees of freedom
        if self.f0 == 6:
            self.cdelta = 0.776  # Reconstruction factor
            self.gamma = 2.32  # Decorrelation factor for time averaging
            self.deltaj0 = 0.60  # Factor for scale averaging
        else:
            self.cdelta = -1
            self.gamma = -1
            self.deltaj0 = -1

    def smooth(self, W, dt, dj, scales):
        """Smoothing function used in coherence analysis.
        Parameters
        ----------
        W :
        dt :
        dj :
        scales :
        Returns
        -------
        T :
        """
        # The smoothing is performed by using a filter given by the absolute
        # value of the wavelet function at each scale, normalized to have a
        # total weight of unity, according to suggestions by Torrence &
        # Webster (1999) and by Grinsted et al. (2004).
        m, n = W.shape

        # Filter in time.
        k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])['n'])
        k2 = k ** 2
        snorm = scales / dt
        # Smoothing by Gaussian window (absolute value of wavelet function)
        # using the convolution theorem: multiplication by Gaussian curve in
        # Fourier domain for each scale, outer product of scale and frequency
        F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
        smooth = fft.ifft(F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
                          axis=1,  # Along Fourier frequencies
                          **fft_kwargs(W[0, :], overwrite_x=True))
        T = smooth[:, :n]  # Remove possibly padded region due to FFT

        if np.isreal(W).all():
            T = T.real

        # Filter in scale. For the Morlet wavelet it's simply a boxcar with
        # 0.6 width.
        wsize = self.deltaj0 / dj * 2
        win = rect(np.int(np.round(wsize)), normalize=True)
        T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

        return T


    def sliding_window(self,x, step_size=1):
        # unfold dimension to make the sliding window
        return x.unfold(0, self.window_size, step_size)

    def wavelet_coherence_correlation(self,y1,y2):
        # Calculate the continous wavelet tranform
        y1_normal = (y1 - y1.mean()) / y1.std()
        y2_normal = (y2 - y2.mean()) / y2.std()

        [W1, W2] = self.wavelet.cwt(np.array([y1_normal, y2_normal]))

        sj=self.wavelet.scales

        scales1 = np.ones([1, y1.size]) * sj[:, None]
        scales2 = np.ones([1, y2.size]) * sj[:, None]
        S1 = self.smooth(np.abs(W1) ** 2 / scales1, self.dt, self.dj, sj)
        S2 = self.smooth(np.abs(W2) ** 2 / scales2, self.dt, self.dj, sj)
        W12 = W1 * W2.conj()
        scales = np.ones([1, y1.size]) * sj[:, None]
        S12 = self.smooth(W12 / scales, self.dt, self.dj, sj)
        WCT = np.abs(S12) ** 2 / (S1 * S2)
        return torch.from_numpy(WCT.sum(axis=0))

    def attention_generation(self,x):
        result=[]
        for y in x:
            y=y.data.cpu().numpy()
            coherence=self.wavelet_coherence_correlation(y[0],y[1])
            s_out=self.sliding_window(coherence)
            result.append(s_out.sum(dim=1))
        return torch.stack(result, dim=0)

    '''
    returns: attention, sectors score
    '''
    def forward(self, x):
        attentions=self.attention_generation(x)
        return Variable(attentions.float()).cuda(),None

