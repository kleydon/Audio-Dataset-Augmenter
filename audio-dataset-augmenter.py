# audio-dataset-augmenter.py

# Script for augmenting small audio datasets for machine learning,
# by introducing (reproducibly) variable:
# * noise
# * reverb
# * gain
# * clipping, 
# * temporal offset
# * background audio
# * ...


# Foreground/middleground/background audio "layer" composition:
#
# fg layer:  ++++++++.......|.......++++++++
# mg layer:            .........|........
# bg layer:  .................|.............
#
# * offset: distance between layer's center |, and
#           center | of the bg layer .

# * padding: space around fg layer sound +++++,
#            accomodated by other layers.

# * space: defined by the size of the bg layer


# *** WORK IN PROGRESS ***
# TO DO:
# * Not handling padding yet
# * Crop layer arrays down to size, based on how they align wrt each other
# * Instead of generating the augmented dataset directly, generate a list of "recipes", then generate
#   the dataset from the list of recipes - so that its possible, after the fact, to determine what 
#   operations produced a given augmented sample...
# * Why is reverb of 0 still creating audible reverb? Should I use other/additional params?
# * What other pysndfx/sox audio effects might it make sense to include?
# * Command line args and json i/o


import numpy as np  # Array stuff, noise
import librosa  # Importing, saving, resampling...
from pysndfx import AudioEffectsChain  # Reverb, SOX-functionality
import glob
import os


SAMPLE_RATE = 44100
MAX_SPACE = 30.0 * SAMPLE_RATE


# File I/O

def dirContents(dirPath="./", fileType=".wav"):
    # https://stackoverflow.com/a/168424
    # https://stackoverflow.com/a/15010678
    dirPath = os.path.join(dirPath, '')  # Ensure trailing slash
    fileNames = filter(os.path.isfile, glob.glob(dirPath + "*" + fileType))
    fileNames.sort(key=lambda x: os.path.getmtime(x))
    return fileNames


# Sample rate conversion functions

def secondsToSamples(sec):
    return int(sec * SAMPLE_RATE)


# Audio Layers

class _Layer:  # Private base class
    def __init__(self,
                 amplitude=1.0,
                 samples=None,
                 offset=0,
                 padding=0,
                 space=MAX_SPACE):
        self.amplitude = amplitude
        self.samples = samples
        self.offset = offset
        self.padding = padding
        self.space = space

    def data(self):
        amped = self.amplitude * self.samples
        diff_len = self.space - amped.shape[0]
        padded = np.pad(amped,
                        (diff_len, 0),
                        'constant',
                        constant_values=1.0)
        centered = np.roll(padded, -int(diff_len/2))
        offsetted = np.roll(centered, self.offset)
        return offsetted


class BGLayer(_Layer):
    # Base, environmental sound
    def __init__(self,
                 amplitude=1.0,
                 samples=None):
        _Layer.__init__(self,
                       amplitude=amplitude,
                       samples=samples,
                       offset=0,
                       padding=0,
                       space=samples.shape[0])


class MGLayer(_Layer):
    # Distinct sounds, happening AROUND a whistle or whistle-like sound
    def __init__(self,
                 amplitude=1.0,
                 samples=None,
                 offset=0,
                 space=MAX_SPACE):
        _Layer.__init__(self,
                       amplitude=amplitude,
                       samples=samples,
                       offset=offset,
                       padding=0,
                       space=space)


class FGLayer(_Layer):
    # Holds a (valid or invalid) whistle sound, or whistle-like sound
    def __init__(self,
                 amplitude=1.0,
                 samples=None,
                 offset=0,
                 padding=2,
                 space=MAX_SPACE):
        _Layer.__init__(self,
                       amplitude=amplitude,
                       samples=samples,
                       offset=offset,
                       padding=padding,
                       space=space)


# Audio Layer Compositor; arranges and combines layers

class LayersCompositor:

    def __init__(self, bg, mgs, fg):
        self.bg = bg  # bg layer
        self.mgs = mgs  # mg layers
        self.fg = fg  # fg layers

    def apply(self):

        # Add in bg, mgs and fg
        res = self.bg.data()
        for mg in self.mgs:
            res += mg.data()
        res += self.fg.data()

        return res


# Recording Device Transforms

class Gain:
    def __init__(self, level=1.0):
        self.level = level

    def apply(self, src):
        tmp = src
        return level * tmp


class Noise:
    def __init__(self, sigma=0.0, type='white'):
        self.type = type
        self.sigma = sigma
        np.random.seed(1)  # For repeatability
        self.rng = np.random

    def apply(self, src):
        tmp = src
        noise = self.rng.normal(0, self.sigma, src.shape[0])
        return tmp + noise


class Reverb:
    def __init__(self, level=0.0):
        self.level = level
        self.fx = AudioEffectsChain().reverb(reverberance=100*level)

    def apply(self, src):
        tmp = src
        if self.level > 0.0:
            tmp = self.fx(tmp)
        return tmp


class Clipping:
    def __init__(self, level=0.0):
        self.type = type
        self.level = level

    def apply(self, src):
        tmp = src
        return 1.0 * tmp # Stub - need to implement


class Filter:
    def __init__(self, type='LP'):
        self.type = type

    def apply(self, src):
        tmp = src
        return 1.0 * tmp # Stub - need to implement


# Recording Device

class Recorder:

    # Model of recording device's effect on audio
    def __init__(self,
                 gain=1.0,
                 noise=None,
                 reverb=None,
                 clip=None,
                 filt=None):
        self.gain = gain
        self.noise = noise
        self.reverb = reverb
        self.clipping = clip
        self.filter = filt

    def record(self, src):

        tmp = self.gain * src

        if self.filter:
            tmp = self.filter.apply(tmp)

        if self.noise:
            tmp = self.noise.apply(tmp)

        if self.reverb:
            tmp = self.reverb.apply(tmp)

        if self.clipping:
            tmp = self.clipping.apply(tmp)

        return tmp


# Main

bg_dir = "./1-SourceSounds/1-BG/"
mg_dir = "./1-SourceSounds/2-MG/"
fg_dir = "./1-SourceSounds/2-FG/"


'''
# Quick, hacky test of reverb library
y, sr = librosa.load("./1-SourceSounds/3-FG/Whistle/whistle-v.wav")
print sr

r = Reverb(level=0.5)
y2 = r.apply(y)
librosa.output.write_wav('XXX.wav', y2, sr)
'''


# Sound layer arrays; start with a simple test case...
fg_sound = np.array([30, 30, 30, 30, 30, 30, 30])
mg2_sound = np.array([20, 20, 20, 20, 20])
mg1_sound = np.array([10, 10, 10])
bg_sound = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
space = bg_sound.shape[0]

fg = FGLayer(amplitude=1.0,
             samples=fg_sound,
             offset=2,
             padding=4,
             space=space)
print("fg: ", fg.data())

mg2 = MGLayer(amplitude=1.0,
              samples=mg2_sound,
              offset=0,
              space=space)
print("mg2: ", mg2.data())

mg1 = MGLayer(amplitude=1.0,
              samples=mg1_sound,
              offset=-2,
              space=space)
mgs = [mg1, mg2]
print("mg1: ", mg1.data())

bg = BGLayer(amplitude=1.0,
             samples=bg_sound)
print("bg: ", bg.data())

lc = LayersCompositor(bg=bg,
                      mgs=mgs,
                      fg=fg)

c = lc.apply()

rd = Recorder(gain=1.0,
              noise=Noise(sigma=0.1),
              reverb=Reverb(reverberance=1.0),
              clip=None,
              filt=None)

res = rd.record(c)

#n = Noise(level=0.5)
#print "Noise:", n.apply(np.ones(30))

print res
