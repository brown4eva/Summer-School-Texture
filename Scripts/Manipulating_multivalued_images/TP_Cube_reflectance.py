### IMPORTS ### ==================================================================================

from hsi_loader import *
from hsi_manipulation import *
from hsi_colorspace import *

import matplotlib.pyplot as plt
from tkinter import filedialog as fd

CIE_1931_path = fd.askopenfilename(title = "CIE 1931 Standard Observer")

# 1 . Load spectral cube
cube, cube_wl = load_envi_image(fd.askopenfilename(title="BIN or DAT, raw data"), fd.askopenfilename(title="Header .hdr"))

# 2 . Load scene illuminant
illuminant, illum_wl = load_spectrum_file(fd.askopenfilename(title="Illumination spectrum (csv or spd)"))

# 3 . Compute reflectance cube

# 4 . Display diffrence maps

# 4-1 . Mono band difference

# 4-2 . Average difference

# 5 . display average spectrum over regions of interests