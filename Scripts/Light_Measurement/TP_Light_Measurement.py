### IMPORTS ### ==================================================================================

from hsi_loader import *
from hsi_manipulation import *
from hsi_colorspace import *

import matplotlib.pyplot as plt
from tkinter import filedialog as fd

# 1 . Load mesured spectras with first illuminant
path_white_spectrum_1        = fd.askopenfilename(title="white reference spectrum under first illuminant (.csv or .spd)")  # White reference acquired with JETI under illuminant 1
path_measured_spectrum_1     = fd.askopenfilename(title="measured spectrum under first illuminant (.csv or .spd)")         # Measured spectra acquired with JETI under illuminant 1

# 1 . Load mesured spectras with first illuminant
path_white_spectrum_2        = fd.askopenfilename(title="white reference spectrum under second illuminant (.csv or .spd)")  # White reference acquired with JETI under illuminant 2
path_measured_spectrum_2     = fd.askopenfilename(title="measured spectrum under second illuminant (.csv or .spd)")         # Measured spectra acquired with JETI under illuminant 2

white_spectrum_1, white_wavelengths_1 = load_spectrum_file(path_white_spectrum_1)
measured_spectrum_1, measured_wavelengths_1 = load_spectrum_file(path_measured_spectrum_1)

white_spectrum_2, white_wavelengths_2 = load_spectrum_file(path_white_spectrum_2)
measured_spectrum_2, measured_wavelengths_2 = load_spectrum_file(path_measured_spectrum_2)

# 1-B . resample
wl = np.arange(start= 380, stop= 781, step= 5)

white_spectrum_1     = resample_spectrum_to_wavelengths(spectrum_values=white_spectrum_1, original_wavelengths=white_wavelengths_1, target_wavelengths=wl, interpolation_method='cubic')
measured_spectrum_1  = resample_spectrum_to_wavelengths(spectrum_values=measured_spectrum_1, original_wavelengths=measured_wavelengths_1, target_wavelengths=wl, interpolation_method='cubic')

white_spectrum_2     = resample_spectrum_to_wavelengths(spectrum_values=white_spectrum_2, original_wavelengths=white_wavelengths_2, target_wavelengths=wl, interpolation_method='cubic')
measured_spectrum_2  = resample_spectrum_to_wavelengths(spectrum_values=measured_spectrum_2, original_wavelengths=measured_wavelengths_2, target_wavelengths=wl, interpolation_method='cubic')

# 2 . Compute reflectance

"""
CODE HERE
"""

# 3 . Display results

