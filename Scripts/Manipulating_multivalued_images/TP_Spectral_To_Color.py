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

# 3 . knowing scene illuminant we can build our XYZ to RGB transfomation matrix
_, _, Xw, Yw, Zw = compute_chromaticity_coordinates(spectrum= illuminant, spectrum_wavelengths= illum_wl, cmf_path= CIE_1931_path)

# 4 . Compute the chromaticity coordinate of the cameras system based on previously aquired sensitivity spectras of the camera channels
red_ssf, red_wl         = load_spectrum_file(fd.askopenfilename(title="Red sensitivity spectrum (csv or spd)"))
green_ssf, green_wl     = load_spectrum_file(fd.askopenfilename(title="Green sensitivity spectrum (csv or spd)"))
blue_ssf, blue_wl       = load_spectrum_file(fd.askopenfilename(title="Blue sensitivity spectrum (csv or spd)"))

xr, yr, _, _, _ = compute_chromaticity_coordinates(spectrum= red_ssf, spectrum_wavelengths= red_wl, cmf_path= CIE_1931_path)
xg, yg, _, _, _ = compute_chromaticity_coordinates(spectrum= green_ssf, spectrum_wavelengths= green_wl, cmf_path= CIE_1931_path)
xb, yb, _, _, _ = compute_chromaticity_coordinates(spectrum= blue_ssf, spectrum_wavelengths= blue_wl, cmf_path= CIE_1931_path)

# les coordonées d'un système RGB (# sRGB in this case)
"""
xr, yr = 0.6400, 0.3300
xg, yg = 0.3000, 0.6000
xb, yb = 0.1500, 0.0600
"""

# Compute the transformation Matrix, XYZ to your RGB space (Camera RGB space)

# 4 . Use said matrix to tranform the spectral image to XYZ then to RGB

# 4-A . Spectral to XYZ

# 4-B . XYZ to RGB

# 5 . Display