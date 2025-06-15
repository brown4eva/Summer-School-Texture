import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hsi_loader import resample_spectrum_to_wavelengths
from hsi_synthetic import *

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def normalize_by_illuminant_cube(spectral_cube, cube_wavelengths, illuminant_spectrum, 
                                illuminant_wavelengths):
    """
    Normalize a hyperspectral cube by an illuminant spectrum to obtain reflectance
    
    Parameters:
    -----------
    spectral_cube : np.ndarray
        3D hyperspectral cube with shape (height, width, bands)
    cube_wavelengths : np.ndarray
        Wavelengths corresponding to the spectral bands
    illuminant_spectrum : np.ndarray
        Illuminant spectral values
    illuminant_wavelengths : np.ndarray
        Wavelengths corresponding to illuminant spectrum
    
    Returns:
    --------
    reflectance_cube : np.ndarray
        3D reflectance cube with same shape as input cube
    """
    
    # Input validation
    if spectral_cube.shape[2] != len(cube_wavelengths):
        raise ValueError("Number of spectral bands must match wavelength vector length")
    
    print(f"Processing hyperspectral cube of shape: {spectral_cube.shape}")
    print(f"Original wavelength range: {cube_wavelengths[0]:.1f} - {cube_wavelengths[-1]:.1f} nm")
    
    # Resample illuminant spectrum to match cube wavelengths
    resampled_illuminant = resample_spectrum_to_wavelengths(
        illuminant_spectrum, illuminant_wavelengths, cube_wavelengths, 'linear'
    )
    
    # Avoid division by zero - set minimum threshold for illuminant
    min_threshold = 1e-10
    resampled_illuminant = np.maximum(resampled_illuminant, min_threshold)
    
    # Initialize reflectance cube
    reflectance_cube = np.zeros_like(spectral_cube, dtype=np.float32)
    
    # Normalize each pixel spectrum by the illuminant
    # Broadcasting: (H, W, B) / (B,) -> (H, W, B)
    print("Performing illuminant normalization...")
    
    for i in range(spectral_cube.shape[2]):  # For each spectral band
        reflectance_cube[:, :, i] = spectral_cube[:, :, i] / resampled_illuminant[i]
    
    # Clip reflectance values to reasonable range [0, 2] to handle noise
    reflectance_cube = np.clip(reflectance_cube, 0, 2)
    
    print("Illuminant normalization completed")
    print(f"Reflectance range: {np.min(reflectance_cube):.4f} - {np.max(reflectance_cube):.4f}")
    
    return reflectance_cube

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def normalize_by_illuminant_spectrum(measured_spectrum, measured_wavelengths, 
                                   illuminant_spectrum, illuminant_wavelengths):
    """
    Normalize a measured spectrum by an illuminant spectrum to obtain reflectance
    
    Parameters:
    -----------
    measured_spectrum : np.ndarray
        Measured spectral values
    measured_wavelengths : np.ndarray
        Wavelengths corresponding to measured spectrum
    illuminant_spectrum : np.ndarray
        Illuminant spectral values
    illuminant_wavelengths : np.ndarray
        Wavelengths corresponding to illuminant spectrum
    
    Returns:
    --------
    reflectance_spectrum : np.ndarray
        Reflectance spectrum at measured wavelengths
    """
    
    print("Normalizing single spectrum by illuminant...")
    
    # Resample illuminant to match measured spectrum wavelengths
    resampled_illuminant = resample_spectrum_to_wavelengths(
        illuminant_spectrum, illuminant_wavelengths, measured_wavelengths, 'linear'
    )
    
    # Avoid division by zero
    min_threshold = 1e-10
    resampled_illuminant = np.maximum(resampled_illuminant, min_threshold)
    
    # Calculate reflectance
    reflectance_spectrum = measured_spectrum / resampled_illuminant
    
    # Clip to reasonable reflectance range
    reflectance_spectrum = np.clip(reflectance_spectrum, 0, 2)
    
    print(f"Reflectance spectrum range: {np.min(reflectance_spectrum):.4f} - {np.max(reflectance_spectrum):.4f}")
    
    return reflectance_spectrum

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def help_hsi_normalization():
    """
    Affiche l’aide pour les fonctions de normalisation par illuminant HSI → Reflectance.
    """
    print("Hyperspectral Illuminant Normalization Functions")
    print("=" * 60)
    print("\n1) normalize_by_illuminant_cube(spectral_cube, cube_wavelengths,")
    print("       illuminant_spectrum, illuminant_wavelengths):")
    print("   • Input :")
    print("     - spectral_cube : np.ndarray (H, W, B) → cube radiance ou intensité mesurée.")
    print("     - cube_wavelengths : np.ndarray (B,) → longueurs d’onde des bandes du cube.")
    print("     - illuminant_spectrum : np.ndarray (L,) → spectre d’illumination.")
    print("     - illuminant_wavelengths : np.ndarray (L,) → longueurs d’onde de l’illuminant.")
    print("   • Résultat : reflectance_cube (H, W, B), obtenu en divisant chaque bande")
    print("     du cube par l’illuminant interpolé aux mêmes longueurs d’onde.")
    print("   • Clipping des valeurs reflectance dans [0, 2] pour gérer le bruit.\n")
    print("2) normalize_by_illuminant_spectrum(measured_spectrum, measured_wavelengths,")
    print("       illuminant_spectrum, illuminant_wavelengths):")
    print("   • Input :")
    print("     - measured_spectrum : np.ndarray (B,) → spectre mesuré d’un pixel ou d’un capteur.")
    print("     - measured_wavelengths : np.ndarray (B,) → longueurs d’onde associées.")
    print("     - illuminant_spectrum : np.ndarray (L,) → spectre d’illumination.")
    print("     - illuminant_wavelengths : np.ndarray (L,) → longueurs d’onde de l’illuminant.")
    print("   • Résultat : reflectance_spectrum (B,), obtenu en divisant le spectre mesuré")
    print("     par l’illuminant interpolé aux longueurs d’onde mesurées.")
    print("   • Clipping des valeurs reflectance dans [0, 2].\n")
    print("Exemples d’utilisation sont dans le bloc __main__.")

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from hsi_synthetic import planck_spectrum, generate_synthetic_hyperspectral_cube

    # 1) Afficher l’aide
    print("\n===== HELP (hsi_normalization) =====")
    help_hsi_normalization()

    # 2) Tester normalize_by_illuminant_cube
    print("\n===== Testing normalize_by_illuminant_cube =====")
    # Générer un petit cube synthétique (10×10×31 bandes 400–700 nm)
    cube, cube_wl, illum_used = generate_synthetic_hyperspectral_cube(
        height=10, width=10,
        wavelength_start=400, wavelength_end=700, wavelength_step=10,
        snr=0.0, noise_type='linear'
    )
    print(f"Synthetic cube shape: {cube.shape}, wavelengths {cube_wl.min():.0f}→{cube_wl.max():.0f} nm")

    # Créer un illuminant Planck 6500 K sur 380–780 nm
    full_wl = np.linspace(380, 780, 401)
    illum_spec = planck_spectrum(full_wl, 6500, snr=0.0)
    print(f"Illuminant (Planck 6500 K) computed on {full_wl.size} wavelengths")

    # Normaliser le cube par cet illuminant
    reflect_cube = normalize_by_illuminant_cube(
        spectral_cube=cube,
        cube_wavelengths=cube_wl,
        illuminant_spectrum=illum_spec,
        illuminant_wavelengths=full_wl
    )
    print(f"Reflectance cube shape: {reflect_cube.shape}")
    print(f"Reflectance range: {reflect_cube.min():.4f} → {reflect_cube.max():.4f}")

    # Afficher une bande radiance vs reflectance pour un pixel central
    i0, j0 = cube.shape[0]//2, cube.shape[1]//2
    band_index = cube.shape[2] // 2
    plt.figure(figsize=(6,3))
    plt.plot(cube_wl, cube[i0, j0, :], label='Radiance (mesurée)', marker='o')
    plt.plot(cube_wl, reflect_cube[i0, j0, :], label='Reflectance (normalisée)', marker='x')
    plt.title(f"Pixel Spectrum at (i={i0}, j={j0}), Band {cube_wl[band_index]:.0f} nm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Tester normalize_by_illuminant_spectrum
    print("\n===== Testing normalize_by_illuminant_spectrum =====")
    # Créer un spectre mesuré simple (par ex. spectre d’un pixel du cube)
    measured_spec = cube[i0, j0, :].copy()
    measured_wl = cube_wl.copy()
    print(f"Measured spectrum length: {measured_spec.size}")

    # Réutiliser le même illuminant sur full_wl
    reflect_spec = normalize_by_illuminant_spectrum(
        measured_spectrum=measured_spec,
        measured_wavelengths=measured_wl,
        illuminant_spectrum=illum_spec,
        illuminant_wavelengths=full_wl
    )
    print(f"Reflectance spectrum length: {reflect_spec.size}")
    print(f"Reflectance values: min={reflect_spec.min():.4f}, max={reflect_spec.max():.4f}")

    # Tracer le spectre mesuré vs le spectre reflectance
    plt.figure(figsize=(6,3))
    plt.plot(measured_wl, measured_spec, label='Measured Spectrum', color='tab:orange')
    plt.plot(measured_wl, reflect_spec, label='Reflectance Spectrum', color='tab:blue')
    plt.title(f"Single-Spectrum Normalization at Pixel (i={i0}, j={j0})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
