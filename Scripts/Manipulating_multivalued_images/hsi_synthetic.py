import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hsi_loader import resample_spectrum_to_wavelengths

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def add_noise_to_spectrum(spectrum, snr=0.05, noise_type='linear'):
    """
    Add noise to a spectrum with specified SNR and noise type
    
    Parameters:
    -----------
    spectrum : np.ndarray
        Input spectrum
    snr : float
        Signal-to-noise ratio (noise standard deviation relative to signal)
    noise_type : str
        Type of noise: 'linear', 'gaussian', 'poisson', 'multiplicative'
    
    Returns:
    --------
    noisy_spectrum : np.ndarray
        Spectrum with added noise
    """
    
    if snr <= 0:
        return spectrum.copy()
    
    # Calculate noise amplitude based on SNR
    signal_power = np.mean(spectrum)
    noise_amplitude = signal_power * snr
    
    if noise_type == 'linear' or noise_type == 'gaussian':
        # Additive Gaussian noise
        noise = np.random.normal(0, noise_amplitude, spectrum.shape)
        noisy_spectrum = spectrum + noise
        
    elif noise_type == 'poisson':
        # Poisson noise (photon noise)
        # Scale spectrum to avoid numerical issues
        scaled_spectrum = spectrum * 1000
        noisy_scaled = np.random.poisson(scaled_spectrum).astype(np.float32)
        noisy_spectrum = noisy_scaled / 1000
        # Add additional Gaussian noise if SNR is specified
        if snr > 0:
            noise = np.random.normal(0, noise_amplitude, spectrum.shape)
            noisy_spectrum += noise
            
    elif noise_type == 'multiplicative':
        # Multiplicative noise
        noise = np.random.normal(1, snr, spectrum.shape)
        noisy_spectrum = spectrum * noise
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy_spectrum

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def planck_spectrum(wavelengths, temperature, snr=0.05, noise_type='linear'):
    """
    Calculate Planck's blackbody spectrum for given temperature with optional noise
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelengths in nanometers
    temperature : float
        Temperature in Kelvin
    snr : float, optional
        Signal-to-noise ratio for added noise (default: 0.05)
    noise_type : str, optional
        Type of noise to add (default: 'linear')
    
    Returns:
    --------
    spectrum : np.ndarray
        Spectral radiance values (normalized)
    """
    
    # Convert wavelengths to meters
    wavelengths_m = wavelengths * 1e-9
    
    # Physical constants
    h = 6.62607015e-34  # Planck constant
    c = 299792458       # Speed of light
    k = 1.380649e-23    # Boltzmann constant
    
    # Planck's law
    numerator = 2 * h * c**2 / wavelengths_m**5
    denominator = np.exp((h * c) / (wavelengths_m * k * temperature)) - 1
    
    spectrum = numerator / denominator
    
    # Normalize
    spectrum = spectrum / np.max(spectrum)
    
    # Add noise if specified
    if snr > 0:
        spectrum = add_noise_to_spectrum(spectrum, snr, noise_type)
        # Ensure positive values
        spectrum = np.maximum(spectrum, 1e-10)
        # Renormalize after noise addition
        spectrum = spectrum / np.max(spectrum)
    
    return spectrum

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def generate_synthetic_macbeth_cube(
    wavelength_start=380, wavelength_end=780, wavelength_step=5,
    illuminant_spectrum=None, illuminant_wavelengths=None,
    snr=0.05, noise_type='linear'
):
    """
    Génère un cube hyperspectral synthétique représentant une mire Macbeth ColorChecker
    (4×6 patches), avec chaque patch de 64×64 px, 4 px de marge entre patchs, et
    4 px de marge grise (15%) tout autour.

    Paramètres :
    ------------
    wavelength_start, wavelength_end : float
        Domaine spectral en nm (défaut 380–780 nm).
    wavelength_step : float
        Pas spectral en nm (défaut 5 nm).
    illuminant_spectrum : np.ndarray ou None
        Spectre d’illuminant. Si None, génère D50 (5000 K).
    illuminant_wavelengths : np.ndarray ou None
        Longueurs d’onde associées à l’illuminant.
    snr : float
        Rapport signal sur bruit pour ajouter du bruit (défaut 0.05).
    noise_type : str
        Type de bruit : 'linear', 'gaussian', 'poisson', 'multiplicative'.

    Retour :
    --------
    synthetic_cube : np.ndarray (H, W, B)
        Cube hyperspectral (radiance) de la mire Macbeth synthétique.
    wavelengths : np.ndarray (B,)
        Vecteur des longueurs d’onde [start, start+step, …, end].
    illuminant_used : np.ndarray (B,)
        Spectre d’illuminant échantillonné sur ces longueurs d’onde.
    """

    # 1) Paramètres spatiaux fixes
    patch_size = 64       # taille de chaque patch en pixels
    margin = 4            # marge interne et externe en pixels
    n_rows, n_cols = 4, 6 # 4 lignes × 6 colonnes de patches

    # Calculer dimensions totales (marges externes + patches + marges internes)
    height = margin + n_rows * patch_size + (n_rows - 1) * margin + margin
    #        = 4 + 4*64 + 3*4 + 4 = 276
    width  = margin + n_cols * patch_size + (n_cols - 1) * margin + margin
    #        = 4 + 6*64 + 5*4 + 4 = 412

    # 2) Construction du vecteur des longueurs d’onde
    wavelengths = np.arange(
        wavelength_start,
        wavelength_end + wavelength_step,
        wavelength_step,
        dtype=np.float32
    )
    num_bands = wavelengths.size

    print(f"Generating synthetic Macbeth ColorChecker cube ({height}×{width}×{num_bands})")
    print(f"Wavelengths: {wavelength_start}–{wavelength_end} nm, step {wavelength_step} nm")

    # 3) Génération ou rééchantillonnage de l’illuminant
    if illuminant_spectrum is None:
        print("No illuminant provided → generating D50 illuminant (5000 K)")
        illuminant_wavelengths = wavelengths.copy()
        illuminant_spectrum = planck_spectrum(illuminant_wavelengths, 5000,
                                              snr=0, noise_type='linear')
    illuminant_resampled = resample_spectrum_to_wavelengths(
        illuminant_spectrum, illuminant_wavelengths, wavelengths
    )

    # 4) Calcul du radiance de marge grise (15% reflectance)
    gray_margin_reflectance = 0.15
    margin_radiance = gray_margin_reflectance * illuminant_resampled

    # 5) Initialisation du cube : on le remplit d’abord avec la marge grise partout
    synthetic_cube = np.tile(
        margin_radiance[np.newaxis, np.newaxis, :],
        (height, width, 1)
    ).astype(np.float32)

    # 6) Définition des spectres de réflectance pour les 24 patches
    def gaussian_peak(center, sigma=30.0, amplitude=1.0):
        return amplitude * np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

    patch_reflectances = []

    # --- Ligne 1 : Dark Skin, Light Skin, Blue Sky, Foliage, Blue Flower, Bluish Green ---
    ds = 0.4 * gaussian_peak(660, sigma=50)
    patch_reflectances.append(ds / ds.max())

    ls = 0.6 * gaussian_peak(580, sigma=60)
    ls += 0.2 * gaussian_peak(700, sigma=100)
    patch_reflectances.append(ls / ls.max())

    bs = 0.7 * gaussian_peak(450, sigma=40)
    patch_reflectances.append(bs / bs.max())

    fg = 0.6 * gaussian_peak(550, sigma=50)
    nir_fg = np.where(wavelengths > 700, 0.5, 0.0)
    foliage = fg + nir_fg
    patch_reflectances.append(foliage / foliage.max())

    bf = 0.5 * gaussian_peak(430, sigma=30)
    bf += 0.2 * gaussian_peak(550, sigma=80)
    patch_reflectances.append(bf / bf.max())

    bg = 0.6 * gaussian_peak(500, sigma=40)
    bg += 0.2 * gaussian_peak(550, sigma=80)
    patch_reflectances.append(bg / bg.max())

    # --- Ligne 2 : Orange, Purplish Blue, Moderate Red, Purple, Yellow Green, Orange Yellow ---
    orng = 0.7 * gaussian_peak(600, sigma=40)
    patch_reflectances.append(0.7 * orng / orng.max())

    pb = 0.5 * gaussian_peak(430, sigma=30) + 0.5 * gaussian_peak(660, sigma=30)
    patch_reflectances.append(0.8 * pb / pb.max())

    mr = 0.8 * gaussian_peak(600, sigma=40)
    mr += 0.1 * gaussian_peak(520, sigma=60)
    patch_reflectances.append(0.7 * mr / mr.max())

    pur = 0.4 * gaussian_peak(450, sigma=30) + 0.4 * gaussian_peak(650, sigma=30)
    patch_reflectances.append(0.5 * pur / pur.max())

    yg = 0.7 * gaussian_peak(570, sigma=40)
    patch_reflectances.append(0.5 * yg / yg.max())

    oy = 0.7 * gaussian_peak(590, sigma=40)
    oy += 0.1 * gaussian_peak(520, sigma=60)
    patch_reflectances.append(0.7 * oy / oy.max())

    # --- Ligne 3 : Blue, Green, Red, Yellow, Magenta, Cyan ---
    blu = 0.33 * gaussian_peak(450, sigma=30)
    patch_reflectances.append( 0.35 * blu / blu.max())

    grn = 0.9 * gaussian_peak(530, sigma=40)
    patch_reflectances.append(0.37 * grn / grn.max())

    rd = 0.9 * gaussian_peak(700, sigma=75)
    patch_reflectances.append(0.72 * rd / rd.max())

    ylw = 0.9 * gaussian_peak(600, sigma=100) + 0.3 * gaussian_peak(700, sigma=80)
    patch_reflectances.append(0.8 * ylw / ylw.max())

    mag = 0.7 * gaussian_peak(420, sigma=30) + 0.7 * gaussian_peak(700, sigma=80)
    patch_reflectances.append(0.8 * mag / mag.max())

    cyn = 0.8 * gaussian_peak(480, sigma=60) + 0.2 * gaussian_peak(780, sigma=30)
    patch_reflectances.append(0.46 * cyn / cyn.max())

    # --- Ligne 4 : Neutres (blanc → noir en 6 niveaux) ---
    gray_levels = [0.9, 0.6, 0.37, 0.2, 0.1, 0.02]
    for g in gray_levels:
        patch_reflectances.append(g * np.ones(num_bands, dtype=np.float32))

    assert len(patch_reflectances) == 24

    # 7) Placement des patches au sein du cube
    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols

        y0 = margin + row * (patch_size + margin)
        y1 = y0 + patch_size
        x0 = margin + col * (patch_size + margin)
        x1 = x0 + patch_size

        refl = patch_reflectances[idx]
        radiance_spectrum = refl * illuminant_resampled

        if snr > 0:
            radiance_spectrum = add_noise_to_spectrum(radiance_spectrum, snr, noise_type)
            radiance_spectrum = np.maximum(radiance_spectrum, 1e-10)

        synthetic_cube[y0:y1, x0:x1, :] = radiance_spectrum[np.newaxis, np.newaxis, :]

    print("Synthetic Macbeth cube generation completed")
    print(f"Radiance range: {synthetic_cube.min():.6f} – {synthetic_cube.max():.6f}")

    return synthetic_cube, wavelengths, illuminant_resampled

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def help_hsi_synthetic():
    """
    Affiche l'aide pour les fonctions de génération de données hyperspectrales synthétiques.
    """
    print("Synthetic Hyperspectral Cube Generation")
    print("=" * 60)
    print("\n1) add_noise_to_spectrum(spectrum, snr=0.05, noise_type='linear'):")
    print("   • Ajoute du bruit (gaussian, poisson, etc.) à un spectre 1D.")
    print("\n2) planck_spectrum(wavelengths, temperature, snr=0.05, noise_type='linear'):")
    print("   • Calcule le spectre du corps noir (Planck) à la température donnée.")
    print("   • Renormalise et ajoute du bruit si snr>0.")
    print("\n3) generate_synthetic_hyperspectral_cube(...) :")
    print("   • Génère un cube HxWxB simulé de radiance, avec différentes signatures (végétation, sol, eau).")
    print("   • Paramètres : dimension, domaine spectral, illuminant (ou D50 par défaut), SNR, type de bruit.")
    print("   • Retourne (cube, wavelengths, illuminant_used).")
    print("\nExemples d'utilisation dans le bloc __main__.")

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Afficher l'aide
    print("\n===== HELP (hsi_synthetic) =====")
    help_hsi_synthetic()

    # 2) Tester planck_spectrum
    wl = np.linspace(380, 780, 400)
    pl_spec = planck_spectrum(wl, 5500, snr=0.0)  # spectre “propre”
    print(f"\nPlanck spectrum computed at 5500 K - min {pl_spec.min():.4e}, max {pl_spec.max():.4e}")

    # Afficher le planck sans bruit
    plt.figure(figsize=(5,3))
    plt.plot(wl, pl_spec, color='orange')
    plt.title("Planck Spectrum @ 5500 K")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Radiance")
    plt.tight_layout()
    plt.show()

    # 3) Ajouter du bruit
    pl_noisy = planck_spectrum(wl, 5500, snr=0.05, noise_type='gaussian')
    plt.figure(figsize=(5,3))
    plt.plot(wl, pl_noisy, color='blue')
    plt.title("Planck Spectrum @ 5500 K (noisy)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Radiance")
    plt.tight_layout()
    plt.show()

    # 4) Tester generate_synthetic_hyperspectral_cube
    print("\n===== Generating a small synthetic cube (16x16, 401 bands) =====")
    cube, cube_wl, illum = generate_synthetic_macbeth_cube(
        wavelength_start=380, wavelength_end=780, wavelength_step=5,
        snr=0.02, noise_type='linear'
    )
    print(f"Cube shape: {cube.shape}, wavelengths {cube_wl.min():.0f}→{cube_wl.max():.0f} nm")

    # Afficher un pixel central (spectre radiance)
    i0, j0 = cube.shape[0]//2, cube.shape[1]//2
    plt.figure(figsize=(5,3))
    plt.plot(cube_wl, cube[i0, j0, :], color='green')
    plt.title(f"Pixel Spectrum at (i={i0}, j={j0})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (normalized)")
    plt.tight_layout()
    plt.show()

    # 5) Tester add_noise_to_spectrum séparément
    test_spec = np.linspace(0.1, 1.0, 100)
    noisy_linear = add_noise_to_spectrum(test_spec, snr=0.1, noise_type='linear')
    noisy_mult = add_noise_to_spectrum(test_spec, snr=0.1, noise_type='multiplicative')
    noisy_poisson = add_noise_to_spectrum(test_spec, snr=0.1, noise_type='poisson')
    plt.figure(figsize=(5,3))
    plt.plot(test_spec, label='Orig', color='black')
    plt.plot(noisy_linear, label='Linear noise', alpha=0.33)
    plt.plot(noisy_mult, label='Multiplicative noise', alpha=0.33)
    plt.plot(noisy_poisson, label='Poisson noise', alpha=0.33)
    plt.legend()
    plt.title("Noise Types on a Test Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()