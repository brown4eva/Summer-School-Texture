import numpy as np
from hsi_loader import resample_spectrum_to_wavelengths
from hsi_synthetic import planck_spectrum
import csv

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def get_cie_color_matching_functions(path=None):
    """
    Renvoie les vraies fonctions de correspondance CIE 1931 XYZ (observateur 2°)
    380-780 nm (pas 1 nm) :
    - si path=None → valeurs brutes codées.
    - si path donné → lecture CSV + interpolation.
    """
    wavelengths = np.arange(380, 781, 1, dtype=np.float32)

    if path is not None:
        wl_list, x_list, y_list, z_list = [], [], [], []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    wl, X, Y, Z = map(float, row[:4])
                except:
                    continue
                if 380 <= wl <= 780:
                    wl_list.append(wl)
                    x_list.append(X)
                    y_list.append(Y)
                    z_list.append(Z)
        src_wl = np.array(wl_list, dtype=np.float32)
        return (
            wavelengths,
            resample_spectrum_to_wavelengths(np.array(x_list), src_wl, wavelengths),
            resample_spectrum_to_wavelengths(np.array(y_list), src_wl, wavelengths),
            resample_spectrum_to_wavelengths(np.array(z_list), src_wl, wavelengths),
        )

    # Valeurs brutes (tronquées ici), issues de la table officielle CIE 1931 (2°)
    x_vals = """
    0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,
    0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,
    0.195360,0.142100,0.095640,0.057950,0.032010,0.014700,0.004900,0.002400,0.009300,
    0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,
    0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,
    1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,
    0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,
    0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,
    0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042,
    0.000030,0.000021,0.000015,0.000011,0.000008,0.000006,0.000004
    """
    y_vals = """
    0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,
    0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,
    0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,
    0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,
    0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,
    0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,
    0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,
    0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,
    0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015,
    0.000011,0.000008,0.000006,0.000004,0.000003,0.000002,0.000001,0.000001,0.000000
    """
    z_vals = """
    0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,
    1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,
    1.287640,1.041900,0.812950,0.616200,0.465180,0.353300,0.272000,0.212300,0.158200,
    0.111700,0.078250,0.057250,0.042160,0.029840,0.020300,0.013400,0.008750,0.005750,
    0.003900,0.002750,0.002100,0.001800,0.001650,0.001400,0.001100,0.001000,0.000800,
    0.000600,0.000340,0.000240,0.000190,0.000100,0.000049,0.000030,0.000020,0.000010,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    """

    x_bar = np.fromstring(x_vals, sep=",", dtype=np.float32)
    y_bar = np.fromstring(y_vals, sep=",", dtype=np.float32)
    z_bar = np.fromstring(z_vals, sep=",", dtype=np.float32)

    x_bar = resample_spectrum_to_wavelengths(x_bar, np.linspace(380,780, np.size(x_bar)), wavelengths)
    y_bar = resample_spectrum_to_wavelengths(y_bar, np.linspace(380,780, np.size(y_bar)), wavelengths)
    z_bar = resample_spectrum_to_wavelengths(z_bar, np.linspace(380,780, np.size(z_bar)), wavelengths)

    assert len(x_bar) == len(wavelengths) == len(y_bar) == len(z_bar), (
        f"Attendu {len(wavelengths)} échantillons, "
        f"obtenus {len(x_bar)}, {len(y_bar)}, {len(z_bar)}"
    )

    return wavelengths, x_bar, y_bar, z_bar

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def cube_to_xyz(reflectance_cube, cube_wavelengths, illuminant_spectrum=None, 
                           illuminant_wavelengths=None):
    """
    Convert reflectance cube to CIE XYZ color space
    
    Parameters:
    -----------
    reflectance_cube : np.ndarray
        3D reflectance cube with shape (height, width, bands)
    cube_wavelengths : np.ndarray
        Wavelengths corresponding to the spectral bands
    illuminant_spectrum : np.ndarray, optional
        Illuminant spectrum for color calculation. If None, uses D65 approximation
    illuminant_wavelengths : np.ndarray, optional
        Wavelengths for illuminant spectrum
    
    Returns:
    --------
    xyz_image : np.ndarray
        3D array with shape (height, width, 3) containing XYZ values
    """
    
    print("Converting reflectance cube to CIE XYZ...")
    print(f"Input cube shape: {reflectance_cube.shape}")
    
    # Get CIE color matching functions
    cie_wavelengths, x_bar, y_bar, z_bar = get_cie_color_matching_functions()
    
    # If no illuminant provided, use D65 approximation
    if illuminant_spectrum is None:
        print("No illuminant provided, using D65 approximation")
        # Simple D65 daylight approximation
        d65_temp = 6500  # Color temperature in Kelvin
        illuminant_spectrum = planck_spectrum(cie_wavelengths, d65_temp)
        illuminant_wavelengths = cie_wavelengths
    
    # Resample everything to common wavelength grid (CIE wavelengths)
    print("Resampling spectra to CIE wavelength grid...")
    
    # Resample color matching functions to cube wavelengths
    x_bar_resampled = resample_spectrum_to_wavelengths(x_bar, cie_wavelengths, cube_wavelengths)
    y_bar_resampled = resample_spectrum_to_wavelengths(y_bar, cie_wavelengths, cube_wavelengths)
    z_bar_resampled = resample_spectrum_to_wavelengths(z_bar, cie_wavelengths, cube_wavelengths)
    
    # Resample illuminant to cube wavelengths
    illuminant_resampled = resample_spectrum_to_wavelengths(
        illuminant_spectrum, illuminant_wavelengths, cube_wavelengths
    )
    
    # Calculate wavelength step for integration
    if len(cube_wavelengths) > 1:
        d_lambda = np.mean(np.diff(cube_wavelengths))
    else:
        d_lambda = 1.0
    
    # Initialize XYZ image
    height, width, bands = reflectance_cube.shape
    xyz_image = np.zeros((height, width, 3), dtype=np.float32)
    
    print("Calculating XYZ values...")
    
    # Calculate XYZ for each pixel
    # Integration: ∫ R(wl) * S(wl) * CMF(wl) dwl
    # where R = reflectance, S = illuminant, CMF = color matching function
    
    for i in range(height):
        for j in range(width):
            pixel_spectrum = reflectance_cube[i, j, :]
            
            # Multiply reflectance by illuminant
            radiance = pixel_spectrum * illuminant_resampled
            
            # Integrate with color matching functions
            X = np.sum(radiance * x_bar_resampled) * d_lambda
            Y = np.sum(radiance * y_bar_resampled) * d_lambda
            Z = np.sum(radiance * z_bar_resampled) * d_lambda
            
            xyz_image[i, j, :] = [X, Y, Z]
    
    # Normalize by illuminant white point
    white_point_Y = np.sum(illuminant_resampled * y_bar_resampled) * d_lambda
    xyz_image /= white_point_Y
    
    print(f"XYZ conversion completed")
    print(f"XYZ ranges - X: {np.min(xyz_image[:,:,0]):.4f} to {np.max(xyz_image[:,:,0]):.4f}")
    print(f"             Y: {np.min(xyz_image[:,:,1]):.4f} to {np.max(xyz_image[:,:,1]):.4f}")
    print(f"             Z: {np.min(xyz_image[:,:,2]):.4f} to {np.max(xyz_image[:,:,2]):.4f}")
    
    return xyz_image

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def xyz_to_rgb(xyz_image, transformation_matrix=None, gamma_correction=True, gamma_threshold=10e-6):
    """
    Convert XYZ image to RGB color space
    
    Parameters:
    -----------
    xyz_image : np.ndarray
        3D array with shape (height, width, 3) containing XYZ values
    transformation_matrix : np.ndarray, optional
        3x3 transformation matrix from XYZ to RGB. If None, uses sRGB matrix
    gamma_correction : bool, optional
        Whether to apply gamma correction (default: True)
    
    Returns:
    --------
    rgb_image : np.ndarray
        3D array with shape (height, width, 3) containing RGB values [0-1]
    """
    
    print("Converting XYZ to RGB...")
    
    # Default sRGB transformation matrix (XYZ to linear RGB)
    if transformation_matrix is None:
        # sRGB transformation matrix (D65 white point)
        transformation_matrix = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570]
        ], dtype=np.float32)
        print("Using sRGB transformation matrix")
    else:
        print("Using custom transformation matrix")
    
    # Get image dimensions
    height, width = xyz_image.shape[:2]
    
    # Reshape XYZ image to (N, 3) for matrix multiplication
    xyz_flat = xyz_image.reshape(-1, 3)
    
    # Apply transformation matrix: RGB = M * XYZ
    rgb_flat = np.dot(xyz_flat, transformation_matrix.T)
    
    # Reshape back to image format
    rgb_image = rgb_flat.reshape(height, width, 3)
    
    # Clip negative values
    rgb_image = np.maximum(rgb_image, 0)
    
    # Apply gamma correction for sRGB
    if gamma_correction != 1.0:
        if transformation_matrix is None:
            print("Applying sRGB gamma correction...")
            # sRGB gamma correction
            gamma_threshold = 0.0031308
            rgb_image = np.where(
                rgb_image <= gamma_threshold,
                12.92 * rgb_image,
                1.055 * np.power(rgb_image, 1/2.4) - 0.055
            )
        else:
            rgb_image = np.where(
                rgb_image <= gamma_threshold,
                rgb_image,
                np.power(rgb_image, 1/gamma_correction)
            )
        
    # Clip to [0, 1] range
    rgb_image /= np.max(rgb_image)
    
    print(f"RGB conversion completed")
    print(f"RGB ranges - R: {np.min(rgb_image[:,:,0]):.4f} to {np.max(rgb_image[:,:,0]):.4f}")
    print(f"             G: {np.min(rgb_image[:,:,1]):.4f} to {np.max(rgb_image[:,:,1]):.4f}")
    print(f"             B: {np.min(rgb_image[:,:,2]):.4f} to {np.max(rgb_image[:,:,2]):.4f}")
    
    return rgb_image

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def help_hsi_colorspace():
    """
    Affiche la documentation et exemples d'utilisation des fonctions de ce module
    """
    print("HSI — Color Space Conversion Functions")
    print("=" * 60)
    print("\n1. Récupérer les fonctions CIE 1931 XYZ :")
    print("   wavelengths, x_bar, y_bar, z_bar =")
    print("       get_cie_color_matching_functions(path=None)")
    print("   # Si path=None, on utilise les valeurs codées en dur; sinon on charge depuis CSV")

    print("\n2. Convertir un cube de réﬂectance en XYZ :")
    print("   xyz_image = reflectance_cube_to_xyz(")
    print("       reflectance_cube, cube_wavelengths,")
    print("       illuminant_spectrum=None, illuminant_wavelengths=None")
    print("   )")
    print("   # Si aucun illuminant n'est donné, on génère D65 via planck_spectrum()")

    print("\n3. Transformer une image XYZ en RGB :")
    print("   rgb_image = xyz_to_rgb(")
    print("       xyz_image, transformation_matrix=None, gamma_correction=True")
    print("   )")
    print("   # Par défaut, on utilise la matrice sRGB + correction gamma")

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def compute_chromaticity_coordinates(
    spectrum: np.ndarray,
    spectrum_wavelengths: np.ndarray,
    cmf_path: str = None
) -> tuple:
    """
    Calcule les coordonnées chromatiques (x, y) d'un spectre donné,
    selon l'observateur standard CIE 1931 (2°).

    Paramètres :
    ------------
    spectrum : np.ndarray (B,)
        Spectre spectral (puissance ou réflectance) à chaque longueur d'onde.
    spectrum_wavelengths : np.ndarray (B,)
        Longueurs d'onde correspondantes (en nm).
    cmf_path : str ou None
        Chemin vers un CSV CIE 1931 XYZ (colonnes λ, X, Y, Z). Si None, utilise
        la table codée en dur de get_cie_color_matching_functions().

    Retour :
    --------
    x, y : float
        Coordonnées chromatiques x et y.
    X, Y, Z : float
        Valeurs tristimulus CIE XYZ non normalisées.
    """
    # Charger les fonctions de correspondance colorimétriques
    cmf_wl, x_bar, y_bar, z_bar = get_cie_color_matching_functions(path=cmf_path)

    # Rééchantillonner les CMFs sur les longueurs d'onde du spectre
    x_r = resample_spectrum_to_wavelengths(x_bar, cmf_wl, spectrum_wavelengths)
    y_r = resample_spectrum_to_wavelengths(y_bar, cmf_wl, spectrum_wavelengths)
    z_r = resample_spectrum_to_wavelengths(z_bar, cmf_wl, spectrum_wavelengths)

    # Calcul du pas spectral moyen Δλ
    if spectrum_wavelengths.size > 1:
        d_lambda = np.mean(np.diff(spectrum_wavelengths))
    else:
        d_lambda = 1.0

    # Intégration (somme discrète) pour obtenir X, Y, Z
    X = np.sum(spectrum * x_r) * d_lambda
    Y = np.sum(spectrum * y_r) * d_lambda
    Z = np.sum(spectrum * z_r) * d_lambda

    # Coordonnées chromatiques
    denom = X + Y + Z
    if denom == 0:
        raise ValueError("Somme X+Y+Z nulle, impossible de calculer les chromaticités.")
    x = X / denom
    y = Y / denom

    return x, y, X, Y, Z

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

import os
import tkinter as tk
from tkinter import filedialog

def save_envi_cube(
    hyperspectral_cube: np.ndarray,
    wavelengths: np.ndarray,
    cube_illuminant: np.ndarray = None,
    cube_illuminant_wavelengths: np.ndarray = None
):
    """
    Sauvegarde un cube hyperspectral en format ENVI (.hdr + .bin) et, si fourni,
    enregistre également le spectre de l’illuminant dans un fichier CSV.

    Ouvre une boîte de dialogue pour choisir le chemin et le nom du fichier de sortie (sans extension).
    Génère :
      - <basename>.hdr : en-tête ENVI ASCII
      - <basename>.bin : données brutes en BSQ (float32)
      - <basename>_illuminant.csv (si cube_illuminant fourni) : colonnes [wavelength, illuminant]

    Paramètres :
    ------------
    hyperspectral_cube : np.ndarray
        Cube hyperspectral de forme (height, width, bands).
        Les données seront écrites en float32.
    wavelengths : np.ndarray
        Vecteur 1D des longueurs d’onde (en nm) correspondant à chaque bande.
        Doit vérifier wavelengths.size == bands.
    cube_illuminant : np.ndarray, optional
        Spectre de l’illuminant à sauvegarder (1D array de même longueur que wavelengths).
    cube_illuminant_wavelengths : np.ndarray, optional
        Wavelengths du spectre de l’illuminant (1D array).
        Doit vérifier cube_illuminant_wavelengths.size == cube_illuminant.size.
    """
    # Vérifications de base
    if hyperspectral_cube.ndim != 3:
        raise ValueError("Le cube hyperspectral doit être un tableau 3D (height, width, bands).")
    height, width, bands = hyperspectral_cube.shape
    if wavelengths.ndim != 1 or wavelengths.size != bands:
        raise ValueError("Le vecteur des longueurs d’onde doit être 1D et de taille égale au nombre de bandes.")

    # Si un illuminant est fourni, vérifier la cohérence
    if cube_illuminant is not None or cube_illuminant_wavelengths is not None:
        if cube_illuminant is None or cube_illuminant_wavelengths is None:
            raise ValueError("Pour sauvegarder le CSV de l’illuminant, fournissez à la fois "
                             "'cube_illuminant' et 'cube_illuminant_wavelengths'.")
        if (cube_illuminant.ndim != 1 or cube_illuminant_wavelengths.ndim != 1 or
                cube_illuminant.size != cube_illuminant_wavelengths.size):
            raise ValueError("Le spectre de l’illuminant et ses longueurs d’onde doivent être des vecteurs 1D de même taille.")

    # Ouvrir une fenêtre Tkinter masquée pour le filedialog
    root = tk.Tk()
    root.withdraw()

    # Demander à l'utilisateur où et sous quel nom enregistrer le fichier (sans extension)
    save_path = filedialog.asksaveasfilename(
        title="Enregistrer le cube hyperspectral en format ENVI",
        defaultextension=".hdr",
        filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")]
    )
    if not save_path:
        print("Opération annulée.")
        return

    # Construire le nom de base (sans extension .hdr)
    base, ext = os.path.splitext(save_path)
    if ext.lower() != ".hdr":
        base = save_path
    hdr_filename = base + ".hdr"
    bin_filename = base + ".bin"
    csv_filename = base + "_illuminant.csv"

    # Écrire le fichier binaire (.bin) en BSQ (float32, little endian)
    cube_to_save = hyperspectral_cube.astype(np.float32)
    with open(bin_filename, "wb") as f_bin:
        # Transposer en (bands, height, width) pour l’ordre BSQ, puis tofile()
        cube_to_save.transpose(2, 0, 1).tofile(f_bin)

    # Construire le contenu du header ENVI
    hdr_lines = [
        "ENVI",
        "description = {Sauvegarde ENVI du cube hyperspectral}",
        f"samples = {width}",
        f"lines   = {height}",
        f"bands   = {bands}",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
        "sensor type = Unknown",
        "wavelength units = Nanometers"
    ]

    # Préparer la liste des longueurs d'onde pour le header
    wl_strs = [f"{wl:.4f}" for wl in wavelengths]
    hdr_lines.append("wavelength = {")
    for i, wl in enumerate(wl_strs):
        sep = "," if i < len(wl_strs) - 1 else ""
        hdr_lines.append(f"    {wl}{sep}")
    hdr_lines.append("}")

    # Écrire le fichier .hdr
    with open(hdr_filename, "w") as f_hdr:
        for line in hdr_lines:
            f_hdr.write(line + "\n")

    # Si l’illuminant est fourni, sauvegarder le CSV
    if cube_illuminant is not None and cube_illuminant_wavelengths is not None:
        with open(csv_filename, "w", newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["wavelength", "illuminant"])
            for wl, val in zip(cube_illuminant_wavelengths, cube_illuminant):
                writer.writerow([f"{wl:.4f}", f"{val:.6e}"])
        print(f"Illuminant CSV saved: {csv_filename}")

    print(f"Fichiers ENVI sauvegardés :\n  {hdr_filename}\n  {bin_filename}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hsi_synthetic import generate_synthetic_macbeth_cube
    from hsi_loader import load_spectrum_file, resample_spectrum_to_wavelengths
    from tkinter import filedialog as fd

    # --- Démo pour get_cie_color_matching_functions() ---
    print("\n>>> Test de get_cie_color_matching_functions()")
    wl, x_bar, y_bar, z_bar = get_cie_color_matching_functions(fd.askopenfilename(title="CIE standard observer"))
    print(f"  - Longueurs d'onde: de {wl[0]} nm à {wl[-1]} nm ({len(wl)} points)")
    print(f"  - X_bar min/max : {x_bar.min():.6f}/{x_bar.max():.6f}")
    print(f"  - Y_bar min/max : {y_bar.min():.6f}/{y_bar.max():.6f}")
    print(f"  - Z_bar min/max : {z_bar.min():.6f}/{z_bar.max():.6f}")

    D65, d65_wav = load_spectrum_file(fd.askopenfilename(title="scene illuminant (CSV / SPD)"))

    # --- Génération d'un cube hyperspectral synthétique pour tester reflectance_cube_to_xyz() ---
    print("\n>>> Génération d'un petit cube synthétique (256×256×N)")
    cube, cube_wl, illuminant = generate_synthetic_macbeth_cube(
        wavelength_start=380, wavelength_end=780, wavelength_step=5, illuminant_spectrum= D65, illuminant_wavelengths= d65_wav,
        snr=0.01
    )

    save_envi_cube(cube, cube_wl, illuminant, cube_wl)
    xyz_img = cube_to_xyz(cube, cube_wl)
    print(f"  - XYZ image shape: {xyz_img.shape}")

    # Affichage d'un pixel exemple en XYZ
    print("  - Exemple pixel [0,0] en XYZ :", xyz_img[0, 0, :])

    # --- Conversion en RGB ---
    print("\n>>> Test de xyz_to_rgb()")
    rgb_img = xyz_to_rgb(xyz_img)
    print(f"  - RGB image shape: {rgb_img.shape}")
    print("  - Exemple pixel [0,0] en RGB :", rgb_img[0, 0, :])

    # Affichage d'une image (canal Y) en niveaux de gris pour visualiser
    plt.figure(figsize=(4,4))
    plt.imshow(xyz_img[:, :, 1], cmap='gray')
    plt.title("Canal Y (Luminosité) de l'image XYZ")
    plt.colorbar(label="Y normalisé")
    plt.tight_layout()
    plt.show()

    # Affichage d'une image RGB pour vérification visuelle
    plt.figure(figsize=(4,4))
    plt.imshow(rgb_img)
    plt.title("Image RGB (après conversion)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()