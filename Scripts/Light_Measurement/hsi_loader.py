import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import re

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def load_envi_image(bin_file_path, hdr_file_path=None):
    """
    Load a hyperspectral image from ENVI format files (.bin/.hdr)
    
    Parameters:
    -----------
    bin_file_path : str
        Path to the binary file containing the raw hyperspectral data (.bin or .dat)
    hdr_file_path : str, optional
        Path to the header file (.hdr). If None, assumes same name as bin_file with .hdr extension
    
    Returns:
    --------
    hyperspectral_cube : np.ndarray
        3D numpy array with shape (height, width, bands)
    wavelengths : np.ndarray
        1D numpy array containing the wavelengths for each band
    """
    import os

    # If header file path is not provided, construct it from binary file path
    if hdr_file_path is None:
        hdr_file_path = os.path.splitext(bin_file_path)[0] + '.hdr'
    
    # Check if files exist
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"Binary file not found: {bin_file_path}")
    if not os.path.exists(hdr_file_path):
        raise FileNotFoundError(f"Header file not found: {hdr_file_path}")
    
    # Parse the header file to extract metadata
    header_info = {}
    wavelengths_list = []
    in_wavelength_block = False

    with open(hdr_file_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            # Detect the start of the wavelength block
            if line.lower().startswith('wavelength') and '=' in line and line.endswith('{'):
                in_wavelength_block = True
                continue
            # If inside the wavelength block, collect values until '}' is found
            if in_wavelength_block:
                if line.endswith('}'):
                    in_wavelength_block = False
                    continue
                # Remove any trailing commas or braces
                value = line.rstrip(',').strip()
                if value:
                    wavelengths_list.append(float(value))
                continue
            # Outside wavelength block, parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                # Only keep non-wavelength entries here
                if not key == 'wavelength':
                    header_info[key] = value
    
    # Extract essential parameters
    try:
        samples = int(header_info['samples'])  # Width
        lines = int(header_info['lines'])      # Height
        bands = int(header_info['bands'])      # Number of spectral bands
        
        # Data type mapping
        data_type_map = {
            '1': np.uint8,
            '2': np.int16,
            '3': np.int32,
            '4': np.float32,
            '5': np.float64,
            '12': np.uint16,
            '13': np.uint32
        }
        data_type = data_type_map.get(header_info.get('data type', '4'), np.float32)
        
        # Interleave format (default is BSQ - Band Sequential)
        interleave = header_info.get('interleave', 'bsq').lower()
        
    except KeyError as e:
        raise ValueError(f"Missing required header parameter: {e}")
    
    # Convert wavelengths list to numpy array
    if wavelengths_list:
        wavelengths = np.array(wavelengths_list, dtype=np.float32)
    else:
        print("Warning: No wavelength information found in header. Creating default range.")
        wavelengths = np.arange(bands, dtype=np.float32)
    
    # Load binary data
    data = np.fromfile(bin_file_path, dtype=data_type)
    
    # Reshape data according to interleave format
    if interleave == 'bsq':  # Band Sequential
        hyperspectral_cube = data.reshape((bands, lines, samples))
        hyperspectral_cube = np.transpose(hyperspectral_cube, (1, 2, 0))  # (lines, samples, bands)
    elif interleave == 'bil':  # Band Interleaved by Line
        hyperspectral_cube = data.reshape((lines, bands, samples))
        hyperspectral_cube = np.transpose(hyperspectral_cube, (0, 2, 1))  # (lines, samples, bands)
    elif interleave == 'bip':  # Band Interleaved by Pixel
        hyperspectral_cube = data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unsupported interleave format: {interleave}")
    
    return hyperspectral_cube, wavelengths

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def load_spectrum_file(file_path):
    """
    Load spectral data from a .spd or .csv file.
    
    The function supports:
    - Files with or without header lines (column names)
    - Flexible separators (comma, semicolon, tab, space)
    - Skips comment lines starting with '#'
    
    Parameters:
    -----------
    file_path : str
        Path to the spectrum file (.spd or .csv)
        File format: first column = wavelengths, second column = values
    
    Returns:
    --------
    spectrum_values : np.ndarray
        1D numpy array containing the spectral values
    wavelengths : np.ndarray
        1D numpy array containing the corresponding wavelengths
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Spectrum file not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext not in ['.csv', '.spd']:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Try different separators and auto-detect header
    separators = [',', ';', '\t', ' ']
    data = None
    
    for sep in separators:
        try:
            # Try reading without header first
            df = pd.read_csv(file_path, sep=sep, comment='#', header=None)
            
            # Try converting first row to float to check if it's a header
            try:
                float(df.iloc[0, 0])
                has_header = False
            except:
                has_header = True
            
            # Reload with header if needed
            if has_header:
                df = pd.read_csv(file_path, sep=sep, comment='#', header=0)
            
            if df.shape[1] >= 2:
                data = df
                break
        except Exception:
            continue

    if data is None or data.shape[1] < 2:
        raise ValueError("Could not parse file or file has less than 2 columns")

    try:
        wavelengths = data.iloc[:, 0].astype(np.float32).values
        spectrum_values = data.iloc[:, 1].astype(np.float32).values
    except Exception as e:
        raise ValueError(f"Could not convert spectral values to float: {e}")
    
    # Remove NaNs
    valid = ~(np.isnan(wavelengths) | np.isnan(spectrum_values))
    wavelengths = wavelengths[valid]
    spectrum_values = spectrum_values[valid]
    
    # Sort by wavelength
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    spectrum_values = spectrum_values[sort_idx]
    
    return spectrum_values, wavelengths

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def resample_spectrum_to_wavelengths(spectrum_values, original_wavelengths, target_wavelengths, 
                                   interpolation_method='linear'):
    """
    Resample spectral data to new wavelength grid using interpolation
    
    Parameters:
    -----------
    spectrum_values : np.ndarray
        Original spectral values
    original_wavelengths : np.ndarray
        Original wavelengths corresponding to spectrum_values
    target_wavelengths : np.ndarray
        Target wavelengths for resampling
    interpolation_method : str, optional
        Interpolation method ('linear', 'cubic', 'nearest'). Default is 'linear'
    
    Returns:
    --------
    resampled_spectrum : np.ndarray
        Resampled spectral values at target wavelengths
    """
    
    # Input validation
    if len(spectrum_values) != len(original_wavelengths):
        raise ValueError("Length of spectrum_values and original_wavelengths must match")
    
    # Sort original data by wavelength (required for interpolation)
    sort_indices = np.argsort(original_wavelengths)
    sorted_wavelengths = original_wavelengths[sort_indices]
    sorted_values = spectrum_values[sort_indices]
    
    # Check for wavelength range compatibility
    target_min, target_max = np.min(target_wavelengths), np.max(target_wavelengths)
    original_min, original_max = np.min(sorted_wavelengths), np.max(sorted_wavelengths)
    
    if target_min < original_min or target_max > original_max:
        print(f"Warning: Target wavelength range [{target_min:.2f}, {target_max:.2f}] "
              f"extends beyond original range [{original_min:.2f}, {original_max:.2f}]. "
              f"Extrapolation will be used.")
    
    # Create interpolation function
    try:
        if interpolation_method == 'linear':
            interp_func = interp1d(sorted_wavelengths, sorted_values, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
        elif interpolation_method == 'cubic':
            interp_func = interp1d(sorted_wavelengths, sorted_values, kind='cubic', 
                                 bounds_error=False, fill_value='extrapolate')
        elif interpolation_method == 'nearest':
            interp_func = interp1d(sorted_wavelengths, sorted_values, kind='nearest', 
                                 bounds_error=False, fill_value='extrapolate')
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
        
        # Perform interpolation
        resampled_spectrum = interp_func(target_wavelengths)
        
    except Exception as e:
        raise ValueError(f"Interpolation failed: {str(e)}")
    
    return resampled_spectrum

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

def resample_spectra_to_common_wavelengths(spectra_list, wavelength_list, 
                                         interpolation_method='linear'):
    """
    Resample multiple spectra to their common wavelength range
    
    Parameters:
    -----------
    spectra_list : list of np.ndarray
        List of spectral value arrays
    wavelength_list : list of np.ndarray
        List of corresponding wavelength arrays
    interpolation_method : str, optional
        Interpolation method ('linear', 'cubic', 'nearest'). Default is 'linear'
    
    Returns:
    --------
    resampled_spectra : list of np.ndarray
        List of resampled spectral values at common wavelengths
    common_wavelengths : np.ndarray
        Common wavelength grid used for resampling
    """
    
    # Input validation
    if len(spectra_list) != len(wavelength_list):
        raise ValueError("Number of spectra and wavelength arrays must match")
    
    if len(spectra_list) == 0:
        raise ValueError("Input lists cannot be empty")
    
    # Find the common wavelength range (intersection of all ranges)
    wavelength_mins = []
    wavelength_maxs = []
    
    for wavelengths in wavelength_list:
        wavelength_mins.append(np.min(wavelengths))
        wavelength_maxs.append(np.max(wavelengths))
    
    # Common range is the intersection
    common_min = np.max(wavelength_mins)
    common_max = np.min(wavelength_maxs)
    
    if common_min >= common_max:
        raise ValueError("No common wavelength range found among all spectra")
    
    print(f"Common wavelength range: [{common_min:.2f}, {common_max:.2f}]")
    
    # Determine common wavelength grid resolution
    # Use the finest resolution among all input spectra within the common range
    min_resolution = float('inf')
    
    for wavelengths in wavelength_list:
        # Filter to common range
        in_range = (wavelengths >= common_min) & (wavelengths <= common_max)
        range_wavelengths = wavelengths[in_range]
        
        if len(range_wavelengths) > 1:
            resolution = np.median(np.diff(np.sort(range_wavelengths)))
            min_resolution = min(min_resolution, resolution)
    
    # Create common wavelength grid
    num_points = int((common_max - common_min) / min_resolution) + 1
    common_wavelengths = np.linspace(common_min, common_max, num_points)
    
    print(f"Common wavelength grid: {len(common_wavelengths)} points, "
          f"resolution: {min_resolution:.3f}")
    
    # Resample all spectra to common grid
    resampled_spectra = []
    
    for i, (spectrum, wavelengths) in enumerate(zip(spectra_list, wavelength_list)):
        try:
            resampled = resample_spectrum_to_wavelengths(
                spectrum, wavelengths, common_wavelengths, interpolation_method
            )
            resampled_spectra.append(resampled)
            
        except Exception as e:
            raise ValueError(f"Failed to resample spectrum {i}: {str(e)}")
    
    return resampled_spectra, common_wavelengths

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

# Example usage and testing functions
def help_hsi_loader():
    """
    Test function to demonstrate usage of the hyperspectral processing functions
    """
    print("Hyperspectral Image and Spectral Data Processing Functions")
    print("=" * 60)
    print("\nExample usage:")
    
    print("\n1. Loading ENVI hyperspectral image:")
    print("   cube, wavelengths = load_envi_image('image.dat', 'image.hdr')")
    print("   # Returns: 3D cube (height, width, bands) and wavelength vector")
    
    print("\n2. Loading spectrum file:")
    print("   values, wavelengths = load_spectrum_file('spectrum.csv')")
    print("   # Returns: spectrum values and corresponding wavelengths")
    
    print("\n3. Resampling to specific wavelengths:")
    print("   target_wl = np.linspace(400, 800, 100)")
    print("   resampled = resample_spectrum_to_wavelengths(values, wavelengths, target_wl)")
    
    print("\n4. Resampling multiple spectra to common grid:")
    print("   resampled_list, common_wl = resample_spectra_to_common_wavelengths(")
    print("       [spectrum1, spectrum2], [wavelengths1, wavelengths2])")

#==============================================================================================================================================================#
#==============================================================================================================================================================#
#==============================================================================================================================================================#

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hsi_synthetic import *

    lambdas = np.linspace(380, 780, int(400 / 5))
    values  = planck_spectrum(lambdas, 6500, 0.01, 'poisson')
    waves   = np.linspace(380, 780, int(400 / 1))

    resampled_linear    = resample_spectrum_to_wavelengths(values, lambdas, waves, 'linear')
    resampled_cubic     = resample_spectrum_to_wavelengths(values, lambdas, waves, 'cubic')
    resampled_nearest   = resample_spectrum_to_wavelengths(values, lambdas, waves, 'nearest')

    plt.plot(waves, resampled_linear, alpha = 0.33, lw = 2, color='red',   label="Linear")
    plt.plot(waves, resampled_cubic, alpha = 0.33, lw = 2, color='green',  label="Cubic")
    plt.plot(waves, resampled_nearest, alpha = 0.33, lw = 2, color='blue', label="Nearest")

    plt.legend()
    plt.show()