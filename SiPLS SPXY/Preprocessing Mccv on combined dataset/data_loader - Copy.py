import numpy as np
import os
import pandas as pd

def extract_wavenumbers(file_path):
    """
    Extract the wavenumbers from the first row of the file, excluding the first column.

    Parameters:
    file_path (str): Path to the file to extract wavenumbers from.

    Returns:
    np.ndarray: The wavenumbers as a NumPy array.
    """
    try:
        with open(file_path, 'r') as file:
            # Read the first line and split by tab
            first_line = file.readline().strip().split('\t')
            # Convert to floats and exclude the first column
            wavenumbers = np.array([float(x) for x in first_line[1:]])
            return wavenumbers
    except Exception as e:
        print(f"Error extracting wavenumbers from {file_path}: {e}")
        return None

def load_file(file_path):
    """
    Load a single file, skipping the first row if it contains headers.

    Parameters:
    file_path (str): Path to the file to load.

    Returns:
    np.ndarray: Loaded data as a NumPy array.
    """
    try:
        return np.loadtxt(file_path, delimiter='\t', skiprows=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
def load_txt_file(filepath):
    """
    Load a single spectral .txt file into (num_points, 3) numpy array.

    Parameters:
    filepath (str): Full path to the .txt file.

    Returns:
    np.ndarray: A transposed array with shape (num_points, spectra).
    """
    try:
        data = np.loadtxt(filepath, delimiter='\t', skiprows=1)
        return data[:, 1:].T  # drop the first column, then transpose to (num_points, spectra)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_unique_identifier(filename):
    """
    Extract the unique identifier from a filename.

    Parameters:
    filename (str): The filename to process.

    Returns:
    str: The unique identifier extracted from the filename.
    """
    parts = filename.split('_')
    return '_'.join(parts[:2])

def group_files_by_identifier(fp):
    """
    Group files by their unique identifiers.

    Parameters:
    fp (str): The file path containing the files.

    Returns:
    dict: A dictionary where keys are unique identifiers and values are lists of filenames.
    """
    files_grouped_by_identifier = {}
    for filename in os.listdir(fp):
        if filename.endswith('.txt'):
            identifier = extract_unique_identifier(filename)
            if identifier not in files_grouped_by_identifier:
                files_grouped_by_identifier[identifier] = []
            files_grouped_by_identifier[identifier].append(filename)
    return files_grouped_by_identifier

def average_arrays(arrays):
    """
    Average multiple arrays by excluding the first row and calculating the mean of the remaining rows.

    Parameters:
    arrays (list): A list of NumPy arrays to average.

    Returns:
    np.ndarray: The mean of the remaining rows across all arrays.
    """
    combined_array = np.vstack([arr[1:, 1:] for arr in arrays])  # Exclude the first row and column, then stack
    mean_values = np.mean(combined_array, axis=0)  # Calculate the mean of the remaining rows
    
    return mean_values

def process_averaged_data(fp, files_grouped_by_identifier):
    """
    Process the grouped files to produce averaged spectra.

    Parameters:
    fp (str): The file path containing the files.
    files_grouped_by_identifier (dict): A dictionary of grouped files by unique identifiers.

    Returns:
    tuple: A tuple containing:
        - wavenumbers (np.ndarray): The X-axis numerical labels.
        - averaged_spectra (np.ndarray): Averaged spectra.
    """
    averaged_rows = []
    sorted_identifiers = sorted(files_grouped_by_identifier.keys())
    wavenumbers = None

    for identifier in sorted_identifiers:
        filenames = files_grouped_by_identifier[identifier]
        arrays = [load_file(os.path.join(fp, filename)) for filename in filenames]
        arrays = [arr for arr in arrays if arr is not None]  # Filter out any failures
        if len(arrays) == 3:  # Ensure exactly three arrays before averaging
            if wavenumbers is None:  # Extract wavenumbers from the first valid array
                wavenumbers = extract_wavenumbers(os.path.join(fp, filenames[0]))
            averaged_row = average_arrays(arrays)
            averaged_rows.append(averaged_row)
        else:
            print(f"Warning: Expected 3 files for {identifier}, found {len(arrays)}")

    averaged_spectra = np.array(averaged_rows)
    
    return wavenumbers, np.transpose(averaged_spectra)

def process_unaveraged_data(fp, files_grouped_by_identifier):
    """
    Process the grouped files to produce unaveraged spectra.

    Parameters:
    fp (str): The file path containing the files.
    files_grouped_by_identifier (dict): A dictionary of grouped files by unique identifiers.

    Returns:
    np.ndarray: Unaveraged spectra.
    """
    unaveraged_arrays = []
    sorted_identifiers = sorted(files_grouped_by_identifier.keys())

    for identifier in sorted_identifiers:
        filenames = files_grouped_by_identifier[identifier]
        arrays = [load_file(os.path.join(fp, filename)) for filename in filenames]
        arrays = [arr for arr in arrays if arr is not None]  # Filter out any failures
        if len(arrays) == 3:  # Verify that we have the expected number of files
            for arr in arrays:
                unaveraged_arrays.append(arr[:, 1:])  # Exclude the first column
        else:
            print(f"Warning: Expected 3 files for {identifier}, found {len(arrays)}")

    unaveraged_spectra = np.vstack(unaveraged_arrays)  # No need to transpose, already in correct shape
    
    return np.transpose(unaveraged_spectra)

def load_data(fp):
    """
    Load and process data from the raw instrument form.

    Parameters:
    fp (str): The file path containing the raw data.

    Returns:
    tuple: A tuple containing:
        - wavenumbers (np.ndarray): The X-axis numerical labels.
        - averaged_spectra (np.ndarray): Averaged spectra.
        - unaveraged_spectra (np.ndarray): Unaveraged spectra.
    """
    files_grouped_by_identifier = group_files_by_identifier(fp)
    
    wavenumbers, averaged_spectra = process_averaged_data(fp, files_grouped_by_identifier)
    all_spectra = process_unaveraged_data(fp, files_grouped_by_identifier)
    
    # Remove Repeated Values on First Row [1:,;]
    return wavenumbers, averaged_spectra[1:,:], all_spectra[1:,:] 




def load_metadata(fp):
    """
    Parameters
    ----------
    fp (str): The filepath to the Metadata CSV file.

    Returns
    -------
    y_label_df : dataframe of metadata with SampleID column
    """
    y_label_df = pd.read_csv(fp)
    y_label_df['SampleID'] = y_label_df.index.astype(str)  # <-- ADD THIS LINE
    return y_label_df

    
    
# def load_metadata(fp):
#     y_label_df = pd.read_csv(fp)
#     # y_label_repeated_df = pd.DataFrame(np.repeat(y_label_df.values, 9, axis=0), columns=y_label_df.columns)
    
#     return y_label_df #, y_label_repeated_df
    



