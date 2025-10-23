import numpy as np
import os
import pandas as pd

def extract_wavenumbers(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().rstrip('\n').split('\t')
            print(f"{file_path} → wavenumber columns: {len(first_line)}")
            wavenumbers = np.array([float(x) for x in first_line[1:]])
            print(f"→ Final wavenumbers: {len(wavenumbers)}")
            return wavenumbers
    except Exception as e:
        print(f"Error extracting wavenumbers from {file_path}: {e}")
        return None

def load_file(file_path):
    try:
        return np.loadtxt(file_path, delimiter='\t', skiprows=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_txt_file(filepath):
    try:
        raw = np.genfromtxt(filepath, delimiter='\t', dtype=str)

        # # Debug: Show row lengths
        # for i, row in enumerate(raw):
        #     # print(f"{os.path.basename(filepath)} - Row {i} length: {len(row)}")

        # Filter out rows that are entirely empty
        filtered = [row for row in raw if any(cell.strip() != '' for cell in row)]

        # Ensure all rows have same number of columns
        row_lengths = [len(row) for row in filtered]
        if len(set(row_lengths)) != 1:
            raise ValueError(f"Inconsistent row lengths in {filepath}: {row_lengths}")

        raw = np.array(filtered)

        # Check for exact expected shape
        if raw.shape[0] != 4:
            raise ValueError(f"{filepath}: Expected 4 rows, got {raw.shape[0]}")
        if raw.shape[1] != 1732:
            raise ValueError(f"{filepath}: Expected 1732 columns, got {raw.shape[1]}")

        spectra = raw[1:, 1:].astype(np.float64)

        if spectra.shape != (3, 1731):
            raise ValueError(f"{filepath}: Cleaned shape {spectra.shape}, expected (3, 1731)")

        return spectra.T  # (1731, 3)

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None




def extract_unique_identifier(filename):
    parts = filename.split('_')
    return '_'.join(parts[:2])

def group_files_by_identifier(fp):
    files_grouped_by_identifier = {}
    for filename in os.listdir(fp):
        if filename.endswith('.txt'):
            identifier = extract_unique_identifier(filename)
            if identifier not in files_grouped_by_identifier:
                files_grouped_by_identifier[identifier] = []
            files_grouped_by_identifier[identifier].append(filename)
    return files_grouped_by_identifier

def average_arrays(arrays):
    return np.mean(np.hstack(arrays), axis=1)  # (1731, 3N) → mean across axis=1

def process_averaged_data(fp, files_grouped_by_identifier):
    averaged_rows = []
    sorted_identifiers = sorted(files_grouped_by_identifier.keys())
    wavenumbers = None

    for identifier in sorted_identifiers:
        filenames = files_grouped_by_identifier[identifier]
        arrays = [load_txt_file(os.path.join(fp, filename)) for filename in filenames]
        arrays = [arr for arr in arrays if arr is not None]
        if len(arrays) == 3:
            if wavenumbers is None:
                wavenumbers = extract_wavenumbers(os.path.join(fp, filenames[0]))
            averaged_row = average_arrays(arrays)
            averaged_rows.append(averaged_row)
        else:
            print(f"Warning: Expected 3 files for {identifier}, found {len(arrays)}")

    averaged_spectra = np.array(averaged_rows)
    return wavenumbers, np.transpose(averaged_spectra)

def process_unaveraged_data(fp, files_grouped_by_identifier):
    unaveraged_arrays = []
    filenames_per_column = []

    for identifier, filenames in files_grouped_by_identifier.items():
        arrays = [load_txt_file(os.path.join(fp, fn)) for fn in filenames]
        arrays = [arr for arr in arrays if arr is not None and arr.shape == (1731, 3)]
        if len(arrays) == 3:
            for fn, arr in zip(filenames, arrays):
                unaveraged_arrays.append(arr)
                filenames_per_column.extend([fn] * 3)  # 3 spectra per file

    return np.hstack(unaveraged_arrays), filenames_per_column


def load_data(data_dir, metadata_path=None, return_filenames=False):

    """
    Load wavenumbers, averaged spectra (if needed), and full unaveraged spectra per sample.
    
    Returns:
        - wavenumbers: shape (1731,)
        - averaged_spectra: shape (1731, N samples) [not used if analyzing per spectrum]
        - all_spectra: shape (1731, 3N spectra)
    """

    grouped_files = group_files_by_identifier(data_dir)

    if metadata_path:
        metadata_df = load_metadata(metadata_path)

        # Explicit mapping for both types
        mapped_ids = metadata_df['Sample_ID'].copy()
        mapped_ids[metadata_df['Type'] == 'Control'] = (
            'AdCo_' + metadata_df['Sample_ID'].str.split('_').str[1]
        )
        mapped_ids[metadata_df['Type'] == 'DM1'] = (
            'DM1_' + metadata_df['Sample_ID'].str.split('_').str[1]
        )
        metadata_df['FilePrefix'] = mapped_ids

        valid_ids = set(metadata_df['FilePrefix'])
        grouped_files = {k: v for k, v in grouped_files.items() if k in valid_ids}

        if not grouped_files:
            raise ValueError("No matching .txt files found for DM1 or Control samples after filtering.")

        print(f"✅ Loaded {len(grouped_files)} valid sample groups from metadata.")

    wavenumbers, averaged_spectra = process_averaged_data(data_dir, grouped_files)
    all_spectra, filenames_per_column = process_unaveraged_data(data_dir, grouped_files)
    
    print("Loaded averaged_spectra shape:", averaged_spectra.shape)
    print("Loaded all_spectra shape:", all_spectra.shape)
    
    if return_filenames:
        return wavenumbers, averaged_spectra, all_spectra, filenames_per_column
    else:
        return wavenumbers, averaged_spectra, all_spectra
# # THIS VERSION LOADS DM1 AND CONTROLS
# def load_metadata(fp):
#     y_label_df = pd.read_csv(fp)
#     y_label_df['Sample_ID'] = y_label_df['Sample_ID'].astype(str)
#     y_label_df = y_label_df[y_label_df['Type'].isin(['DM1', 'Control'])].copy()
#     return y_label_df
# # LOADS DM1 Samples only
def load_metadata(fp):
    y_label_df = pd.read_csv(fp)
    y_label_df['Sample_ID'] = y_label_df['Sample_ID'].astype(str)
    y_label_df = y_label_df[y_label_df['Type'] == 'DM1'].copy()
    return y_label_df
