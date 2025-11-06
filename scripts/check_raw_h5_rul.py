"""Check RUL values in raw H5 files"""
import h5py
import numpy as np
import sys
if len(sys.argv) < 2:
    print("Usage: python check_raw_h5_rul.py <h5_file>")
    sys.exit(1)
h5_file = sys.argv[1]
print(f"\n{'='*70}")
print(f"Checking RUL in raw H5 file: {h5_file}")
print(f"{'='*70}\n")
with h5py.File(h5_file, 'r') as hdf:
    # Check what's in the file
    print("Keys in H5 file:")
    print(list(hdf.keys()))
    # N-CMAPSS format uses Y_dev and Y_test for RUL
    for split_key in ['Y_dev', 'Y_test']:
        if split_key in hdf:
            split_name = split_key.replace('Y_', '').upper()
            print(f"\n{'='*70}")
            print(f"{split_name} SPLIT (Key: {split_key})")
            print(f"{'='*70}")
            rul_data = hdf[split_key][:]
            print(f"\nRUL Statistics:")
            print(f"  Min RUL: {np.min(rul_data):.2f}")
            print(f"  Max RUL: {np.max(rul_data):.2f}")
            print(f"  Mean RUL: {np.mean(rul_data):.2f}")
            print(f"  Std RUL: {np.std(rul_data):.2f}")
            print(f"  Total samples: {len(rul_data):,}")
            print(f"  Shape: {rul_data.shape}")
            # Check how many exceed certain thresholds
            print(f"\nRUL Distribution:")
            print(f"  RUL > 94: {np.sum(rul_data > 94):,} samples ({np.sum(rul_data > 94)/len(rul_data)*100:.2f}%)")
            print(f"  RUL > 100: {np.sum(rul_data > 100):,} samples ({np.sum(rul_data > 100)/len(rul_data)*100:.2f}%)")
            print(f"  RUL > 125: {np.sum(rul_data > 125):,} samples ({np.sum(rul_data > 125)/len(rul_data)*100:.2f}%)")
            print(f"  RUL > 150: {np.sum(rul_data > 150):,} samples ({np.sum(rul_data > 150)/len(rul_data)*100:.2f}%)")
            print(f"  RUL > 200: {np.sum(rul_data > 200):,} samples ({np.sum(rul_data > 200)/len(rul_data)*100:.2f}%)")
            # Show some high RUL values
            high_rul = rul_data[rul_data > 100]
            if len(high_rul) > 0:
                print(f"\nSample of high RUL values (> 100):")
                unique_high = np.unique(high_rul)
                print(f"  First 30 unique values: {unique_high[:30]}")
                print(f"  Last 10 unique values: {unique_high[-10:]}")
        else:
            print(f"\n{split_key} not found in file")
print(f"\n{'='*70}\n")