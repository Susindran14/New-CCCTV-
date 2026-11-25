import os
import glob

def check_dataset_split():
    print("\n📊 DATASET DISTRIBUTION CHECK 📊\n")
    for folder in ['train', 'val']:
        path = f"processed_features/{folder}"
        if not os.path.exists(path):
            print(f"Folder '{path}' not found.\n")
            continue
        
        # CORRECTED: Added /*/ to search inside Fight/ and NonFight/ subfolders
        all_files = glob.glob(f"{path}/*/*.npz")
        xd_files = glob.glob(f"{path}/*/xd_*.npz")
        
        total = len(all_files)
        xd_count = len(xd_files)
        rwf_count = total - xd_count
        
        print(f"[{folder.upper()} FOLDER]")
        print(f"Total Files : {total}")
        print(f"RWF-2000    : {rwf_count} (Silent CCTV)")
        print(f"XD-Violence : {xd_count} (Audio + Video)")
        print("-" * 30)

if __name__ == "__main__":
    check_dataset_split()