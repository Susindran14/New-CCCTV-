import os
import glob
import shutil
import random

def fix_dataset_split():
    print("Moving XD-Violence (Audio) files to the Training folder...\n")
    val_dir = "processed_features/val"
    train_dir = "processed_features/train"
    
    # Get the class subfolders (e.g., Fight, NonFight)
    if not os.path.exists(val_dir):
        print(f"Error: {val_dir} not found.")
        return
        
    class_folders = [f.name for f in os.scandir(val_dir) if f.is_dir()]
    total_moved = 0
    
    for cls in class_folders:
        xd_files = glob.glob(f"{val_dir}/{cls}/xd_*.npz")
        
        if not xd_files:
            continue
            
        # Shuffle files to ensure a random 80/20 mix
        random.seed(42)
        random.shuffle(xd_files)
        
        # Calculate 80% to move to the training set
        split_idx = int(len(xd_files) * 0.8)
        files_to_move = xd_files[:split_idx]
        
        target_cls_dir = f"{train_dir}/{cls}"
        os.makedirs(target_cls_dir, exist_ok=True)
        
        for file_path in files_to_move:
            file_name = os.path.basename(file_path)
            shutil.move(file_path, f"{target_cls_dir}/{file_name}")
            total_moved += 1
            
        print(f"✅ Moved {len(files_to_move)} files for class '{cls}'.")
        
    print(f"\n🎉 Successfully moved {total_moved} total XD-Violence files into the training set!")
    print("Run 'python check_data.py' again to verify, then you can start training!")

if __name__ == "__main__":
    fix_dataset_split()