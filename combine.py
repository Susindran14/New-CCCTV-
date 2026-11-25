import shutil
from pathlib import Path

def merge_all_archive_files():
    archive_dir = Path("archive")
    target_dir = Path("Dataset_Combined")
    
    (target_dir / "Fight").mkdir(parents=True, exist_ok=True)
    (target_dir / "NonFight").mkdir(parents=True, exist_ok=True)

    copied_count = 0
    valid_extensions = {".mp4", ".avi", ".mkv"}

    print("Scanning for all video files...")
    
    # Check every file in the archive
    for video_file in archive_dir.rglob("*"):
        if video_file.suffix.lower() in valid_extensions:
            # Get the folder name (e.g., 'A', 'V', 'Fight')
            folder_class = video_file.parent.name.upper()
            
            # If the folder is 'A' or has 'Normal' in the name, it's NonFight.
            # Otherwise, assume it's a Fight/Violence video.
            if folder_class == "A" or "NON" in folder_class or "NORMAL" in folder_class:
                target_subfolder = "NonFight"
            else:
                target_subfolder = "Fight"
            
            # Create a unique filename
            split_name = video_file.parent.parent.name
            new_filename = f"xd_{split_name}_{folder_class}_{video_file.name}"
            target_path = target_dir / target_subfolder / new_filename
            
            # Copy if it doesn't already exist
            if not target_path.exists():
                shutil.copy(video_file, target_path)
                copied_count += 1

    print(f"✅ Successfully copied {copied_count} additional files into {target_dir}")

if __name__ == "__main__":
    merge_all_archive_files()