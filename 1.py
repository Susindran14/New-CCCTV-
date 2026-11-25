import os
from pathlib import Path

archive_dir = Path("archive")

print(f"Inspecting contents of: {archive_dir.resolve()}\n")

# Show subdirectories
subdirs = [p.name for p in archive_dir.iterdir() if p.is_dir()]
print("Subfolders found:", subdirs if subdirs else "None (flat directory)")

# Show sample files
files = list(archive_dir.rglob("*.*"))
print(f"Total files found: {len(files)}")
print("\nFirst 10 sample files:")
for f in files[:10]:
    print(" -", f.relative_to(archive_dir))