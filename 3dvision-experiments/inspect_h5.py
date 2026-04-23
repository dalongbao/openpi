"""Print h5 structure of the first file in a directory."""
import sys, h5py, pathlib

d = sys.argv[1] if len(sys.argv) > 1 else "/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery"
f = h5py.File(sorted(pathlib.Path(d).glob("*.h5"))[0], "r")

def show(name, obj):
    if hasattr(obj, "shape"):
        print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")

print(f"File: {f.filename}")
f.visititems(show)
f.close()
