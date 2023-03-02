import io
import matplotlib.pyplot as plt
from PIL import Image

def save_lzw_tiff_mpl(fpath_out, **kwargs):
    
    tif1 = io.BytesIO()
    plt.savefig(tif1, **kwargs)
    
    # Load this image into PIL and save
    tif2 = Image.open(tif1)
    tif2.save(fpath_out, compression='tiff_lzw')
    tif1.close()
    tif2.close()