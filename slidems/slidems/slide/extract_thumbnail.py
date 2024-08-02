'''
Created on 2024/04/11 21:35

@Author - Dudi Levi
'''

import os
import logging
from openslide import open_slide

log = logging.getLogger(__name__)



def GetSlideThumbnail(slide, size=(3000,1000)):
    """Return a PIL.Image containing an RGB thumbnail of the image.
       size:     the maximum size of the thumbnail."""
        
    assert os.path.isfile(slide), f"Slide file was not found at {slide}"
    slide_ = open_slide(slide)
    return slide_.get_thumbnail(size)

