

OPENSLIDE_PATH = r'C:\Users\shimon.cohen\openslide\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin'

import os
def add_openslide():
    if hasattr(os, 'add_dll_directory'):
        # Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide

if __name__ == '__main__':
    add_openslide()
    import openslide
    print(f'after importing openslide')