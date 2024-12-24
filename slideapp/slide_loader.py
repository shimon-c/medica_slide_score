# pip install git+https://github.com/RTLucassen/slideloader
# https://github.com/RTLucassen/slideloader
import slideloader
#from slideloader import SlideLoader

# Initialize SlideLoader instance
loader = slideloader.SlideLoader()

# Define the path to your DICOM tiles and load the slide
path = '/home/shimon/Desktop/sectra-9.12.24/bad/ANON6V2ELJ1IK/ANON6V2ELJ1IK_1_1.dcm'
loader.load_slide(path)

# Get the dimensions of the slide
dimensions = loader.get_dimensions(magnification=5.0)

# Get the slide image
slide_image = loader.get_image(magnification=5.0)

# Save the slide image
slide_image.save('path/to/save/your/slide_image.png')