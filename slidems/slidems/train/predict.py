import os
from fastcore.all import *
from fastai.vision.all import *
from PIL import Image



learn = load_learner("/home/dudi/privet/med/deeplearning/model/resize_192_squish_resnet18_epoc5.pkl")


#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix()




tssPath = Path("/home/dudi/privet/med/deeplearning/imgdb/train_set/GoodFocus/ANONFACHSI1RE_8_1_5_38.jpeg")
is_good,_,probs = learn.predict(PILImage.create(tssPath))
print(f"This is a: {is_good}. with probability {probs[0]:.4f}")

tssPath = Path("/home/dudi/privet/med/deeplearning/imgdb/test_set/GoodFocus/GoodFocus_ANONJSBHSI1F2_1_1_level_17_size_840/30_37.jpeg")
is_good,_,probs = learn.predict(PILImage.create(tssPath))
print(f"This is a: {is_good}. with probability {probs[0]:.4f}")


tssPath = Path("/home/dudi/privet/med/deeplearning/imgdb/test_set/BadFocus/BadFocus_ANONFACHSI1RE_4_1_level_17_size_840/18_31.jpeg")
is_good,_,probs = learn.predict(PILImage.create(tssPath))
print(f"This is a: {is_good}. with probability {probs[0]:.4f}")

