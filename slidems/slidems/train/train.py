import os
from PIL import Image
from pathlib import Path
import torch

#import fastbook
from fastcore.all import *
from fastai.vision.all import *
from fastai.vision.widgets import *
#from jmd_imagescraper.imagecleaner import *
#import matplotlib as plt


def save_confusion_matrix(confusion_matrix, save_file, vocab,
                          normalize:bool=False, # Whether to normalize occurrences
                          title:str='Confusion matrix', # Title of plot
                          cmap:str="Blues", # Colormap from matplotlib
                          norm_dec:int=2, # Decimal places for normalized occurrences
                          plot_txt:bool=True, # Display occurrence in matrix
                          **kwargs):
    "Plot the confusion matrix, with `title` and using `cmap`."
    # This function is mainly copied from the sklearn docs
    cm = confusion_matrix
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(vocab))
    plt.xticks(tick_marks, vocab, rotation=90)
    plt.yticks(tick_marks, vocab, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white"
                     if cm[i, j] > thresh else "black")

    ax = fig.gca()
    ax.set_ylim(len(vocab)-.5,-.5)

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)
    plt.savefig(save_file)


working_dir = Path(__file__).parent.parent.parent.absolute()
trsPath = Path(f"{working_dir}/imgdb/train_set")

trainFiles = get_image_files(trsPath)
print (f"Found - {len(trainFiles)} images in train set")
failed = verify_images(trainFiles)
if len(failed):
    failed.map(Path.unlink)
    print(f"{len(failed)} images are not good")
    raise(f"{len(failed)} images are not good")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Will work on device = {device}")


reSize=512
slices = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   get_y=parent_label,
                   item_tfms=Resize(reSize))
dls = slices.dataloaders(trsPath)
dls.show_batch(max_n=4, nrows=1, unique=False)

# Enalrge train set with Random resize crop
slices = slices.new(item_tfms=RandomResizedCrop(256))
dls = slices.dataloaders(trsPath)

#dls.train.show_batch(max_n=4, nrows=1, unique=False)
#dls.train.show_batch(max_n=4, nrows=1, unique=True)


# Train
epoc=10
network = "resnet50"
pklFile = f"{working_dir}/model/resize{reSize}_RandomResizedCrop256_{network}_epoc{epoc}_2.pkl"
if os.path.isfile(pklFile):
    print(f"Loading model from - {pklFile}")
    learn = load_learner(pklFile)
else:
    learn = vision_learner(dls, resnet50, metrics=error_rate)
    learn.to(device)
    learn.fine_tune(epoc)
    os.makedirs(f"{working_dir}/model", exist_ok=True)
    learn.export(pklFile)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    vocab = learn.dls[1].new(shuffle=False, drop_last=False)
    save_confusion_matrix(interp.confusion_matrix(), f"{pklFile}.png", vocab)

    interp.plot_top_losses(6, nrows=2, figsize=(15,10))

testGoodImage = Path(f"{working_dir}/imgdb/test_set/GoodFocus/GoodFocus_ANONJSBHSI1F2_1_1_level_17_size_840/26_10.jpeg")
img = PILImage.create(testGoodImage)
className,classNum,probs = learn.predict(PILImage.create(img))
print(f"This is a: {className}. with probability {probs[classNum]:.4f}")

img.to_thumb(500,500).show()

print("Done")
