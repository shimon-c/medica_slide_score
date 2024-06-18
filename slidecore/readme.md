slidecore algos:
how to train:
python slidescore\net\train1.py --yaml_path C:\Users\shimon.cohen\PycharmProjects\medica\medica\config\train1.yaml

The train parameters are in the yamlfile above. These parameters dictate the architecture of the net
as well as the augmentation.
You should add to your PYTHONPATH the current directory.

predict_imgs:
It can predict for several image at one time (batch).
python slidescore\predict_imgs\predict_imgs 

Creating an Ensemble:
python slidescore\net\ensemble.py --model_path "fullpath of a model"
it will pick up all the models in that directory and generate an Ensemble from it.
