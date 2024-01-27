#importing module
import numpy as np
import argparse, json
from utils import Util

argpa = argparse.ArgumentParser(description='Predict.py')
argpa.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
argpa.add_argument('checkpoint', default='checkpoint.pth')
argpa.add_argument('--gpu', action='store_true', default=False)
argpa.add_argument('input_img', default='flowers/test/1/image_06752.jpg')
argpa.add_argument('--top_k', dest="top_k", type=int,  action="store", default=5)


parsearg = argpa.parse_args()
category_names = parsearg.category_names
checkpoint = parsearg.checkpoint
hardware = "gpu" if parsearg.gpu else "cpu"
input_img = parsearg.input_img
number_of_outputs = parsearg.top_k

training_loader, testing_loader, validation_loader, _ = Util.load_data()

model = Util.load_checkpoint(checkpoint, hardware)

probabilities = Util.predict(input_img, model, number_of_outputs, hardware)

with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].cpu().numpy())]
probability = np.array(probabilities[0][0].cpu().numpy())

for i in range(0, number_of_outputs):
    print("{} with a probability of {}%".format(labels[i], probability[i]*100))
