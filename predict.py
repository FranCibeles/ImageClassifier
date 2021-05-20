import argparse
import json
import utils
import numpy as np

ap = argparse.ArgumentParser(description='prediction-file')


ap.add_argument('--input_img', default='./flowers/test/13/image_05769.jpg', nargs='*', action="store", type = str)
ap.add_argument('--checkpoint', default='.checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

pa = ap.parse_args()
image_path = ''.join(pa.input_img)
number_of_outputs = pa.top_k
input_img = ''.join(pa.input_img)
path = ''.join(pa.checkpoint)

trainloader, validloader, testloader, train_data, valid_data, test_data = utils.load_data()

model = utils.load_checkpoint()

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

probabilities = utils.predict(image_path, model, number_of_outputs)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].cpu())]
probability = np.array(probabilities[0][0].cpu())

i=0

print("#### Printing results ####")

while i < number_of_outputs:
    print("{} with probability {}".format(labels[i], probability[i]))
    i += 1

print("#### Prediction is finished ####")



