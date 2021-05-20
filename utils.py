import matplotlib.pyplot as plt
import json
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from PIL import Image



def load_data(where="./flowers"):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=65, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=65, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=65, shuffle=True)

    return trainloader, validloader, testloader, train_data, valid_data, test_data


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def nn_setup(input=1664, output=102, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Densenet121 as Pre-trained model
    model = models.densenet169(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(1664, 832),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(832, 416),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(416, 208),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(208, 102),
                                     nn.LogSoftmax(dim=1))

    # Defining the model
    model.fc = model.classifier
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.fc.parameters(), lr)

    model.to(device)

    return model, optimizer, criterion


def train_network(model, criterion, optimizer, trainloader, validloader, epochs=5, print_every=5):
    steps = 0
    running_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate Accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(validloader):.3f}")
                running_loss = 0
                model.train()


def accuracy_test(testloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            result = model(inputs)
            _, predicted = torch.max(result.data, 1)
            test_loss += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    accuracy_image = accuracy / test_loss * 100
    print(f'The accuracy is:{round(accuracy_image, 2)} %')


def checkpoint(train_data, model, optimizer, epochs=10):
    model.class_to_idx = train_data.class_to_idx

    torch.save({'input_size': 1664,
                'output_size': 102,
                'architecture': 'densenet169',
                'classifier': model.classifier,
                'learning_rate': 0.001,
                'epochs': epochs,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, 'checkpoint.pth')


def load_checkpoint():
    checkpoint = torch.load('checkpoint.pth')
    input = checkpoint['input_size']
    output = checkpoint['output_size']
    lr = checkpoint['learning_rate']
    model = nn_setup(input, output, lr)
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    img_pil = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = img_transform(img_pil)

    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
    model.eval()

    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to("cpu")

    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(topk)
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    print(top_probs)
    print(top_labels)

    return top_probs, top_labels, top_flowers






