# Optical Character Recognition Using PyTorch

This project provides an implementation of an Optical Character Recognition (OCR) model using PyTorch. We train a Convolutional Neural Network (CNN) to recognize individual characters in natural images.

## Dataset

We make use of the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/), which provides images of individual characters, including variations in scale, and maintaining the original resolution of the characters as they appear in the original images. 

Specifically, we are using the `EnglishFnt.tgz` file, which contains characters from computer fonts with 4 variations (combinations of italic, bold, and normal). This dataset has 62 classes, consisting of digits 0-9, uppercase letters A-Z, and lowercase letters a-z.

## Task 1: Image Preprocessing

The images are preprocessed before feeding them into the model. The preprocessing steps include resizing the images, randomly flipping the images horizontally for data augmentation, converting the images into PyTorch tensors, and normalizing the pixel values.

```python
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## Task 2: Data Loading and Preparation

We load the dataset and prepare it by splitting into training, validation, and testing sets. 

```python
data = torchvision.datasets.ImageFolder(root='./EnglishFnt/English/Fnt', transform=transform)
train_loader, val_loader, test_loader = load_split(data, batch_size=36, test_split=0.3)
```

## Task 3: Building a CNN and One-hot-encoding

We then define a simple Convolutional Neural Network (CNN) architecture for the OCR model. The output size of the fully connected layer is set to 62, corresponding to the 62 classes in our dataset.

```python
class OCRNet(nn.Module):
    ...
```

## Task 4: Set the Optimizer and Loss Functions

Stochastic Gradient Descent (SGD) is used as the optimizer, and Cross Entropy Loss is used as the loss function.

```python
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
```

## Task 5: Training the Model

The model is trained using the training set. The loss for each epoch is printed to monitor the training process.

```python
train(model, train_loader, optimizer, criterion, num_epochs=10)
```

## Task 6: Validate the Model

After training, we validate the model using the validation set.

```python
validate(model, val_loader, criterion)
```

## Task 7: Testing the Model

Finally, the model is tested using the test set.

```python
test(model, test_loader, criterion)
```

## Task 8: Saving and Loading the Model

We can save the trained model for later use or further training. To load the model, we first initialize an instance of the model and then load the parameters.

```python
torch.save(model.state_dict(), '/path/to/save/model.pth')

model = OCRNet(num_features)
model.load_state_dict(torch.load('/path/to/saved/model.pth'))
model.eval()  # Set the model to evaluation mode
```

## Task 9: Making Predictions

The trained model can be used to predict the class of a single image.

```python
predict(model, '/path/to/an/image.png', transform)
```

## Requirements

- Python
- PyTorch
- torchvision
- PIL
- numpy
- mathplotlab
