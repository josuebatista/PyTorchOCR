# Optical Character Recognition Using PyTorch

This is a simple project that demonstrates how to build an Optical Character Recognition (OCR) model using PyTorch.

## Task 1: Image Preprocessing

Firstly, we need to preprocess the images before feeding them into the model. The `torchvision.transforms` module contains several transforms that can be chained together using `transforms.Compose`. The transforms we used here include `Resize` to adjust the image size, `RandomHorizontalFlip` for data augmentation, `ToTensor` to convert the images into PyTorch tensors, and `Normalize` to normalize the pixel values.

```python
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## Task 2: Data Preparation

Secondly, the data is loaded and split into training, validation, and testing sets using the `torchvision.datasets.ImageFolder` and `torch.utils.data.DataLoader` classes.

```python
data = torchvision.datasets.ImageFolder(root='./EnglishFnt/English/Fnt', transform=transform)
train_loader, val_loader, test_loader = load_split(data, batch_size=36, test_split=0.3)
```

## Task 3: Building a CNN and One-hot-encoding

Then, we define a simple Convolutional Neural Network (CNN) architecture for the OCR model. The model consists of two convolutional layers and a fully connected layer. Since there are 62 classes (10 digits, 26 lowercase letters, and 26 uppercase letters), the output size of the fully connected layer is set to 62.

```python
class OCRNet(nn.Module):
    ...
```

## Task 4: Set the Optimizer and Loss Functions

The optimizer and loss function are set up as follows. Stochastic Gradient Descent (SGD) is used as the optimizer, and Cross Entropy Loss is used as the loss function, which is suitable for multi-class classification problems.

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

After training, the model is validated using the validation set.

```python
validate(model, val_loader, criterion)
```

## Task 7: Testing the Model

Finally, the model is tested using the test set.

```python
test(model, test_loader, criterion)
```

## Task 8: Saving and Loading the Model

We can save the trained model for later use or further training.

```python
torch.save(model.state_dict(), '/path/to/save/model.pth')
```

To load the model, we first need to initialize an instance of the model and then load the parameters.

```python
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
- matplotlib

For detailed implementation, please refer to the [source code](https://github.com/username/repo).
