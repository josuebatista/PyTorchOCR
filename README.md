# Optical Character Recognition Using PyTorch

This project provides an implementation of an Optical Character Recognition (OCR) model using PyTorch. We train a Convolutional Neural Network (CNN) to recognize individual characters in natural images.

# Introduction

Welcome to our Optical Character Recognition (OCR) project using PyTorch! In simple terms, OCR is a technology that extracts text from images, such as scans of printed text, photos of documents, or even text overlaid on an image. Think of it as a digital eye that can read just like humans, only faster and more accurately.

In the context of our project, we're training a machine learning model to recognize individual characters from images, specifically, digits and letters in different variations of computer fonts. It's like teaching our computer to read from scratch, except instead of using children's books, we're using a curated dataset of character images. 

For businesses, the applications of OCR are immense. Imagine being able to automatically read invoices, receipts, or other documents. This could drastically cut down on manual data entry tasks, reducing costs and improving efficiency. For example, in the banking sector, OCR could be used to automate the process of reading checks or identification documents. In healthcare, it could help to digitize handwritten patient records or prescription information. 

Moreover, this project forms a foundation that could be extended in many ways. With a solid OCR system in place, we can work towards more complex tasks like understanding the context and semantics of the text or even handwriting recognition. We could also expand it to recognize characters in other languages, thus enabling businesses to engage with a global customer base. 

Finally, OCR also has exciting connections to generative AI, where AI models can create new, original content. With a trained OCR system, a generative AI model could not only generate text, but also create realistic images of that text in various styles and fonts. This could be particularly useful in design or advertising sectors, where companies could use it to automatically generate unique, personalized advertisements or design elements.

To sum it up, this project is a stepping stone towards a world where AI interacts with text just like we do - reading it, understanding it, and even creating it. The potential is truly vast, and we're excited to embark on this journey. Join us!

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
