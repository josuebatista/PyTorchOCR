# Optical Character Recognition Using PyTorch

Welcome! This project is all about my journey in implementing an Optical Character Recognition (OCR) model using PyTorch. Together, we'll see how I trained a Convolutional Neural Network (CNN) to recognize individual characters in natural images.

[Colab Notebook](https://github.com/josuebatista/PyTorchOCR/blob/main/PyTorch_OCR.ipynb)

# Introduction

Glad you're here! You're looking at a project I've been working on that uses a technology called Optical Character Recognition, or OCR for short. OCR is like teaching a computer to read images like a human, except that it does it faster and more accurately.

I trained a machine learning model to recognize characters from images. Think of it like teaching a computer to read from scratch. But instead of children's books, we're using a collection of character images.

Now, you might be wondering, how can this be useful in business? Let's take a quick look. Picture this, being able to automatically read invoices, receipts, or other documents. Sounds cool, right? It can drastically cut down on manual data entry tasks, which means reducing costs and improving efficiency. In sectors like banking, healthcare, and many more, this can revolutionize many processes.

And the beauty of this project? It's just the start. Once we have a solid OCR system, we can do more - understanding the context of the text, recognizing handwriting, or even expanding it to recognize characters in other languages.

The cherry on top? OCR has fascinating links to generative AI, where AI models create new, original content. Pairing OCR with generative AI could allow for generating text and even creating images of that text in various styles and fonts. Picture this in design or advertising, where you could automatically generate unique, personalized ads or design elements.

In a nutshell, this project is your gateway into a future where AI interacts with text just as we do - reading, understanding, and even creating it. The possibilities are immense, and I can't wait for you to be a part of this exciting journey!

## Dataset

I used the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/), which provides images of individual characters. It's a great collection that has variations in scale and maintains the original resolution of the characters.

## Here's a brief of the steps I took:

### Task 1: Image Preprocessing
I preprocessed the images before feeding them into the model, making them ready and easier for the model to work with.

### Task 2: Data Loading and Preparation
Next, I loaded and prepared the dataset by splitting it into training, validation, and testing sets.

### Task 3: Building a CNN and One-hot-encoding
Then, I defined a simple Convolutional Neural Network (CNN) architecture for the OCR model.

### Task 4: Set the Optimizer and Loss Functions
I used Stochastic Gradient Descent (SGD) as the optimizer, and Cross Entropy Loss as the loss function.

### Task 5: Training the Model
I trained the model using the training set, making sure to monitor the training process.

### Task 6: Validate the Model
After training, I validated the model using the validation set to ensure it was learning well.

### Task 7: Testing the Model
Then, I tested the model using the test set to see how it would perform on unseen data.

### Task 8: Saving and Loading the Model
I saved the trained model for later use or further training. You can even load the model later when you need it.

### Task 9: Making Predictions
Finally, I used the trained model to predict the class of a single image. So exciting!

I did all these using a set of amazing tools like Python, PyTorch, torchvision, PIL, numpy, and matplotlib. These are wonderful open-source tools that are widely used in the field of machine learning and data science.

As you can see, this project is a hands-on, practical application of OCR technology and is a great starting point for any individual or business looking to automate processes involving text in images. 

So if you've ever been curious about how machines read text or if you're looking to streamline your document handling processes, this project could be a valuable resource. And who knows? It might even inspire you to start your own project!

Join me on this exciting journey as we step into a future where AI not only reads and understands text but can also generate new content in creative and innovative ways. This isn't just about recognizing text in images - it's about making our interactions with technology more seamless, efficient, and personalized. And you're invited to be a part of it!

## Requirements

- Python
- PyTorch
- torchvision
- PIL
- numpy
- mathplotlab
