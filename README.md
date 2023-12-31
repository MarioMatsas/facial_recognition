# facial_recognition
A program that allows users to create and train their own facial identification Model, while also presenting an example of how one could implement said Model in an application.
## Idea
In this Siamese Neural Network the idea is to use 2 images -> calculate how similar they are -> output **True** if they are the very similar or **False** if they are not, as shown below.
![false_output](https://github.com/MarioMatsas/facial_recognition/assets/114489498/881b5dc4-8f6d-42e4-8f89-88ff0d9681d2)
![true_output](https://github.com/MarioMatsas/facial_recognition/assets/114489498/f161561f-402c-49dc-8e59-61974fc1637c)

Of course this is a very simple way of describing the process, however it helps paint the main idea as well as the thought process we will follow. For a more refined explanation and better understanding I recommend you take a look at the research paper titled **"Siamese Neural Networks for One-shot Image Recognition"** in the following link: [one_shot](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), considering that this entire project is based on it.
## Things to Note
No Model will be provided so in order to use the program of `example.py` you need to make sure that you have already trained and saved a Model by running the `train_siamese.ipynb` file.


