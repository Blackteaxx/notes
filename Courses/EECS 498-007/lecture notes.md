## Lecture 1: Introduction

Computer Vision is building AI systems that process, perceive and reason about the visual data(**all continuous signal**).

Learning is the concept of building AI systems that can learn from data and experience.

In some sense, CV is orthogonal to learning.

We can summarize that, **CV is teaching computers to see**, and **learning is teaching computers to learn**.

![20240921162416](https://cdn.jsdelivr.net/gh/Blackteaxx/Graph@master/img/20240921162416.png)

### History of CV and DL

#### CV

Hubel and Wiesel: Nobel Prize in Physiology or Medicine in 1981, finding that the following properties of the visual cortex:

![20240921163140](https://cdn.jsdelivr.net/gh/Blackteaxx/Graph@master/img/20240921163140.png)

- Simple cells: respond to edges and bars
- Complex cells: respond to light orientation and movement
- Hypercomplex cells: respond to movement with an end point

Larry Roberts: First computer vision system, 1963. It could recognize simple shapes and find that the edges are important.

David Marr: Vision, 1982. 4 stages of vision:

John Canny: Edge detection, 1986. **Canny edge detector** is still used today.

Viola and Jones: Face detection, 2001.

IMAGENET: 2012. AlexNet won the competition.

![20240921164925](https://cdn.jsdelivr.net/gh/Blackteaxx/Graph@master/img/20240921164925.png)

#### DL

Perceptron: firstly implemented in hardward!!

Neo-cognitron: 1980, a deep learning model similar to CNN.

Backpropagation: 1986, the algorithm to **train** deep learning models.

CNN: 1998, LeCun LeNet, applied to process handwritten digits.

### Nowaday

Learning: Data, Model, Algorithm.

CV still has a long way to go...

## Lecture 2: Image Classification

Challenges: **Viewpoint** variation, **intraclass** variation, scale variation, background clutter, illumination robustness, deformation, occlusion.

Image Classification can be a subproblem of object detection, image captioning, etc.

Machine Learning: Data-Driven Approach.

### Data

- MNIST: 60,000 training images, 10,000 test images. Drosohila of CV.
- CIFAR-10: 60,000 32x32 color images in 10 classes.
- CIFAR-100: 100 classes.
- ImageNet: 1.2 million images in 1000 classes. metric is Top-5 accuarcy rate.
- MIT places: 205 scene categories.
- Omniglot: 1623 handwritten characters from 50 alphabets. 20 examples per category. used for few-shot learning.

### First classifier: k-NN

- Distance / Dissimilarity: l1, l2

![20240923221655](https://cdn.jsdelivr.net/gh/Blackteaxx/Graph@master/img/20240923221655.png)

The k-NN Classifier is a lazy learner, which we can not afford slow prediction.$\mathcal{O}(N)$

The Assumption of the k-NN Classifier build the decision boundary for each class. But the boundary can be very noisy, easily affected by outliers. Instead we can enlarge the k to reduce the noise.

We can also change the distance metric to improve the performance. Like useing tf-idf to calculate the distance between two text.

Care about the ability of generlization on unseen data. The best way is to split the data into training set, validation set and test set.

After that, we should produce the cross-validation to tune the hyperparameters and test the ability of the model.

We also care about the universal approximation that introduce that as the number of training samples approaches infinity, the k-NN classifier approaches the Bayes error rate.

Due to the curse of dimensionality, the k-NN classifier is not suitable for high-dimensional data.(not enough data to cover the space)

Drawbacks:

- Slow in testing
- Distance metric on pixels is not informative

But K-NN using feature vectors computed by CNN can achieve good performance.(Devlin et al, "Exploring Nearest Neighbor Approaches for Image Captioning", CVPR 2015.)
