import random
from net import Model, Input
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import cProfile as cp
import matplotlib.pyplot as plt



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images = train_images / 255
test_images = test_images / 255

train_images = train_images.reshape((-1, 28 * 28))
test_images = test_images.reshape((-1, 28 * 28))

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

training_data = np.array([Input(img, lbl) for img, lbl in zip(train_images, train_labels)])
testing_data = np.array([Input(img, lbl) for img, lbl in zip(test_images, test_labels)])


NN = Model([28 ** 2, 128, 10])

# epochs = 3
# batch_size = 64
# NN.train(training_data, batch_size=batch_size, epochs=epochs, learning_rate=0.1, saving=True, filename="saves/mnist")



def testInterface(modelFileName):
    NN.load(modelFileName)
    NN.evaluate(testing_data)
    while 1:
        img, lbl = random.choice(list(zip(test_images, test_labels)))
        x = NN.process(img)
        print(f'____\nActual: {list(lbl).index(max(lbl))}\nModel: {list(x).index(max(x))}')
        data = np.zeros((28, 28))
        for i in range(28):
            for j in range(28):
                data[i][j] = img[28 * i + j]
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='gray', vmax=1, vmin=0, interpolation='none')
        plt.show()

testInterface("saves/mnist_v3.pickle")
