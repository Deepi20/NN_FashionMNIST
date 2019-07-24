from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import np_utils
num_classes = 10
batch_size = 64
epochs = 40
input_shape = (28,28,1)
seed = 5
np.random.seed(seed)
y_train = to_categorical(y_train, num_classes)
