import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
# from keras.layers import LeakyReLU
def conv_block(x, num_filters):
    s=1
    x = Conv2D(num_filters, (5, 5),strides =(s,s), padding="same",kernel_initializer = glorot_uniform(seed=0))(x)
    # x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    # x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = Conv2D(num_filters, (3, 3),strides =(s,s), padding="same",kernel_initializer = glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis =3)(x)
    x = Activation("relu")(x)

    # x = Conv2D(num_filters, (3, 3), padding="same")(x)
    # x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = Conv2D(num_filters, (3, 3),strides =(s,s), padding="same",kernel_initializer = glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis= 3)(x)
    x =Activation("relu")(x)

    x = Conv2D(num_filters, (1, 1),strides =(s,s), padding="same",kernel_initializer = glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis =3)(x)
    x = Activation("relu")(x)
    return x

def build_model():
    size = 256
    num_filters = [32,64, 128, 256]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


if __name__ == "__main__":
    model = build_model()