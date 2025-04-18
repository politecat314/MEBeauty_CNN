import tensorflow as tf
from tensorflow.keras import layers
from keras_vggface.vggface import VGGFace

def ResNet50(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    resnet._name = "resnet50"

    base_cnn = tf.keras.models.Model(resnet.input, resnet.layers[-34].output)
    base_cnn._name = "resnet50"

    train_cnn = tf.keras.models.Model(resnet.layers[-33].input, resnet.output)
    train_cnn._name = "resnet50_stage5"

    return base_cnn, train_cnn, resnet

# def VGG16(input_shape=(224, 224, 3)):
#     vgg = VGGFace(
#         model="vgg16",
#         include_top=False,
#         input_shape=input_shape,
#         weights="vggface"
#     )
#     vgg._name = "vgg16"

#     base_cnn = tf.keras.models.Model(vgg.input, vgg.get_layer("pool4").output)
#     base_cnn._name = "vgg16"

#     train_cnn = tf.keras.models.Model(vgg.get_layer("conv5_1").input, vgg.output)
#     train_cnn._name = "vgg16_stage5"

#     return base_cnn, train_cnn, vgg


def VGG16(input_shape=(224, 224, 3)):
    vgg = VGGFace(
        model="vgg16",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    vgg._name = "vgg16"

    base_cnn = tf.keras.models.Model(vgg.input, vgg.get_layer("pool4").output)
    base_cnn._name = "vgg16"

    x_input = tf.keras.Input(shape=(14, 14, 512))  # Shape should match pool4 output
    x = vgg.get_layer("conv5_1")(x_input)
    x = vgg.get_layer("conv5_2")(x)
    x = vgg.get_layer("conv5_3")(x)
    x = vgg.get_layer("pool5")(x)
    train_cnn = tf.keras.models.Model(inputs=x_input, outputs=x, name="vgg16_stage5")
    train_cnn._name = "vgg16_stage5"

    return base_cnn, train_cnn, vgg



def SENet50(input_shape=(224, 224, 3)):
    senet = VGGFace(
        model="senet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    senet._name = "senet50"

    base_cnn = tf.keras.models.Model(senet.input, senet.layers[-55].output)
    base_cnn._name = "senet50"

    train_cnn = tf.keras.models.Model(senet.layers[-54].input, senet.output)
    train_cnn._name = "senet50_stage5"

    return base_cnn, train_cnn, senet