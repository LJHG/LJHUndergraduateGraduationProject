from typing import Optional, Sequence, Union

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory

# bi-real net在ImageNet数据集上的实现

@factory
class BiRealNetFactory(ModelFactory):

    filters: int = Field(64)

    input_quantizer = "approx_sign"
    kernel_quantizer = "magnitude_aware_sign"
    kernel_constraint = "weight_clip"

    kernel_initializer: Union[tf.keras.initializers.Initializer, str] = Field(
        "glorot_normal"
    )

    def residual_block(
        self, x, double_filters: bool = False, filters: Optional[int] = None
    ) -> tf.Tensor:
        assert not (double_filters and filters)

        # Compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x

        if in_filters != out_filters:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = tf.keras.layers.Conv2D(
                out_filters,
                (1, 1),
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            (3, 3),
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

        return tf.keras.layers.add([x, shortcut])

    def build(self) -> tf.keras.models.Model:
        # Layer 1
        out = tf.keras.layers.Conv2D(
            self.filters,
            (7, 7),
            strides=2,
            kernel_initializer=self.kernel_initializer,
            padding="same",
            use_bias=False,
        )(self.image_input)
        out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
        out = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding="same")(out)

        # Layer 2
        out = self.residual_block(out, filters=self.filters)

        # Layer 3 - 5
        for _ in range(3):
            out = self.residual_block(out)

        # Layer 6 - 17
        for _ in range(3):
            out = self.residual_block(out, double_filters=True)
            for _ in range(3):
                out = self.residual_block(out)

        # Layer 18
        if self.include_top:
            out = utils.global_pool(out)
            out = tf.keras.layers.Dense(self.num_classes)(out)
            out = tf.keras.layers.Activation("softmax", dtype="float32")(out)

        model = tf.keras.Model(inputs=self.image_input, outputs=out, name="birealnet18")

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="birealnet",
                    version="v0.3.0",
                    file="birealnet_weights.h5",
                    file_hash="6e6efac1584fcd60dd024198c87f42eb53b5ec719a5ca1f527e1fe7e8b997117",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="birealnet",
                    version="v0.3.0",
                    file="birealnet_weights_notop.h5",
                    file_hash="5148b61c0c2a1094bdef811f68bf4957d5ba5f83ad26437b7a4a6855441ab46b",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


def BiRealNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:

    return BiRealNetFactory(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
    ).build()


def getBiRealNet():
    model = BiRealNet(
        input_shape=(32,32,3),
        input_tensor=None,
        weights=None,
        include_top=True,
        num_classes=100
    )
    return model

if __name__ == "__main__":
    model = getBiRealNet()
    lq.models.summary(model)
    model.save("bireal_1000.h5")
    