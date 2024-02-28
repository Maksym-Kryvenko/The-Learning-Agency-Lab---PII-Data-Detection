from main import ModelConfiguration
import keras
from keras import ops
import keras_nlp
import math
import plotly.express as px
import numpy as np

# LOSS AND METRICS
class CrossEntropy(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, ignore_class=-100, reduction=None, **args):
        super().__init__(reduction=reduction, **args)
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1, ModelConfiguration.num_labels])
        loss = super().call(y_true, y_pred)
        if self.ignore_class is not None:
            valid_mask = ops.not_equal(
                y_true, ops.cast(self.ignore_class, y_pred.dtype)
            )
            loss = ops.where(valid_mask, loss, 0.0)
            loss = ops.sum(loss)
            loss /= ops.maximum(ops.sum(ops.cast(valid_mask, loss.dtype)), 1)
        else:
            loss = ops.mean(loss)
        return loss
    
class FBetaScore(keras.metrics.FBetaScore):
    def __init__(self, ignore_classes=[-100, 12], average="micro", beta=5.0,
                 name="f5_score", **args):
        super().__init__(beta=beta, average=average, name=name, **args)
        self.ignore_classes = ignore_classes or []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
        
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1, ModelConfiguration.num_labels])
            
        valid_mask = ops.ones_like(y_true, dtype=self.dtype)
        if self.ignore_classes:
            for ignore_class in self.ignore_classes:
                valid_mask &= ops.not_equal(y_true, ops.cast(ignore_class, y_pred.dtype))
        valid_mask = ops.expand_dims(valid_mask, axis=-1)
        
        y_true = ops.one_hot(y_true, ModelConfiguration.num_labels)
        
        if not self._built:
            self._build(y_true.shape, y_pred.shape)

        threshold = ops.max(y_pred, axis=-1, keepdims=True)
        y_pred = ops.logical_and(
            y_pred >= threshold, ops.abs(y_pred) > 1e-9
        )

        y_pred = ops.cast(y_pred, dtype=self.dtype)
        y_true = ops.cast(y_true, dtype=self.dtype)
        
        tp = ops.sum(y_pred * y_true * valid_mask, self.axis)
        fp = ops.sum(y_pred * (1 - y_true) * valid_mask, self.axis)
        fn = ops.sum((1 - y_pred) * y_true * valid_mask, self.axis)
            
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

# LEARNING RATE SETUP
def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 6e-6, 2.5e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        fig = px.line(x=np.arange(epochs),
                      y=[lrfn(epoch) for epoch in np.arange(epochs)], 
                      title='LR Scheduler',
                      markers=True,
                      labels={'x': 'epoch', 'y': 'lr'})
        fig.update_layout(
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'
            )
        )
        fig.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

# Your model architecture code here
def create_model():
    # ...
    # BUILDING THE MODEL
    print("TRAINING: Creating the model ...")
    backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
        ModelConfiguration.preset,
    )
    out = backbone.output
    out = keras.layers.Dense(ModelConfiguration.num_labels, name="logits")(out)
    out = keras.layers.Activation("softmax", dtype="float32", name="prediction")(out)
    model = keras.models.Model(backbone.input, out)

    # Compile model for optimizer, loss and metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        loss=CrossEntropy(),
        metrics=[FBetaScore()],
    )


# Your training code here
def train_model(processed_train_data, InputData):
    # ...
    lr_cb = get_lr_callback(ModelConfiguration.train_batch_size, mode=ModelConfiguration.lr_mode, plot=True)
    model = create_model()

    # TRAINING
    if ModelConfiguration.train:
        print("TRAINING: Training the model ...")
        train_ds, valid_ds = processed_train_data
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=ModelConfiguration.epochs,
            callbacks=[lr_cb],
            verbose=1,
        )
        model.save_weights("model.weights.h5")
        model.evaluate(valid_ds, return_dict=True, verbose=0)

    else:
        print("TRAINING: Loading pre-trained model ...")
        model.load_weights(InputData.trained_model)
    
    print("TRAINING: Model is ready to use!")
    return model


