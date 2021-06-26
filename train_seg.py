import segmentation_models as sm
from data import load_data, tf_dataset

BACKBONE = 'densenet121'
preprocess_input = sm.get_preprocessing(BACKBONE)
path="D:\Kvasir-SEG"
# load your data
x_train, y_train, x_val, y_val = load_data(path)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(
   x=x_train,
   y=y_train,
   batch_size=16,
   epochs=100,
   validation_data=(x_val, y_val),
)
model.save("ver1.h5")