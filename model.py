import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_directory = "PokemonData"

datagen = ImageDataGenerator(
rescale=1./255,
horizontal_flip=True,
vertical_flip=True,
rotation_range=20,
zoom_range=0.2,
fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
local_directory,
target_size=(128, 128),
batch_size=32,
class_mode="sparse"
)

input_shape = (128, 128, 3)
n_classes = 150


dense_model = tf.keras.applications.MobileNetV2(
include_top=False,
weights='imagenet',
input_shape=input_shape
)


for layer in dense_model.layers[:120]:
    layer.trainable = False

for layer in dense_model.layers[120:]:
    layer.trainable = True



# Build the model
model = tf.keras.models.Sequential()
model.add(dense_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(150, activation="softmax"))
#model.summary()

# Compile the model
model.compile(
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
metrics=['accuracy']
)

# Train the model
history = model.fit(
train_generator,
steps_per_epoch=225,
verbose=1,
epochs=20,

)
tf.keras.models.save_model(model,"myMLmodel_mobilenet.keras")
# After training the model...


