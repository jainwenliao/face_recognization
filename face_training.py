from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

'''借鉴猫狗大战，设置的四层卷积、四层池化'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())#将三维展平
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation = 'sigmoid')) 

#把人脸识别看成分类问题，这里用二分类
model.compile(loss='binary_crossentropy',
optimizer = optimizers.RMSprop(lr = 1e-4),
metrics = ['acc'])

#图片处理
train_datagen = ImageDataGenerator(rescale = 1. / 255)
test_datagen = ImageDataGenerator(rescale = 1. / 255)
validation_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    'E:/face_recongnition/train',
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)
validation_generator = validation_datagen.flow_from_directory(
    'E:/face_recongnition/validation',
    target_size = (150,150),
    batch_size= 20,
    class_mode = 'binary'
)
test_generator = test_datagen.flow_from_directory(
    'E:/face_recongnition/test',
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = 10
)

model.save('face_training.h5')