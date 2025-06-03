from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import os
import re
import datetime


# 1. 获取图片路径和标签
def get_image_paths_and_labels(image_dir):
    image_paths = []
    labels = []

    for fname in os.listdir(image_dir):
        if fname.endswith('.jpg'):
            match = re.match(r'training_\d+_(\d+)\.jpg', fname)
            if match:
                label = int(match.group(1))  # 提取YY作为标签
                image_paths.append(os.path.join(image_dir, fname))
                labels.append(label)
    return image_paths, labels


# 2. 图像加载与预处理函数
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)  # 灰度图
    image = tf.image.resize(image, [28, 28])  # 统一大小
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    return image, tf.one_hot(label, depth=10)  # 返回 one-hot 标签


# 3. 创建 tf.data.Dataset
# 重新创建数据集函数，返回拆分好的 train 和 val
def create_train_val_datasets(image_dir, batch_size=32, val_split=0.1,):
    image_paths, labels = get_image_paths_and_labels(image_dir)

    # 转为 TensorFlow Dataset
    full_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    full_ds = full_ds.shuffle(buffer_size=len(image_paths), seed=42)

    # 计算验证集大小
    val_size = int(len(image_paths) * val_split)

    # 加载和预处理映射
    full_ds = full_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # 拆分验证集和训练集
    val_ds = full_ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = full_ds.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

train_ds, val_ds = create_train_val_datasets('mnist_jpg',batch_size=64)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 使用tensorboard来进行可视化处理，命令行进入到当前logs的文件目录，运行tensorboard --logdir=logs/fit，打开网址即可
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_ds, epochs=5, validation_data=val_ds,callbacks=[tensorboard_callback])

