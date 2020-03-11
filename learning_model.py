from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


# Urban Sound Dataset 특징들 불러오기
sound_data = np.load('urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']
group_data = sound_data['groups']

X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

# 텐서플로우 모델 생성
training_epochs = 100
n_dim = 193
n_classes = 10

# 특징 배열 1차원 -> 3차원 변환
X_train = tf.reshape(X_train, [-1, 1, n_dim, 1])
X_val = tf.reshape(X_val, [-1, 1, n_dim, 1])
X_test = tf.reshape(X_test, [-1, 1, n_dim, 1])

# CNN 모델 구축
model = tf.keras.Sequential([
    # 레이어 1
    tf.keras.layers.Conv2D(50, (1, 5), activation='relu', padding='same', input_shape=(1, n_dim, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2),

    # 레이어 2
    tf.keras.layers.Conv2D(100, (1, 5), activation='tanh', padding='same', input_shape=(1, 93, 50)),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2),

    # 레이어 3
    tf.keras.layers.Conv2D(100, (1, 5), activation='relu', padding='same', input_shape=(1, 46, 100)),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2),
    tf.keras.layers.Flatten(),

    # 출력 레이어
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=training_epochs, validation_data=(X_val, y_val))

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

# 학습된 모델 저장
model.save('sound_model.h5')