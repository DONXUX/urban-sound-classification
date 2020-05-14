from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Urban Sound Dataset 특징들 불러오기
sound_data = np.load('urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']
group_data = sound_data['groups']

X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

# 텐서플로우 모델 생성
training_epochs = 5
n_dim = 193
n_classes = 9

# 특징 배열 1차원 -> 3차원 변환
X_train = tf.reshape(X_train, [-1, 1, n_dim, 1])
X_val = tf.reshape(X_val, [-1, 1, n_dim, 1])
X_test = tf.reshape(X_test, [-1, 1, n_dim, 1])

# CNN 모델 구축
model = tf.keras.Sequential([
    # 레이어 1
    tf.keras.layers.Conv2D(64, (1, 5), activation='relu', padding='same', input_shape=(1, n_dim, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2),

    # 레이어 2
    tf.keras.layers.Conv2D(128, (1, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2),

    # 레이어 3
    tf.keras.layers.Conv2D(128, (1, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=2),

    # 출력 정보 1차원 변환
    tf.keras.layers.Flatten(),

    # 출력 레이어
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=training_epochs, validation_data=(X_val, y_val))
print("모델 학습 완료")

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# 모델 평가 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

# 학습된 모델 저장
model.save('sound_model.h5')
print("모델 저장 완료")