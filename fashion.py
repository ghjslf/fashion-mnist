from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt

# Загружаем данные
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразование размерности изображений
x_train = x_train.reshape(60000, 784)

# Нормализация данных
x_train = x_train / 255 

# Преобразуем метки в категории
y_train = utils.to_categorical(y_train, 10)

# Создаем последовательную модель
model = Sequential()

# Добавляем уровни сети
model.add(Dense(1000, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Компилируем модель 
model.compile(loss='categorical_crossentropy', # Указываем фунукцию ошибки (здесь это не среднеквадратичная)
              optimizer='SGD', # Указываем метод оптимизации (гардиентный спуск)
              metrics=['accuracy']) # Указываем метрику качества (доля правильных ответов)

# Печатаем параметры модели
print(model.summary())

# Обучаем сеть
model.fit(x_train,
          y_train,
          batch_size=200, 
          epochs=100, 
          verbose=1) 

# Запускаем сеть на входных данных
predictions = model.predict(x_train)

# Выводим один из результатов распознования 
print(predictions[0])
print(np.argmax(predictions[0]))
print(np.argmax(y_train[0]))

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

n = 0
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

classes[np.argmax(predictions[n])]
