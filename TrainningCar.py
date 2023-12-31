import matplotlib.pyplot as plt

print('Setting UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_selection import train_test_split

# Step1
path = 'data_car'
data = importDatainfor(path)

# Step2
data = balanceData(data, display=False)

# Step3
imagesPath, steerings = loadData(path, data)
# print(imagesPath[0],steerings[0])

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Số ảnh đem train :', len(xTrain))
print('Số ảnh test :', len(xVal))

# Step4


# Step5

# Step6
model = createModel()
model.summary()

# Step7
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10,
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

# Step8
model.save('model.h5')
print('Model đã được lưu')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
