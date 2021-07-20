import os
import ssl

import PIL.ImageOps
import cv2
import numpy
import pandas
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

print("Getting Data...")
x = numpy.load('image.npz')['arr_0']
y = pandas.read_csv("https://raw.githubusercontent.com/whitehatjr/Alphabet-detection/master/project%20123/labels.csv")["labels"]
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
           "W", "X", "Y", "Z"]
n_classes = len(classes)
print("Done Getting Data!")

print("\nPreforming Train Test Split...")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=7500, test_size=2500)
print("Done Preforming Train Test Split!")

print("\nScaling X train and test...")
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0
print("Done Scaling X train and test!")

print("\nPreforming Logistic Regression...")
log_reg = LogisticRegression(solver='saga', multi_class='multinomial')
log_reg.fit(x_train_scaled, y_train)
print("Done Preforming Logistic Regression!")

print("\nCalculating Accuracy...")
y_pred = log_reg.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nGetting Video Capture...")
cap = cv2.VideoCapture(0)
print("Done Getting Video Capture!")

print("\nRunning...")
while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28, 28), Image.ANITALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixel_filter = 20
        min_pixel = numpy.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = numpy.clip(image_bw_resized_inverted - min_pixel, 0, 255)

        max_pixel = numpy.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = numpy.asarray(image_bw_resized_inverted_scaled) / max_pixel

        test_sample = numpy.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = log_reg.predict(test_sample)
        print("predicted class is ", test_pred)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()

print("\nExiting App")
