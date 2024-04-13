import cv2
# import joblib
# from opencv_python_headless import cv2
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
import requests
from PIL import Image


#--------- Defining functions -----------------

def find_background(my_matrix):
    new_matrix = my_matrix.reshape(28, 28)

    first_max = np.max(new_matrix[:2])
    last_max = np.max(new_matrix[-2:])

    return np.max([first_max,last_max])


def remove_dead_space(my_matrix, control):
    new_matrix = my_matrix.reshape(28, 28)

    empty_matrix = []
    count_length_deadspace = 0

    for idx, num in enumerate(new_matrix):
        if num.sum() < 1:
            count_length_deadspace += 1
        if num.sum() > 0:
            empty_matrix.append(num)

    empty_matrix = np.array(empty_matrix)
    diff = 28 - empty_matrix.shape[0]

    zeros = np.zeros((diff, empty_matrix.shape[1]))

    if control == True:

        new_empty_matrix = np.concatenate((empty_matrix, zeros), axis=0)
    else:
        new_empty_matrix = np.concatenate((zeros, empty_matrix), axis=0)

    new_empty_matrix = new_empty_matrix.transpose()
    count_length_digit = 28 - count_length_deadspace

    return new_empty_matrix, count_length_deadspace, count_length_digit


def make_thicker(my_matrix):

    zeros = np.zeros((1, 28))

    move_down = my_matrix[:-1]
    one_move_down = np.concatenate((zeros, move_down), axis=0)
    together_down = my_matrix + one_move_down

    together_down = together_down.transpose()

    move_down = together_down[1:]

    one_move_left = np.concatenate((move_down, zeros), axis=0)

    together_left = together_down + one_move_left

    return together_left.transpose()


def process_image(my_image):

    img_resized = cv2.resize(my_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)  # invert image
    img_resized = img_resized.reshape(-1, 784)


    background = find_background(img_resized)
    image_max_pixel = np.max(img_resized)
    transformed_img = img_resized/image_max_pixel

    for item in transformed_img:

        item[item < (background/image_max_pixel) * 1.05] = 0

    img_resized = transformed_img

    new_item = remove_dead_space(img_resized, True)[0]
    img_resized = remove_dead_space(new_item, False)[0]


    my_thick_img = make_thicker(img_resized)

    my_thick_img_transformed = my_thick_img.flatten()/np.max(my_thick_img)

    current_model = build_model()

    st.title("How the computer envisions the image")
    st.image(img_resized, channels="RGB", use_column_width=True)
    st.title(f"Number predicted: {current_model.predict(img_resized.flatten().reshape(1,-1))[0]}")

    current_model = []

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

#####---------- building model ----------
def build_model():
    # my_model = joblib.load(r'C:\Users\Elin\Desktop\DS23\Machine Learning\Kunskapskontroll 2\model.pkl')
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)

    # Looping through MNIST data to put digit in top right corner
    X_new = []

    for item in X:
        #     print(len(item))
        new_item = remove_dead_space(item, True)[0]
        X_new.append(remove_dead_space(new_item, False)[0])

    # Normalize MNIST data
    X_new = np.array(X_new)
    X_new = X_new.reshape(-1,784)
    X_new = X_new / 255 #highest datapoint is always 254-255 in MNIST.

    random_forest_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    random_forest_clf.fit(X_new, y)

    my_model = random_forest_clf

    return my_model

# Function to fetch image from GitHub
def get_image_from_github(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

#####---------- building app ----------

st.title("Take a photo of a handwritten digit")

github_url_1 = "https://raw.githubusercontent.com/FilipHolmbrg/Machine-Learning/main/example_screenshot.jpg"
# Display the image
image = get_image_from_github(github_url_1)
st.image(image, caption="Example image showing how thick the digit needs to be.")
my_img = st.camera_input("")

if my_img is not None:

    bytes_data = my_img.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)

    # save image locally and load local image (jpg)
    # cv2.imwrite("captured_image.jpg", frame)
    # st.success("Image saved as captured_image.jpg")
    # download_image = cv2.imread(r'C:\Users\Elin\Desktop\DS23\Machine Learning\Kunskapskontroll 2\captured_image.jpg',
    #                             cv2.IMREAD_GRAYSCALE)
    # st.image(download_image)


    img_resized = cv2.resize(frame, (28, 28), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)  # invert image
    # st.write(img_resized)

    st.subheader("Adjust and crop image")
    crop_x = st.slider("<- Right/Left ->", 0, frame.shape[1] - 1, int(0.28*frame.shape[1]))  # 200
    crop_y = st.slider("<- Down/Up ->", 0, frame.shape[0] - 1, int(0.13*frame.shape[0]))  # 50
    st.write(crop_y)
    crop_width = st.slider("Width", 1, frame.shape[1], int(0.50*frame.shape[1]))  # 350
    crop_height = st.slider("Height", 1, frame.shape[0], int(0.75*frame.shape[0]))  # 300

    cropped_image = crop_image(frame, crop_x, crop_y, crop_width, crop_height)

    # Drawing horizontal lines:
    line_color = (0, 5, 0)  # Green color (BGR format)
    line_thickness = 1
    line_y = [int(crop_height * 0.08), int(crop_height * 0.92)] # Adjust this to your desired y-coordinate
    cv2.line(cropped_image, (0, line_y[0]), (cropped_image.shape[1], line_y[0]), line_color, line_thickness)
    cv2.line(cropped_image, (0, line_y[1]), (cropped_image.shape[1], line_y[1]), line_color, line_thickness)

    # Display the image with the horizontal line
    st.write('Captured image:')
    st.image(cropped_image, channels="RGB", use_column_width=True)

    github_url_2 = "https://raw.githubusercontent.com/FilipHolmbrg/Machine-Learning/main/example_image.png"
    # Display the image
    image = get_image_from_github(github_url_2)
    st.write("Example image:")
    st.image(image, caption='"Example of image that stays off the above/below lines."')



    if st.button("Use image and predict value"):
        st.subheader("Building model from MNIST data from scratch. Please wait, this may take a minute..")
        process_image(cropped_image)





