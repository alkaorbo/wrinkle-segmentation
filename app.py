import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import dlib
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K

# ========================
# Custom Loss Functions
# ========================
def sigmoid_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.5):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())  # Prevent instability
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # Compute probability term
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)  # Class balancing
    loss = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)  # Focal loss formula
    return K.mean(loss)

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def f1_score(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    precision = intersection / (K.sum(y_pred_f) + smooth)
    recall = intersection / (K.sum(y_true_f) + smooth)
    return (2. * precision * recall) / (precision + recall + smooth)
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
def combined_dice_focal_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * sigmoid_focal_crossentropy(y_true, y_pred)

# ========================
# Load Model
# ========================
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(
        "C:/Users/guest_/tazza_env/face_alignment_augument1.h5",custom_objects={"dice_coefficient": dice_coefficient,"iou":iou, "combined_dice_focal_loss": combined_dice_focal_loss,
                        "Precision": Precision, "Recall": Recall}
    )

model = load_model()

# ========================
# Face Detection & Preprocessing
# ========================
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/guest_/Downloads/ss.dat")

def rect_to_tuple(rect):
    scale = 1.4
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2

    new_w, new_h = int(w * scale), int(h * scale)
    x1_new, y1_new = max(0, cx - new_w // 2), max(0, cy - new_h // 2)
    x2_new, y2_new = cx + new_w // 2, cy + new_h // 2

    return x1_new, y1_new, x2_new, y2_new

def extract_eye_center(shape, eye_indices):
    points = [shape.part(i) for i in eye_indices]
    xs, ys = [p.x for p in points], [p.y for p in points]
    return sum(xs) // len(xs), sum(ys) // len(ys)

def angle_between_2_points(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def get_rotation_matrix(p1, p2, img_shape):
    angle = angle_between_2_points(p1, p2)
    h, w = img_shape[:2]
    return cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

def crop_image(image, det):
    x1, y1, x2, y2 = rect_to_tuple(det)
    return image[y1:y2, x1:x2]

def preprocess_image(img):
    img = np.array(img)  
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = detector(gray_img)
    if len(faces) == 0:
        return None, None  # No face detected

    face = faces[0]  # Process only the first detected face
    landmarks = predictor(gray_img, face)
    left_eye = extract_eye_center(landmarks, LEFT_EYE_INDICES)
    right_eye = extract_eye_center(landmarks, RIGHT_EYE_INDICES)

    M = get_rotation_matrix(left_eye, right_eye, gray_img.shape)
    rotated = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0]), flags=cv2.INTER_CUBIC)
    cropped = crop_image(rotated, face)

    cropped = cropped / 255.0  # Normalize
    cropped = np.stack([cropped] * 3, axis=-1)  # Ensure 3 channels
    cropped = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_CUBIC)

    x_tensor = tf.convert_to_tensor(cropped, dtype=tf.float32)
    x_tensor = tf.expand_dims(x_tensor, axis=0)
    return x_tensor, cropped

# ========================
# Overlay Mask on Image
# ========================
def overlay_mask(cropped, mask, alpha=0.5):
    mask_resized = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    
    cropped_uint8 = (cropped * 255).astype(np.uint8)
    overlay_cropped = cv2.addWeighted(cropped_uint8, 1, colored_mask, alpha, 0)

    return overlay_cropped  # Return the overlayed image

# ========================
# Streamlit App
# ========================
st.title("Wrinkle Segmentation App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    x_tensor, cropped = preprocess_image(image)

    if x_tensor is None:
        st.error("No face detected. Please upload another image.")
    else:
        mask = model.predict(x_tensor)[0]
        mask = np.squeeze(mask)
        mask = np.clip(mask, 0, 1) * 255
        mask = mask.astype(np.uint8)

        result = overlay_mask(cropped, mask)

        st.image(result, caption="Predicted Wrinkles Overlay", use_column_width=True)


# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# import matplotlib.pyplot as plt 
# from tensorflow.keras.losses import BinaryCrossentropy
# import tensorflow.keras.backend as K
# from tensorflow.keras.metrics import Precision, Recall


# def sigmoid_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.5):
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())  # Prevent instability
#     p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # Compute probability term
#     alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)  # Class balancing
#     loss = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)  # Focal loss formula
#     return K.mean(loss)

# def dice_coefficient(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def iou(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
#     return (intersection + smooth) / (union + smooth)

# def f1_score(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     precision = intersection / (K.sum(y_pred_f) + smooth)
#     recall = intersection / (K.sum(y_true_f) + smooth)
#     return (2. * precision * recall) / (precision + recall + smooth)
# def dice_loss(y_true, y_pred):
#     return 1 - dice_coefficient(y_true, y_pred)
# def combined_dice_focal_loss(y_true, y_pred):
#     return 0.5 * dice_loss(y_true, y_pred) + 0.5 * sigmoid_focal_crossentropy(y_true, y_pred)

# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model("C:/Users/guest_/tazza_env/face_alignment_augument1.h5", custom_objects={"dice_coefficient":dice_coefficient, "combined_dice_focal_loss":combined_dice_focal_loss, "iou":iou, "Precision":Precision, "Recall":Recall})
#     return model

# model = load_model()

# import cv2
# import dlib
# import numpy as np

# LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# def rect_to_tuple(rect):
#     scale=1.4
#     x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    
#     # Compute center and size
#     w, h = x2 - x1, y2 - y1
#     cx, cy = x1 + w // 2, y1 + h // 2

#     # Expand the box
#     new_w, new_h = int(w * scale), int(h * scale)
    
#     # Calculate new corners
#     x1_new = max(0, cx - new_w // 2)
#     y1_new = max(0, cy - new_h // 2)
#     x2_new = cx + new_w // 2
#     y2_new = cy + new_h // 2

#     return x1_new, y1_new, x2_new, y2_new
#     # return rect.left(), rect.top(), rect.right(), rect.bottom()

# def extract_eye(shape, eye_indices):
#     return [shape.part(i) for i in eye_indices]

# def extract_eye_center(shape, eye_indices):
#     points = extract_eye(shape, eye_indices)
#     xs = [p.x for p in points]
#     ys = [p.y for p in points]
#     return sum(xs) // len(xs), sum(ys) // len(ys)

# def angle_between_2_points(p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2
#     return np.degrees(np.arctan2(y2 - y1, x2 - x1))

# def get_rotation_matrix(p1, p2, img_shape):
#     angle = angle_between_2_points(p1, p2)
#     h, w = img_shape[:2]
#     center = (w // 2, h // 2)
#     return cv2.getRotationMatrix2D(center, angle, 1.0)

# def crop_image(image, det):
#     left, top, right, bottom = rect_to_tuple(det)
#     return image[top:bottom, left:right]

# # Function to preprocess input image
# def preprocess_image(img):
#     img = np.array(img)  

#     # Now apply OpenCV conversion
#     gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("C:/Users/guest_/Downloads/ss.dat")
#     # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray_img)
#     if len(faces) == 0:
#         print("No face detected.")
#         cropped = gray_img
#     else:
#         for face in faces:
#             landmarks = predictor(gray_img, face)  # Predict landmarks
#             left_eye = extract_eye_center(landmarks, LEFT_EYE_INDICES)
#             right_eye = extract_eye_center(landmarks, RIGHT_EYE_INDICES)
#             M = get_rotation_matrix(left_eye, right_eye, gray_img.shape)
#             rotated = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0]), flags=cv2.INTER_CUBIC)
            
#             cropped = crop_image(rotated, face)  # Crop face region

#     # Normalize & Prepare for Model
#     cropped = cropped / 255.0
#     cropped = np.stack([cropped] * 3, axis=-1)  # Ensure 3 channels
#     cropped = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_CUBIC)

#     x_tensor = tf.convert_to_tensor(cropped, dtype=tf.float32)
#     x_tensor = tf.expand_dims(x_tensor, axis=0)
#     return x_tensor, cropped

# # Function to overlay mask on image
# def overlay_mask(cropped, mask, alpha=0.5):
#     # Resize Mask to Match Cropped Face Size
#     print(cropped.shape)
#     print(cropped.dtype)
#     mask_resized = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_CUBIC)

#     # Apply Color Map
#     colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

#     # Convert cropped face to uint8
#     cropped_uint8 = (cropped * 255).astype(np.uint8)

#     # Blend Mask with Cropped Face
#     alpha = 0.5  
#     overlay_cropped = cv2.addWeighted(cropped_uint8, 1, colored_mask, alpha, 0)

#     # Display Results (Only Cropped Face and Mask)
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.imshow(cv2.cvtColor(cropped_uint8, cv2.COLOR_BGR2RGB))
#     plt.title("Cropped Face")
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.imshow(cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB))
#     plt.title("mask")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.imshow(cv2.cvtColor(overlay_cropped, cv2.COLOR_BGR2RGB))
#     plt.title("Cropped Face with Mask Overlay")
#     plt.axis("off")

#     plt.show() 


# st.title("Wrinkle Segmentation App")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess and predict
#     x_tensor, cropped = preprocess_image(image)
#     mask = model.predict(x_tensor)[0] 
#     mask = np.squeeze(mask)  # Remove batch dimension
#     mask = np.clip(mask, 0, 1)  # Ensure values are in [0,1]
#     mask = (mask * 255).astype(np.uint8) 

    
#     # Assuming single-channel output

#     # Convert mask to binary (thresholding)
#     # mask_binary = (predicted_mask > 0.5).astype(np.uint8)

#     # # Convert image to numpy array
#     # image_np = np.array(image)

#     # Overlay mask
#     result = overlay_mask(cropped, mask)

#     # Display results
#     st.image(result, caption="Predicted Wrinkles Overlay", use_column_width=True)
