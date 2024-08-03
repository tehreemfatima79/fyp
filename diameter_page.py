import cv2
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    gamma_corrected_image = np.power(image / 255.0, gamma)
    gamma_corrected_image = np.uint8(gamma_corrected_image * 255)
    return gamma_corrected_image

def resize_image(image, size=(1600, 1200)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    resized_image = pil_image.resize(size)
    return cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

def crop_image(image, crop_box=(500, 150, 1050, 1050)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cropped_image = pil_image.crop(crop_box)
    return cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

def sharpen_image(image):
    sharpened_image = cv2.convertScaleAbs(image - cv2.Laplacian(image, cv2.CV_64F))
    return sharpened_image

def median_filtering(image, kernel_size=5):
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image

def edge_detection(image):
    b, g, r = cv2.split(image)
    b_blurred = cv2.GaussianBlur(b, (3, 3), 0)
    g_blurred = cv2.GaussianBlur(g, (3, 3), 0)
    r_blurred = cv2.GaussianBlur(r, (3, 3), 0)
    b_edges = cv2.Canny(b_blurred, 60, 90)
    g_edges = cv2.Canny(g_blurred, 60, 90)
    r_edges = cv2.Canny(r_blurred, 60, 90)
    edges = np.maximum(np.maximum(b_edges, g_edges), r_edges)
    return edges

def calculate_diameter(edges, original_image, pixel_size):
    height, width = edges.shape
    y_divided_by_2 = height // 2
    x_coordinates_fixed_y = [x for x in range(width) if edges[y_divided_by_2, x] == 255]
    if x_coordinates_fixed_y:
        min_x = min(x_coordinates_fixed_y)
        max_x = max(x_coordinates_fixed_y)
        distance = max_x - min_x
        calc_dia = distance * pixel_size

        for x in x_coordinates_fixed_y:
            cv2.circle(original_image, (x, y_divided_by_2), 1, (0, 255, 0), -1)
        
        return calc_dia, original_image
    else:
        print("No edge points found at y = y_divided_by_2.")
        return None, None

def calculate_error(measured_diameter, actual_diameter):
    error = abs(measured_diameter - actual_diameter)
    return error

# def process_image(image_path, gamma=0.77, crop_box=(500, 150, 1050, 1050), pixel_size=21910/88):
#     image = cv2.imread(image_path)
#     corrected_image = gamma_correction(image, gamma)
#     resized_image = resize_image(corrected_image)
#     cropped_image = crop_image(resized_image, crop_box)
#     sharpened_image = sharpen_image(cropped_image)
#     denoised_image = median_filtering(sharpened_image)
#     edges = edge_detection(denoised_image)
#     diameter, final_image = calculate_diameter(edges, denoised_image, pixel_size)
#     return diameter, final_image

# def show_diameter_page():
#     st.title("Diameter Measurement")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         actual_diameter = st.number_input("Enter the actual diameter in micrometers:", min_value=0.0, format="%.2f")
        
#         image_path = f'/mnt/data/{uploaded_file.name}'
#         with open(image_path, 'wb') as f:
#             f.write(uploaded_file.getbuffer())

#         diameter, final_image = process_image(image_path)
#         if diameter:
#             st.image(final_image, caption=f"Measured Diameter: {diameter} micrometers", use_column_width=True)
#             error = calculate_error(diameter, actual_diameter)
#             st.write(f"The error is: {error} micrometers")

def process_image(image, gamma=0.77, crop_box=(500, 150, 1050, 1050), pixel_size=21910/88):
    corrected_image = gamma_correction(image, gamma)
    resized_image = resize_image(corrected_image)
    cropped_image = crop_image(resized_image, crop_box)
    sharpened_image = sharpen_image(cropped_image)
    denoised_image = median_filtering(sharpened_image)
    edges = edge_detection(denoised_image)
    diameter, final_image = calculate_diameter(edges, denoised_image, pixel_size)
    return diameter, final_image

from io import BytesIO

def show_diameter_page():
    st.title("Diameter Measurement")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        actual_diameter = st.number_input("Enter the actual diameter in micrometers:", min_value=0.0, format="%.2f")
        
        # Use BytesIO to read the file directly
        image_bytes = BytesIO(uploaded_file.read())
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        
        diameter, final_image = process_image(image)
        if diameter:
            # Resize the image to fit within a specific width while maintaining aspect ratio
            max_width = 800  # Set max width for display
            height, width, _ = final_image.shape
            if width > max_width:
                aspect_ratio = height / width
                new_width = max_width
                new_height = int(max_width * aspect_ratio)
                final_image = cv2.resize(final_image, (new_width, new_height))
            
            # Convert the image to RGB format for displaying in Streamlit
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            st.image(final_image_rgb, caption=f"Measured Diameter: {diameter} micrometers", use_column_width=True)
            error = calculate_error(diameter, actual_diameter)
            st.write(f"The error is: {error} micrometers")

