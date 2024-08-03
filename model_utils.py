import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from PIL import Image

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

def display_image_with_boxes(image, boxes, labels, scores, score_threshold=0.5, max_width=800):
    # Resize the image to fit within max_width while maintaining aspect ratio
    image.thumbnail((max_width, max_width))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    ax = plt.gca()

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            edgecolor = 'r' if label == 1 else 'g'
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor=edgecolor, facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, f'Label: {label}, Score: {score:.2f}')

    plt.axis('off')
    return plt

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
