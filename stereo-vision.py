"""
stereo-vision.py

This program generates the 3D coordinates of two images

Author: Héctor Camacho Zamora
Date: 12-05-2024
Organization: UDEM

Example: 

python3 stereo-vision.py --l_img left_infrared_image.png --r_img right_infrared_image.png --cal_file calibration-parameters.json
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import sys
import os

left_points = []
right_points = []
left_clk_block = False 


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for stereo-vision.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='stereo-vision')
    parser.add_argument('--l_img', type=str, help='left image')
    parser.add_argument('--r_img', type=str, help='right image')
    parser.add_argument('--cal_file', type=str, help='calibration file')
    args = parser.parse_args()
    return args


def Mouse_events(event:int, x:int, y:int, flags:int, param:int)->None:
    """
    A function to handle mouse events for stereo vision.

    Args:
        event (int): The type of event detected.
        x (int): The x-coordinate where the event occurred.
        y (int): The y-coordinate where the event occurred.
        flags (int): Flags associated with the event.
        param (int): Additional parameters for the event.

    Returns:
        None
    """
    global left_clk_block
    global left_points
    global right_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        left_points.append([x,y])  
        print('Punto en la imagen izquierda: ', len(left_points), ' colocado')
        
    if event == cv2.EVENT_RBUTTONDOWN:
        right_points.append([x,y])
        print('Punto en la imagen derecha: ', len(right_points), ' colocado')
    
    if event == cv2.EVENT_MBUTTONDOWN:
        if len(left_points) >= 1 and len(right_points) >= 1:
            print('\nCalculando puntos en 3D...\n')
            left_clk_block = True  
        else: 
            print('No hay suficientes puntos')
    if flags & cv2.EVENT_FLAG_CTRLKEY:
        print('Borrando puntos...\n\n')
        left_points.clear()
        right_points.clear()
        left_clk_block = False
        run_pipeline()

def load_image(args:str)-> cv2:
    """
    Loads an image using OpenCV based on the provided image path.

    Args:
        args (str): Path to the image file.

    Returns:
        cv2: Loaded image if successful.

    Raises:
        Exception: If the image could not be read.
    """
    img = cv2.imread(args)
    if img is None:
        raise Exception("Could not read the image.")
    return img

def show_image(img:cv2, title:str) -> None:
    """
    Displays an image using OpenCV with the provided title.
    
    Args:
        img (cv2): The image to be displayed.
        title (str): The title of the window displaying the image.

    Returns:
        None
    """
    cv2.imshow(title, img)

def draw_points(img:cv2, points:list) -> None:
    """
    Draws circles on the input image at the specified points.

    Args:
        img (cv2): The image on which circles will be drawn.
        points (list): List of points where circles will be drawn.

    Returns:
        None
    """
    for point in points:
        cv2.circle(img,(point[0],point[1]), 5, (0,0,255), -1)
        cv2.waitKey(10)

def load_calibration_parameters_from_json_file(
        args:argparse.ArgumentParser
        )->None:
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        json_data: A dictionary containing the camera calibration parameters.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """

    # Check if JSON file exists
    json_filename = args.cal_file
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        
        return json_data
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)


def compute_coordinates_and_disparity(points_left: list,points_right: list, json_data: dict) -> tuple:
    """
    A function to compute the coordinates and disparity between two sets of points.

    Args:
        points_left (list): List of points from the left image.
        points_right (list): List of points from the right image.
        json_data (dict): Dictionary containing calibration parameters.

    Returns:
        tuple: A tuple containing the disparity list and the coordinates for the left and right images.
    """
    #coordinates respect image center
    coords_left = []
    coords_right = []
    d_list = []
    cx = json_data['rectified_cx']
    cy = json_data['rectified_cy']
    #left image
    for point in points_left:
        x_left = point[0] - cx
        y_left = point[1] - cy
        coords_left.append([x_left,y_left])

    #right image
    for point in points_right:
        x_right = point[0] - cx
        y_right = point[1] - cy
        coords_right.append([x_right,y_right])

    #disparity
    for coord in coords_left:
        d = coords_right[len(coord)][0] - coord[0]
        d_list.append(d)
        

    return d_list, coords_left, coords_right

def ccompute_final_coords(d: list, coords_left: list, coords_right: list, json_data: dict) -> tuple:
    """
    Compute the final coordinates based on the given parameters and JSON data.

    Parameters:
    - d (list): A list of values
    - coords_left (list): A list of coordinates
    - coords_right (list): A list of coordinates
    - json_data (dict): A dictionary containing JSON data with keys: 'rectified_fx', 'rectified_fy', 'baseline'

    Returns:
    - tuple: A tuple containing final coordinates, X coordinates, Y coordinates, Z coordinates
    """
    fx = json_data['rectified_fx']
    fy = json_data['rectified_fy']
    baseline = json_data['baseline']
    X_coord = []
    Y_coord = []
    Z_coord = []
    final_coordinates = []
    i = 0

    for coord in coords_left:
        Z = (fx * baseline) / d[i]
        X = coord[0]*(Z/fx)
        Y = coord[1]*(Z/fy)
        
        Z_coord.append(Z)
        X_coord.append(X)
        Y_coord.append(Y)
        final_coordinates.append([X,Y,Z])
        i+=1

    return final_coordinates, X_coord, Y_coord, Z_coord

def plot_coordinates(final_coords: list)->None:
    """
    A function to plot the coordinates in a 3D scatter plot.

    Parameters:
    - final_coords (list): A list of final coordinates to be plotted.
    - x_coord (list): A list of X coordinates.
    - y_coord (list): A list of Y coordinates.
    - z_coord (list): A list of Z coordinates.

    Returns:
    - None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(len(final_coords)):
        ax.scatter( final_coords[i][0], final_coords[i][1], final_coords[i][2], c='r', marker='o')
   # ax.axis('equal')
    plt.show()
    
def run_pipeline()-> None:
    """
    Runs the stereo vision pipeline to process images and display the results in real-time.

    This function loads images, calibration parameters, and sets up the GUI windows for left and right images. It then enters a loop to handle mouse events, compute coordinates and disparity, and display the final coordinates in a 3D plot. The loop continues until the user presses 'q' to exit.

    Parameters:
    - None

    Returns:
    - None
    """
    global left_clk_block
    
    args = parse_args()
    img_l = load_image(args.l_img)
    img_r = load_image(args.r_img)
    json_data = load_calibration_parameters_from_json_file(args)

    cv2.namedWindow('Left image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Left image', 640, 480)

    cv2.namedWindow('Right image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Right image', 640, 480)

    cv2.setMouseCallback('Left image', Mouse_events)
    cv2.setMouseCallback('Right image', Mouse_events)
    while True:
        
        img_l_copy = img_l.copy()
        img_r_copy = img_r.copy()
    
        draw_points(img_l_copy,left_points)
        draw_points(img_r_copy,right_points)
            
        
        if left_clk_block == True:
            d_list, coords_left, coords_right = compute_coordinates_and_disparity(right_points,left_points, json_data)  
            final_coordinates, X_coord, Y_coord, Z_coord = ccompute_final_coords(d_list, coords_left, coords_right, json_data)
            plot_coordinates(final_coordinates)
            


        show_image(img_l_copy, 'Left image')
        show_image(img_r_copy, 'Right image')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            left_clk_block = False
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\nBienvenido al programa stereo-vision.")
    print("\nPor favor, coloque los puntos correspondientes en la imagen izquierda y derecha.")
    print("Asegurse de utilizar el click derecho en la imagen de la derecha \ny el click izquierdo en la imagen de la izquierda")
    print("\nDe click a la tecla ctrl para borrar los puntos.")
    print("Presione 'q' para finalizar el programa.")
    run_pipeline()
    

