#%%
import os
import numpy as np
from scipy import signal, optimize
from skimage import io
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import shift
from PIL import Image, ImageSequence
from scipy.optimize import curve_fit
import glob
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
#%%
#input parameters
#folder to save aligned images, cannot be under the monitored folder
<<<<<<< HEAD:Alignimg_1DGaufit_fwhm_figcsv_20250302.py
parent = r'C:\Users\lxiaoyang\Desktop\N41'
save_path = os.path.join(parent, 'aligned')
#if not monitor:
monitor = False
raw_data_path = os.path.join(parent, 'N41')
=======
parent = r'C:\Users\yuchunglin\Documents\1BMB_Mar25\Mar2025\STTR\Zyla\data'
pixel_value= 0.65
save_path = os.path.join(parent, 'aligned')
#if not monitor:

monitor = False
raw_data_path = os.path.join(parent, 'P14_S3')
>>>>>>> a524d40d564594319a740c7a31db7e35354e85f9:Alignimg_1DGaufit_fwhm_figcsv_20250324.py

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created folder: {save_path}")
#folder to save all processed images
processed_path = os.path.join(parent, 'processed')
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
    print(f"Created folder: {processed_path}")
#folder to save polynomial fit plot
plot_save = os.path.join(parent, 'result')
if not os.path.exists(plot_save):
    os.makedirs(plot_save)
    print(f"Created folder: {plot_save}")
#folder to save csv file and csv filename
csv_path = os.path.join(parent,'result')
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
    print(f"Created folder: {csv_path}")
fn = 'fit_results.csv'
#for 2D gaussian fitting
ini_sigma_x, ini_sigma_y = 10, 10 #initial guess for size 
<<<<<<< HEAD:Alignimg_1DGaufit_fwhm_figcsv_20250302.py
crop = True
=======
crop = False
>>>>>>> a524d40d564594319a740c7a31db7e35354e85f9:Alignimg_1DGaufit_fwhm_figcsv_20250324.py
#for 1D gaussian fitting
roi_width = 140
sigma_1d = 6
<<<<<<< HEAD:Alignimg_1DGaufit_fwhm_figcsv_20250302.py
crop_r = 7
=======
crop_r = 50
>>>>>>> a524d40d564594319a740c7a31db7e35354e85f9:Alignimg_1DGaufit_fwhm_figcsv_20250324.py
#%%
def read_images_from_folder(folder):
    '''
    images have same size (.tif)
    to read all images in a folder as a stack
    '''
    
    files = sorted(glob.glob(os.path.join(folder, '*.tif')))
    img_ref = io.imread(files[0])
    img = np.zeros([len(files), img_ref.shape[0], img_ref.shape[1]])
    for i, file in enumerate(files):
        img[i,:,:] = io.imread(file)
    return img

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    Define a 2D Gaussian function
    '''
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def circle(cen, r, size):
    row, col = size[0], size[1]
    x = np.arange(col)
    y = np.arange(row)
    x,y = np.meshgrid(x,y)
    img_circle = np.zeros([row, col],dtype=bool)
    img_circle[(x-cen[1])**2 + (y-cen[0])**2 < r**2] = True  
    #img_circle = np.float32(img_circle)
    return img_circle

def apply_mask(image, cen, r):
    # Get the size of the image (assuming it's a grayscale image)
    size = image.shape
    # Generate the circular mask
    mask = circle(cen, r, size)
    
    # Apply the mask: set the outside area to 0
    image_with_mask = image * mask  # Multiplies element-wise
    
    return image_with_mask
<<<<<<< HEAD:Alignimg_1DGaufit_fwhm_figcsv_20250302.py

# Gaussian function
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C
def residuals(params, x, y):
    return y - gaussian(x, *params)
def fit_and_plot(data, axis_label, subplot_index):
    '''
    center is max value position
    manually set roi_width = 100
    manually set sigma = 6
    '''
    #use max y value to find ROI
    center_index = np.where(data==np.max(data))[0][0]
    roi_start = max(center_index - roi_width, 0)
    roi_end = min(center_index + roi_width, len(data))

    x_data = np.arange(roi_start, roi_end)
    y_data = data[roi_start:roi_end]
    #lower_bounds = [-np.inf, -np.inf, 0.1, -np.inf]  # Ensure sigma > 0.1
    #upper_bounds = [np.inf, np.inf, 100, np.inf]  # Prevent sigma from blowing up

    # Perform Gaussian fit
    initial_guess = [max(y_data)-min(y_data), center_index, sigma_1d, min(y_data)]  # [A, mu, sigma, C]
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess,maxfev=10000)
    A, mu, sigma, C = popt
    '''''
    if A < C:
        print('Do least square fit')
        res = least_squares(
        residuals,
        initial_guess,
        args=(x_data, y_data),
        bounds=(lower_bounds, upper_bounds),
        max_nfev=50000,
        loss='soft_l1'
            )   
        A, mu, sigma, C = res.x

    else:
        pass
    '''
    fwhm = 2.355 * np.abs(sigma)

    # Plot the summation and fit
    plt.subplot(2, 2, subplot_index)
    plt.plot(data, label="Summation")
    plt.axvline(center_index, color='gray', linestyle='--', label="Center")
    plt.plot(x_data, gaussian(x_data, A,mu,sigma,C), 'r--', label=f"Gaussian Fit\nFWHM={fwhm:.2f}")

    plt.title(f"Summation with Gaussian Fit ({axis_label})")
    plt.xlabel("Index")
    plt.ylabel("Sum")
    plt.legend()

    # Plot the ROI with fit
    plt.subplot(2, 2, subplot_index + 1)
    plt.plot(x_data, y_data, 'bo', label="ROI Data")
    plt.plot(x_data, gaussian(x_data, A,mu,sigma,C), 'r--', label=f"Gaussian Fit\nFWHM={fwhm:.2f}")
    plt.title(f"ROI with Gaussian Fit ({axis_label})")
    plt.xlabel("Index (ROI)")
    plt.ylabel("Sum")
    plt.legend()


    return fwhm, mu, A, C
#%%
if monitor:
    if len(sys.argv) < 2:
        print("Error: No folder path provided.")
        sys.exit(1)
    # Get the folder path from the argument
    folder = sys.argv[1]
else:
    folder = raw_data_path
print(f"Processing folder: {folder}")
files = os.listdir(folder)
print(f"Files in {folder}: {len(files)} files")
img = read_images_from_folder(folder)
last_part = os.path.basename(folder) #image folder name
beam_x_list = []
beam_y_list = []
sigma_x_list = []
sigma_y_list = []
ncols = min(4,len(files))
nrows = math.ceil(len(files)/ncols)
fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))  # Adjust figure size
ax = ax.flatten()  # Flatten in case of multiple rows/cols
for i,z in enumerate(range(img.shape[0])):
    ref = img[z,:,:]
    ini_x = np.where(img[z] == img[z].max())[1][0] #initial guess for beam position x
    ini_y = np.where(img[z] == img[z].max())[0][0] #initial guess for beam position y
    # Create x and y indices
    x = np.linspace(0, ref.shape[1] - 1, ref.shape[1])
    y = np.linspace(0, ref.shape[0] - 1, ref.shape[0])
    x, y = np.meshgrid(x, y)

=======

# Gaussian function
def gaussian(x, A, mu, sigma, C):
    return (A / (sigma*np.sqrt(2*np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C
def residuals(params, x, y):
    return y - gaussian(x, *params)
def fit_and_plot(data, axis_label, subplot_index):
    '''
    center is max value position
    manually set roi_width = 100
    manually set sigma = 6
    '''
    #use max y value to find ROI
    center_index = np.where(data==np.max(data))[0][0]
    roi_start = max(center_index - roi_width, 0)
    roi_end = min(center_index + roi_width, len(data))

    x_data = np.arange(roi_start, roi_end)
    y_data = data[roi_start:roi_end]
    #lower_bounds = [-np.inf, -np.inf, 0.1, -np.inf]  # Ensure sigma > 0.1
    #upper_bounds = [np.inf, np.inf, 100, np.inf]  # Prevent sigma from blowing up

    # Perform Gaussian fit
    initial_guess = [max(y_data)-min(y_data), center_index, sigma_1d, min(y_data)]  # [A, mu, sigma, C]
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess,maxfev=100000)
    A, mu, sigma, C = popt
    #from scipy.optimize import least_squares
    #def residuals(params, x, y):
    #    return y - gaussian(x, *params)

    #res = least_squares(
    #residuals,
    #initial_guess,
    #args=(x_data, y_data),
    #loss='soft_l1',
    #max_nfev=10000
    #)
    #A, mu, sigma, C = res.x
    '''''
    if A < C:
        print('Do least square fit')
        res = least_squares(
        residuals,
        initial_guess,
        args=(x_data, y_data),
        bounds=(lower_bounds, upper_bounds),
        max_nfev=50000,
        loss='soft_l1'
            )   
        A, mu, sigma, C = res.x

    else:
        pass
    '''
    fwhm = 2.355 * np.abs(sigma)

    # Plot the summation and fit
    plt.subplot(2, 2, subplot_index)
    plt.plot(data, label="Summation")
    plt.axvline(center_index, color='gray', linestyle='--', label="Center")
    plt.plot(x_data, gaussian(x_data, A,mu,sigma,C), 'r--', label=f"Gaussian Fit\nFWHM={fwhm:.2f}")

    plt.title(f"Summation with Gaussian Fit ({axis_label})")
    plt.xlabel("Index")
    plt.ylabel("Sum")
    plt.legend()

    # Plot the ROI with fit
    plt.subplot(2, 2, subplot_index + 1)
    plt.plot(x_data, y_data, 'bo', label="ROI Data")
    plt.plot(x_data, gaussian(x_data, A,mu,sigma,C), 'r--', label=f"Gaussian Fit\nFWHM={fwhm:.2f}")
    plt.title(f"ROI with Gaussian Fit ({axis_label})")
    plt.xlabel("Index (ROI)")
    plt.ylabel("Sum")
    plt.legend()


    return fwhm, mu, A, C
#%%
if monitor:
    if len(sys.argv) < 2:
        print("Error: No folder path provided.")
        sys.exit(1)
    # Get the folder path from the argument
    folder = sys.argv[1]
else:
    folder = raw_data_path
print(f"Processing folder: {folder}")
files = os.listdir(folder)
print(f"Files in {folder}: {len(files)} files")
img = read_images_from_folder(folder)
last_part = os.path.basename(folder) #image folder name
beam_x_list = []
beam_y_list = []
sigma_x_list = []
sigma_y_list = []
ncols = min(4,len(files))
nrows = math.ceil(len(files)/ncols)
fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))  # Adjust figure size
ax = ax.flatten()  # Flatten in case of multiple rows/cols
for i,z in enumerate(range(img.shape[0])):
    ref = img[z,:,:]
    ini_x = np.where(img[z] == img[z].max())[1][0] #initial guess for beam position x
    ini_y = np.where(img[z] == img[z].max())[0][0] #initial guess for beam position y
    # Create x and y indices
    x = np.linspace(0, ref.shape[1] - 1, ref.shape[1])
    y = np.linspace(0, ref.shape[0] - 1, ref.shape[0])
    x, y = np.meshgrid(x, y)

>>>>>>> a524d40d564594319a740c7a31db7e35354e85f9:Alignimg_1DGaufit_fwhm_figcsv_20250324.py
    # Initial guess for the parameters
    initial_guess = (np.max(ref), ini_x, ini_y, ini_sigma_x, ini_sigma_y, 0, 10)
    # Bounds for the parameters
    bounds = (
        [0, 0, 0, 0, 0, -np.pi/4, 0],  # Lower bounds
        [np.inf, ref.shape[1], ref.shape[0], np.inf, np.inf, np.pi/4, np.max(ref)]  # Upper bounds
    )

    # Fit the Gaussian model to the data
    popt, _ = optimize.curve_fit(gaussian_2d, (x, y), ref.ravel(), p0=initial_guess,bounds=bounds,maxfev=10000)

    # Extract the beam position from the fit parameters
    beam_x = popt[1]
    beam_y = popt[2]
    sigma_x = popt[3]
    sigma_y = popt[4]

    print(f"Slice: {z}, Beam position: x = {beam_x}, y = {beam_y}, sigma_x = {sigma_x}, sigma_y = {sigma_y}")
    beam_x_list.append(beam_x)
    beam_y_list.append(beam_y)
    sigma_x_list.append(float(sigma_x))
    sigma_y_list.append(float(sigma_y))
    #if use previous beam position as initial guess
<<<<<<< HEAD:Alignimg_1DGaufit_fwhm_figcsv_20250302.py
    ini_x = beam_x #update initial guess for beam position x
    ini_y = beam_y #update initial guess for beam position y
=======
    #ini_x = beam_x #update initial guess for beam position x
    #ini_y = beam_y #update initial guess for beam position y
>>>>>>> a524d40d564594319a740c7a31db7e35354e85f9:Alignimg_1DGaufit_fwhm_figcsv_20250324.py
    #use previous sigma as initial guess
    ini_sigma_x = sigma_x
    ini_sigma_y = sigma_y
    # Use the optimized parameters to create the fitted Gaussian
    data_fitted = gaussian_2d((x, y), *popt)

    # Plot the results
    ax[i].imshow(np.log1p(ref), cmap='viridis', origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))#,vmin=np.min(ref), vmax=np.max(ref))
    ax[i].contour(x, y, data_fitted.reshape(ref.shape[0], ref.shape[1]), 6, colors='w')
    ax[i].set_title(f"Slice {z+1} 2D fit ", fontsize=12, fontweight="bold")
for j in range(i+1, len(ax)):
        fig.delaxes(ax[j])
plt.tight_layout()
os.makedirs(processed_path, exist_ok=True)
plt.savefig(f'{processed_path}\\{last_part}_2Dgaussian_fit.jpg')
plt.close(fig)
print(f'Done {last_part} 2D image gaussian fit')
# %%
#shift image
ref = img[0,:,:]
mv_y = []
mv_x = []
for i, y in enumerate(beam_y_list):
    if i == 0:
        ref_y = y
        ref_x = beam_x_list[i]
    else:
        y_mv = ref_y - y
        x_mv = ref_x - beam_x_list[i]
        mv_y.append(y_mv)
        mv_x.append(x_mv)
for i,y2 in enumerate(mv_y):
    img[i+1,:,:] = shift(img[i+1,:,:], (y2, mv_x[i]))
print(f'Done {last_part} shift')
# %%
#save image
os.makedirs(save_path, exist_ok=True)
io.imsave(f'{save_path}\\{last_part}.tif',np.float32(img))
print(f'Done {last_part} shifted image save')
# %%
# Load the TIFF stack file
image_path = f'{save_path}\\{last_part}.tif'
print(f"Loading image from: {image_path}")
if not crop:
    # Open the TIFF stack
    image = io.imread(image_path)
    fwhm_col_list = []
    fwhm_row_list = []
    pdf_path = f'{processed_path}\\{last_part}_1Dgaussian_fit.pdf'
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"Deleted existing file: {pdf_path}")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    # Iterate through frames in the TIFF stack
    with PdfPages(pdf_path) as pdf:
        for i, frame in enumerate(image):
            # Convert the frame to a NumPy array
            frame_array = np.array(frame)
<<<<<<< HEAD:Alignimg_1DGaufit_fwhm_figcsv_20250302.py
            
            # Calculate row and column summations
            sum_rows = np.sum(frame_array, axis=1)  # Summing columns
            sum_columns = np.sum(frame_array, axis=0)  # Summing rows

            # Create a new figure for each frame
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f"slice {i+1} 1D fit", fontsize=12, fontweight="bold")
            # Fit and plot for row summation
            fwhm_row, mu_row, A_row, C_row = fit_and_plot(sum_rows, "Row Direction", 1)
            fwhm_row_list.append(fwhm_row)
            # Fit and plot for column summation
            fwhm_col, mu_col, A_col, C_col = fit_and_plot(sum_columns, "Column Direction", 3)
            fwhm_col_list.append(fwhm_col)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Frame {i + 1}:")
            print(f"  Row Direction - FWHM: {fwhm_row:.2f}, Center: {mu_row:.2f}, Amplitude: {A_row:.2f}, Baseline: {C_row:.2f}")
            print(f"  Column Direction - FWHM: {fwhm_col:.2f}, Center: {mu_col:.2f}, Amplitude: {A_col:.2f}, Baseline: {C_col:.2f}")
else:
    # Open the TIFF stack
    image = io.imread(image_path)
    fwhm_col_list = []
    fwhm_row_list = []
    pdf_path = f'{processed_path}\\{last_part}_1Dgaussian_fit.pdf'
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"Deleted existing file: {pdf_path}")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    # Iterate through frames in the TIFF stack
    with PdfPages(pdf_path) as pdf:
        yc, xc = np.where(image[0] == np.max(image[0]))
        for i, frame in enumerate(image):
            # Convert the frame to a NumPy array
            frame_array = np.array(frame)
            frame_array = apply_mask(frame_array, (yc[0],xc[0]), r=crop_r)
            # Calculate row and column summations
            sum_rows = np.sum(frame_array, axis=1)  # Summing columns
            sum_columns = np.sum(frame_array, axis=0)  # Summing rows

=======
            center_index = np.where(frame==np.max(frame))[0][0]
            roi_start = center_index - roi_width
            roi_end = center_index + roi_width
            # Calculate row and column summations
            '''
            sum_rows = np.sum(frame_array, axis=1)  # Summing columns
            sum_columns = np.sum(frame_array, axis=0)  # Summing rows
            '''
            center_y, center_x = np.unravel_index(np.argmax(frame_array), frame_array.shape)
            crop_r = roi_width  # you can adjust this radius as needed

            # Define cropping boundaries
            y_start = max(center_y - crop_r, 0)
            y_end = min(center_y + crop_r, frame_array.shape[0])
            x_start = max(center_x - crop_r, 0)
            x_end = min(center_x + crop_r, frame_array.shape[1])

            # Crop the frame around the peak
            roi_frame = frame_array[y_start:y_end, x_start:x_end]

            # Perform 1D summation only on the cropped region
            sum_rows = np.sum(roi_frame, axis=1)  # Along columns in cropped frame
            sum_columns = np.sum(roi_frame, axis=0)  # Along rows in cropped frame


            # Create a new figure for each frame
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f"slice {i+1} 1D fit", fontsize=12, fontweight="bold")
            # Fit and plot for row summation
            fwhm_row, mu_row, A_row, C_row = fit_and_plot(sum_rows, "Row Direction", 1)
            fwhm_row_list.append(fwhm_row)
            # Fit and plot for column summation
            fwhm_col, mu_col, A_col, C_col = fit_and_plot(sum_columns, "Column Direction", 3)
            fwhm_col_list.append(fwhm_col)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Frame {i + 1}:")
            print(f"  Row Direction - FWHM: {fwhm_row:.2f}, Center: {mu_row:.2f}, Amplitude: {A_row:.2f}, Baseline: {C_row:.2f}")
            print(f"  Column Direction - FWHM: {fwhm_col:.2f}, Center: {mu_col:.2f}, Amplitude: {A_col:.2f}, Baseline: {C_col:.2f}")
else:
    # Open the TIFF stack
    image = io.imread(image_path)
    fwhm_col_list = []
    fwhm_row_list = []
    pdf_path = f'{processed_path}\\{last_part}_1Dgaussian_fit.pdf'
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"Deleted existing file: {pdf_path}")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    # Iterate through frames in the TIFF stack
    with PdfPages(pdf_path) as pdf:
        yc, xc = np.where(image[0] == np.max(image[0]))
        for i, frame in enumerate(image):
            # Convert the frame to a NumPy array
            frame_array = np.array(frame)
            frame_array = apply_mask(frame_array, (yc[0],xc[0]), r=crop_r)
            # Calculate row and column summations
            sum_rows = np.sum(frame_array, axis=1)  # Summing columns
            sum_columns = np.sum(frame_array, axis=0)  # Summing rows

>>>>>>> a524d40d564594319a740c7a31db7e35354e85f9:Alignimg_1DGaufit_fwhm_figcsv_20250324.py
            # Create a new figure for each frame
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f"slice {i+1} 1D fit", fontsize=12, fontweight="bold")
            # Fit and plot for row summation
            fwhm_row, mu_row, A_row, C_row = fit_and_plot(sum_rows, "Row Direction", 1)
            fwhm_row_list.append(fwhm_row)
            # Fit and plot for column summation
            fwhm_col, mu_col, A_col, C_col = fit_and_plot(sum_columns, "Column Direction", 3)
            fwhm_col_list.append(fwhm_col)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Frame {i + 1}:")
            print(f"  Row Direction - FWHM: {fwhm_row:.2f}, Center: {mu_row:.2f}, Amplitude: {A_row:.2f}, Baseline: {C_row:.2f}")
            print(f"  Column Direction - FWHM: {fwhm_col:.2f}, Center: {mu_col:.2f}, Amplitude: {A_col:.2f}, Baseline: {C_col:.2f}")
            
print(f'Done {last_part} 1D gaussian fit')

#%%
# Polynomial fitting
files = os.listdir(folder)[0]
center = float(files.split('_')[1].split('mm')[0])
width = float(files.split('_')[2].split('mm')[0])
x = np.arange(center-width//2,center+width//2+1,2)
degree = 12
# Plot the updated data points
fig = plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots(1, 1)
fwhm_col_list = np.abs(fwhm_col_list)
fwhm_row_list = np.abs(fwhm_row_list)
ax1.scatter(x, fwhm_col_list*pixel_value, label="Updated Data points Horizontal", color="blue")
ax1.scatter(x, fwhm_row_list*pixel_value, label="Updated Data points Vertical", color="green")
# Polynomial fitting on updated data
coefficients_updated_col = np.polyfit(x, fwhm_col_list, degree)
polynomial_updated_col = np.poly1d(coefficients_updated_col)

coefficients_updated_row = np.polyfit(x, fwhm_row_list, degree)
polynomial_updated_row = np.poly1d(coefficients_updated_row)
# Generate values for the fitted curve
x_fit = np.linspace(min(x), max(x), 500)
y_fit_updated_col = polynomial_updated_col(x_fit)
y_fit_updated_row = polynomial_updated_row(x_fit)

# Plot the polynomial fit for updated data
ax1.plot(x_fit, y_fit_updated_col*pixel_value, label=f"Polynomial fit Horizontal (degree {degree})", color="blue")
ax1.plot(x_fit, y_fit_updated_row*pixel_value, label=f"Polynomial fit Vertical (degree {degree})", color="green")

# Add labels and legend
ax1.set_xlabel("Distance from lenses (mm)")
ax1.set_ylabel("Focal spot size (FWHM) (um)")
ax1.set_title(f"Polynomial Fit for {last_part} fwhm")
ax1.legend()
# ax1.grid(True)

y_fit_updated_col_min = np.min(y_fit_updated_col)
min_x_col = x_fit[np.argmin(y_fit_updated_col)]
y_fit_updated_row_min = np.min(y_fit_updated_row)
min_x_row = x_fit[np.argmin(y_fit_updated_row)]

print(f"The minimum y value for col is {y_fit_updated_col_min:.2f} at x = {min_x_col:.5f}.")
print(f"The minimum y value for row is {y_fit_updated_row_min:.2f} at x = {min_x_row:.5f}.")
fig.savefig(f'{plot_save}\\{last_part}_fwhm.tif')
print(f'Done {last_part} polynomial fit')

# %%
df_path = os.path.join(csv_path, fn)
# Check if the file exists in the current directory
if not os.path.exists(df_path):
    df = pd.DataFrame(columns=["fn", 
                               "x at col min", 
                               "col min", 
                               "x at row min", 
                               "row min"])
    df.to_csv(df_path,index=False)
else:
    df = pd.read_csv(df_path)
if last_part in df['fn'].values:
    print(f'Warning: {last_part} already analyzed')
df.loc[len(df)] = [last_part,
                   round(min_x_col,3), 
                   round(y_fit_updated_col_min,3), 
                   round(min_x_row,3), 
                   round(y_fit_updated_row_min,3)]
df.to_csv(df_path,index=False)
print(f'Done add {last_part} to csv')
print(f'{folder} All done!')
