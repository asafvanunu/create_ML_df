# %%
import os
import rioxarray
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from datetime import datetime, timedelta, time
import shapely
import rasterio
import rioxarray
import warnings
import xarray as xr
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
import itertools
import random

# %%
def open_MCMI(MCMI_path, band_number):
    """This function opens the MCMI NetCDF file and returns the band of the given band number

    Args:
        MCMI_path (string): The path of the MCMI NetCDF file for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'
        band_number (int): The band number. for example 1

    Returns:
        xarray.DataArray: The band of the given band number
    """
    if (band_number < 1) or (band_number > 16): ## Check if the band number is between 1 and 16
        raise ValueError("The band number should be between 1 and 16") ## Raise an error
    
    band = f"CMI_C{band_number:02d}" ## The band name
    try:
        GOES_file = rioxarray.open_rasterio(MCMI_path) ## Open the MCMI NetCDF file
        GOES_CRS = GOES_file.rio.crs ## Get the CRS of the file
        MCMI = GOES_file.copy() ## Copy the MCMI file
    except: ## If there is an error
        print(f"Error in opening the MCMI file: {MCMI_path}") ## Print an error message
        return None ## Return None
    MCMI = MCMI.astype("float32") ## Convert the MCMI to float32
    MCMI_add_factor = MCMI[band].attrs["add_offset"] ## Get the add offset
    MCMI_scale_factor = MCMI[band].attrs["scale_factor"] ## Get the scale factor
    MCMI_fill_value = MCMI[band].attrs["_FillValue"] ## Get the fill value
    MCMI_values = MCMI[band].values[0] ## Get the values of the band
    MCMI_values[MCMI_values == MCMI_fill_value] = np.nan ## set the fill value to nan
    MCMI[band].values[0] = MCMI_values * MCMI_scale_factor + MCMI_add_factor ## Get the values of the band
    MCMI[band] = MCMI[band].rio.write_crs(GOES_CRS) ## Write the CRS of the band
    return MCMI[band] ## Return the values of the band

# %%
def open_FDC(FDC_path, product_name):
    """This function opens the FDC NetCDF file and returns the fire detection confidence

    Args:
        FDC_path (string): The path of the FDC NetCDF file for example 'F:\\ML_project\\GOES_16\\FDC\\OR_ABI-L2-FDCC-M6_G16_s202301010751.nc'
        product_name (string): The product name. for example 'Mask' or "Temp" or "Power"

    Returns:
        xarray.DataArray: The fire product values
    """
    try:
        GOES_file = rioxarray.open_rasterio(FDC_path) ## Open the FDC NetCDF file
        GOES_CRS = GOES_file.rio.crs ## Get the CRS of the file
        FDC = GOES_file.copy() ## Copy the FDC file
        FDC = FDC.astype("float32") ## Convert the FDC to float32
        FDC_add_factor = FDC[product_name].attrs["add_offset"] ## Get the add offset
        FDC_scale_factor = FDC[product_name].attrs["scale_factor"] ## Get the scale factor
        FDC_fill_value = FDC[product_name].attrs["_FillValue"] ## Get the fill value
        FDC_values = FDC[product_name].values[0] ## Get the values of the band
        FDC_values[FDC_values == FDC_fill_value] = np.nan ## set the fill value to nan
        FDC[product_name].values[0] = FDC_values * FDC_scale_factor + FDC_add_factor ## Get the values of the fire detection confidence
        FDC[product_name] = FDC[product_name].rio.write_crs(GOES_CRS) ## Write the CRS of the fire detection confidence
        return FDC[product_name] ## Return the values of the fire detection confidence
    except: ## If there is an error
        print(f"Error in opening the FDC file: {FDC_path}") ## Print an error message
        return None ## Return None

# %%
def open_ACM(ACM_path, product_name):
    """This function opens the ACM NetCDF file and returns the clear sky mask

    Args:
        ACM_path (string): The path of the ACM NetCDF file for example 'F:\\ML_project\\GOES_16\\ACM\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc'
        product_name (string): The product name. for example 'ACM' for 4 level classification where:
        0: Clear, 1: Probably Clear, 2: Probably Cloudy, 3: Cloudy
        and BCM for 2 level classification where:
        0: Clear, 1: Cloudy 

    Returns:
        xarray.DataArray: clear sky mask values
    """
    try:
        GOES_image = rioxarray.open_rasterio(ACM_path) ## Open the ACM NetCDF file
        GOES_CRS = GOES_image.rio.crs ## Get the CRS of the file
        ACM = GOES_image.copy() ## Copy the ACM file
        ACM = ACM.astype("float32") ## Convert the ACM to float32
        ACM_add_factor = ACM[product_name].attrs["add_offset"] ## Get the add offset
        ACM_scale_factor = ACM[product_name].attrs["scale_factor"] ## Get the scale factor
        ACM_values = ACM[product_name].values[0] * ACM_scale_factor + ACM_add_factor ## Get the values of the active fire pixels
        ACM_fill_value = ACM[product_name].attrs["_FillValue"] ## Get the fill value
        ACM_values = ACM[product_name].values[0] ## Get the values of the band
        ACM_values[ACM_values == ACM_fill_value] = np.nan ## set the fill value to nan
        ACM[product_name].values[0] = ACM_values * ACM_scale_factor + ACM_add_factor ## Get the values of the fire detection confidence
        ACM[product_name] = ACM[product_name].rio.write_crs(GOES_CRS) ## Write the CRS of the band
        return ACM[product_name] ## Return the values of the fire detection confidence
    except: ## If there is an error
        print(f"Error in opening the ACM file: {ACM_path}") ## Print an error message
        return None ## Return None

# %%
def fix_fill_values(cropped_GOES_image):
    """This function gets a cropped GOES image and replace the fill values with nan and return the fixed image

    Args:
        cropped_GOES_image (xr.DataArray): a rioxarray DataArray
    """
    if not isinstance(cropped_GOES_image, xr.DataArray):
        raise ValueError("cropped_GOES_image should be a rioxarray DataArray")
    else:
        GOES_fill_value = cropped_GOES_image.attrs["_FillValue"] ## Get the fill value
        cropped_GOES_image.values[0][cropped_GOES_image.values[0] == GOES_fill_value] = np.nan ## Set the fill value to nan
        return cropped_GOES_image ## Return the fixed image

# %%
def get_VIIRS_bounds_polygon(VIIRS_fire_path):
    """
    Get a VIIRS fire product path return a polygon of VIIRS image bounds

    :VIIRS_fire_path: VIIRS fire product for example "C:\\Asaf\\VIIRS\\VNP14IMG.A2020251.2042.001.2020258064138"
    
    :return: geodataframe of bounds in lat/lon WGS84 projection
    """ 
    VIIRS = xr.open_dataset(VIIRS_fire_path) ## Open file
    ## Now for the bbox bounds
    lat_bounds = VIIRS.attrs['GRingPointLatitude']
    lon_bounds = VIIRS.attrs['GRingPointLongitude']

    pol_bounds = [] ## Empty bounds list
    for i in range(len(lat_bounds)): ## for all bounds
        coord = (lon_bounds[i], lat_bounds[i]) ## Take lon/lat
        pol_bounds.append(coord) ## append
    pol_bounds.append((lon_bounds[0], lat_bounds[0])) ## append firt coords to "close" the polygon

    pol = shapely.Polygon(pol_bounds) ## Create polygon
    gdf_pol = gpd.GeoDataFrame({'geometry': [pol]}, crs=4326) ## Create as gdf

    return(gdf_pol)

# %%
def crop_GOES_using_VIIRS(GOES_path, GOES_band ,VIIRS_path):
    """This function crops the GOES file using the VIIRS file. It returns the cropped GOES file

    Args:
        GOES_path (string): The path of the GOES file. Can be MCMI, FDC, or ACM files. for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc
        GOES_band (string\int): The band of the GOES file. It can be "all" for all MCMI bands or can be 7 for MCMI. For FDC can be "Mask", "Temp", "Power". For ACM can be "Mask", "Temp", "Power". For ACM can be "ACM" or "BCM"
        VIIRS_path (string): The path of the VIIRS file. for example 'F:\\ML_project\\east_us\\VIIRS\\VIIRS_fire\\VNP14IMG.nc'
    """
    CMI_bands = list(range(1,17)) ## The CMI bands
    CMI_bands.append("all") ## Add all to the CMI bands
    FDC_bands = ["Mask", "Temp", "Power"] ## The FDC bands
    ACM_bands = ["ACM", "BCM"] ## The ACM bands
    band_types = CMI_bands + FDC_bands + ACM_bands ## All band types
    file_name = os.path.basename(GOES_path) ## Get the base name of the GOES file
    file_type = file_name.split("-")[2] ## Get the file type of the GOES file
    if file_type not in ["MCMIPC", "FDCC", "ACMC"]: ## If the file type is not MCMIPC, FDCC, or ACMC
        raise ValueError("The GOES file should be either MCMI, FDC, or ACM files") ## Raise an error
    if GOES_band not in band_types:
        raise ValueError("The GOES band should be either CMI for MCMI, Mask, Temp, Power for FDC, and ACM, BCM for ACM")
    
    if (file_type == "MCMIPC") and (GOES_band == "all"): ## If the file type is MCMIPC and the band is all
        GOES_image = rioxarray.open_rasterio(GOES_path) ## Open the MCMI file
        CMI_list = [f"CMI_C{band:02d}" for band in range(1, 17)] ## Get the CMI bands
    elif file_type == "MCMIPC": ## If the file type is MCMIPC
        GOES_image = open_MCMI(MCMI_path=GOES_path, band_number=GOES_band) ## Open the MCMI file
    elif file_type == "FDCC": ## If the file type is FDCC
        GOES_image = open_FDC(FDC_path=GOES_path, product_name=GOES_band) ## Open the FDC file
    elif file_type == "ACMC": ## If the file type is ACMC
        GOES_image = open_ACM(ACM_path=GOES_path, product_name=GOES_band) ## Open the ACM file
        
    try: ## Try to get the VIIRS polygon
        GOES_CRS = GOES_image.rio.crs ## Get the CRS of the GOES file
        VIIRS_polygon = get_VIIRS_bounds_polygon(VIIRS_path) ## Get the VIIRS polygon
    except: ## If there is an error
        print(f"Error in getting the VIIRS polygon for the VIIRS file: {VIIRS_path}") ## Print an error message
        return None
    
    try: ## Try to crop the GOES image
        VIIRS_polygon = VIIRS_polygon.to_crs(GOES_CRS) ## Convert the VIIRS polygon to the CRS of the GOES image
        GOES_cropped = GOES_image.rio.clip(VIIRS_polygon.geometry) ## Clip the GOES image using the VIIRS polygon
        if GOES_band == "all": ##
            GOES_cropped = GOES_cropped.astype("float32") ## Convert the MCMI to float32
            for band in CMI_list: ## For all CMI bands 
                band_add_factor = GOES_cropped[band].attrs["add_offset"] ## Get the add offset
                band_scale_factor = GOES_cropped[band].attrs["scale_factor"] ## Get the scale factor
                band_fill_value = GOES_cropped[band].attrs["_FillValue"] ## Get the fill value
                band_values = GOES_cropped[band].values[0] ## Get the values of the band
                band_values[band_values == band_fill_value] = np.nan ## set the fill value to nan
                GOES_cropped[band].values[0] = band_values * band_scale_factor + band_add_factor ## Get the values of the band
            GOES_cropped = GOES_cropped.rio.write_crs(GOES_CRS) ## Write the CRS of the band
            return GOES_cropped ## Return the cropped GOES image

        else: ## If the band is not all      
            corrected_GOES_cropped = fix_fill_values(GOES_cropped) ## Fix the fill values of the cropped GOES image
            return corrected_GOES_cropped ## Return the cropped GOES image
    except: ## If there is an error
        print(f"Error in cropping the GOES image: {GOES_path}")
        return None
    

# %%
def rasterize_VIIRS(cropped_GOES_image, filter_VIIRS_gdf, rasterize_type="count" ,number_of_VIIRS_points=1, VIIRS_band = None):
    """This function rasterizes the VIIRS points and returns rasterized VIIRS image in the shape of the cropped GOES image

    Args:
        cropped_GOES_image (xr.DataArray): Cropped GOES image in xarray format
        filter_VIIRS_gdf (gpd.GeoDataFrame): filtered VIIRS GeoDataFrame
        rasterize_type (str, optional): count (number of VIIRS in each pixel), max (max VIIRS I04), mean (mean of VIIRS I04). Defaults to ["count"].
        number_of_VIIRS_points (int, optional):Number of VIIRS points above them we rasterize. Defaults to 1 and above. If we set to 2 only 2 and above will be rasterized.
        VIIRS_band ([int], optional): Here we can choose which band to use for our max or mean calculation for example "I4" or "FI" or "I5". Defaults to None if we only want count
    """
    
    if isinstance(filter_VIIRS_gdf, gpd.GeoDataFrame) == False: ## check if the filter_VIIRS_gdf is a GeoDataFrame
        raise ValueError("VIIRS should be a GeoDataFrame")
    if isinstance(cropped_GOES_image, xr.DataArray) == False:
        raise ValueError("cropped_GOES_image should be a xarray DataArray") ## check if the cropped_GOES_image is a xarray DataArray
    if rasterize_type not in ["count", "mean"]:
        raise ValueError("rasterize_type should be count, max or mean") ## check if the rasterize_type is count, max or mean
    if rasterize_type in ["mean"] and VIIRS_band == None:
        raise ValueError("VIIRS_band should be a string")
    if isinstance(number_of_VIIRS_points, int) == False: ## check if the number_of_VIIRS_points is an integer
        raise ValueError("number_of_VIIRS_points should be an integer")
    if number_of_VIIRS_points < 1: ## check if the number_of_VIIRS_points is less than 1
        raise ValueError("number_of_VIIRS_points should be greater or equal to  1")
    if VIIRS_band not in [None, "I4", "I5", "FI"]: ## check if the VIIRS_band is I4, I5 or FI
        raise ValueError("VIIRS_band should be I4, I5 or FI")
    
    VIIRS = filter_VIIRS_gdf.to_crs(cropped_GOES_image.rio.crs) ## convert the VIIRS GeoDataFrame to the same crs as the cropped GOES image
    if VIIRS_band == "I4":
        VIIRS_band_name = "FP_T4"
    elif VIIRS_band == "I5":
        VIIRS_band_name = "FP_T5"
    elif VIIRS_band == "FI":
        VIIRS_band_name = "Fire_index"
    
    raster_shape = cropped_GOES_image.values[0].shape ## get the shape of the cropped GOES image
    raster_values = cropped_GOES_image.values[0]
    transform_matrix = cropped_GOES_image.rio.transform() ## get the transform matrix of the cropped GOES image
    data_type = cropped_GOES_image.values[0].dtype ## get the data type of the cropped GOES image
    fill_value = 0 ## set the fill value to 0
    VIIRS["count"] = 1 ## create a new column called count and set it to 1
    VIIRS_geom_value_count = [(geom,value) for geom, value in zip(VIIRS.geometry, VIIRS["count"])] ## create a list of tuples of the geometry and the value
    ## rasterize the count of the VIIRS points
    rasterized_add = features.rasterize(shapes=VIIRS_geom_value_count,
                                    out_shape=raster_shape,
                                    fill=fill_value,
                                    transform=transform_matrix,
                                    merge_alg=MergeAlg.add,
                                    dtype=data_type) ## rasterize the VIIRS points
    if rasterize_type == "mean": ## check if the rasterize_type is mean
        ## add the values of the band to the rasterized image
        VIIRS_geom_value_band = [(geom,value) for geom, value in zip(VIIRS.geometry, VIIRS[VIIRS_band_name])] ## create a list of tuples of the geometry and the value
        rasterized_replace = features.rasterize(shapes=VIIRS_geom_value_band,
                                    out_shape=raster_shape,
                                    fill=fill_value,
                                    transform=transform_matrix,
                                    merge_alg=MergeAlg.add,
                                    dtype=data_type)
        ## calculate the mean of the pixel values
        rasterized_mean = rasterized_replace / rasterized_add ## calculate the mean of the pixel values
        rasterized_mean[np.isnan(rasterized_mean)] = 0 ## set the nan values to 0
        rasterized_mean[np.isnan(raster_values)] = np.nan ## set where the raster values are nan to nan
        rasterized_mean[rasterized_mean < number_of_VIIRS_points] = 0 ## set the rasterized mean to the number_of_VIIRS_points
        return rasterized_mean ## return the rasterized mean
    elif rasterize_type == "count": ## check if the rasterize_type is count
        rasterized_add[np.isnan(raster_values)] = np.nan ## set where the raster values are nan to nan
        rasterized_add[rasterized_add < number_of_VIIRS_points] = 0 ## set the rasterized add to the number_of_VIIRS_points
        return rasterized_add ## return the rasterized count

# %%
def get_my_neighbores(array, row_i, col_j, distance=1, value_or_index="value"):
    """This function returns the neighbors of a pixel in a raster image

    Args:
        array (numpy array): the raster image
        row_i (int): the row index of the pixel
        col_j (int): the column index of the pixel 
        distance (int, optional): the distance of the neighbors 1 is 3x3 and 2 is 5x5. Defaults to 1.
        value_or_index (str, optional): return the value of the neighbors or the index of them. Defaults to "value".
    """
    if (0>distance) or (distance>2): ## check if the distance is between 0 and 2
        raise ValueError("The distance should be between 0 and 2")
    if value_or_index not in ["value", "index"]: ## check if the value_or_index is either value or index
        raise ValueError("The value_or_index should be either value or index")
    if isinstance(array, np.ndarray) == False: ## check if the array is a numpy array
        raise ValueError("The array should be a numpy array")
    
    array_shape = array.shape ## get the shape of the array
    if (row_i < 0) or (row_i >= array_shape[0]): ## check if the row index is within the range of the array
        raise ValueError("The row index is out of range")
    if (col_j < 0) or (col_j >= array_shape[1]): ## check if the column index is within the range of the array
        raise ValueError("The column index is out of range")
    
    pixel_loc = [row_i, col_j] ## get the location of the pixel
    if distance == 1: ## check if the distance is 1 (3x3)
        neighbors = [[row_i-1, col_j-1],
                     [row_i-1, col_j],
                     [row_i-1, col_j+1],
                     [row_i, col_j-1],
                     [row_i, col_j],
                     [row_i, col_j+1],
                     [row_i+1, col_j-1],
                     [row_i+1, col_j],
                     [row_i+1, col_j+1]]
        remove_list = []
        for i in range(len(neighbors)): ## loop through the neighbors
            ## check if the neighbors are out of the range of the array
            if (neighbors[i][0] < 0) or (neighbors[i][0] >= array_shape[0]) or (neighbors[i][1] < 0) or (neighbors[i][1] >= array_shape[1]):
                remove_list.append(neighbors[i]) ## add the neighbors to the remove_list
                
        for bad_pixel in remove_list: ## loop through the bad pixels
            neighbors.remove(bad_pixel) ## remove the bad pixels from the neighbors
            
        if value_or_index == "value": ## check if the value_or_index is value
            list_of_values = [] ## create an empty list
            for i in range(len(neighbors)):
                pixel_value = array[neighbors[i][0], neighbors[i][1]] ## get the value of the pixel
                list_of_values.append(pixel_value) ## add the value to the list_of_values
            return list_of_values ## return the list_of_values
        elif value_or_index == "index": ## check if the value_or_index is index
            return neighbors
        
    elif distance == 2: ## check if the distance is 2 (5x5)
        neighbors = [[row_i-2, col_j-2],
                     [row_i-2, col_j-1],
                     [row_i-2, col_j],
                     [row_i-2, col_j+1],
                     [row_i-2, col_j+2],
                     [row_i-1, col_j-2],
                     [row_i-1, col_j-1],
                     [row_i-1, col_j],
                     [row_i-1, col_j+1],
                     [row_i-1, col_j+2],
                     [row_i, col_j-2],
                     [row_i, col_j-1],
                     [row_i, col_j],
                     [row_i, col_j+1],
                     [row_i, col_j+2],
                     [row_i+1, col_j-2],
                     [row_i+1, col_j-1],
                     [row_i+1, col_j],
                     [row_i+1, col_j+1],
                     [row_i+1, col_j+2],
                     [row_i+2, col_j-2],
                     [row_i+2, col_j-1],
                     [row_i+2, col_j],
                     [row_i+2, col_j+1],
                     [row_i+2, col_j+2]]
        remove_list = []
        for i in range(len(neighbors)): ## loop through the neighbors
            ## check if the neighbors are out of the range of the array
            if (neighbors[i][0] < 0) or (neighbors[i][0] >= array_shape[0]) or (neighbors[i][1] < 0) or (neighbors[i][1] >= array_shape[1]):
                remove_list.append(neighbors[i])   
        for bad_pixel in remove_list:
            neighbors.remove(bad_pixel)
            
        if value_or_index == "value": ## check if the value_or_index is value
            list_of_values = [] ## create an empty list
            for i in range(len(neighbors)): ## loop through the neighbors
                pixel_value = array[neighbors[i][0], neighbors[i][1]] ## get the value of the pixel
                list_of_values.append(pixel_value) ## add the value to the list_of_values
            return list_of_values ## return the list_of_values
        elif value_or_index == "index": ## check if the value_or_index is index
            return neighbors ## return the neighbors
   

# %%
def VIIRS_locations_to_kill(rasterize_VIIRS):
    """Get rasterized VIIRS and return the locations that can't be used for sampling non-fire pixels

    Args:
        rasterized_VIIRS (array): rasterized VIIRS image
    """
    if isinstance(rasterize_VIIRS, np.ndarray) == False:
        raise ValueError("rasterized_VIIRS should be a numpy array")
    
    VIIRS_locations_to_kill = [] ## list of locations we can't use anymore for sampling
    VIIRS_rows, VIIRS_cols = np.where(rasterize_VIIRS>0) ## get the row and column indices of the rasterize_VIIRS array
    for VIIRS_fp_row, VIIRS_fp_col in zip(VIIRS_rows, VIIRS_cols): ## loop through the row and column indices

    ## get the row and column indices of the 8 neighbors of the current location
        VIIRS_fp_locations = get_my_neighbores(array = rasterize_VIIRS, row_i = VIIRS_fp_row,
                                           col_j = VIIRS_fp_col, distance=1, value_or_index="index")
    
    ## get the values of the 8 neighbors of the current location for the VIIRS array
        for VIIRS_fp_loc in VIIRS_fp_locations: ## loop through the locations of the 8 neighbors
            VIIRS_locations_to_kill.append(VIIRS_fp_loc) ## append the location to the VIIRS_locations
            
    corrected_list = list(VIIRS_locations_to_kill for VIIRS_locations_to_kill,_ in itertools.groupby(VIIRS_locations_to_kill))
    return corrected_list ## return the VIIRS_locations_to_kill

# %%
def get_GOES_actual_fire_pixel_locations(GOES_Fire_Index_array, rasterize_VIIRS):
    """Get the actual fire pixel locations in the GOES image

    Args:
        GOES_Fire_Index_array (array): GOES Fire Index array
        rasterized_VIIRS (array): rasterized VIIRS image
    """
    if isinstance(GOES_Fire_Index_array, np.ndarray) == False:
        raise ValueError("GOES_Fire_Index_array should be a numpy array")
    if isinstance(rasterize_VIIRS, np.ndarray) == False:
        raise ValueError("rasterized_VIIRS should be a numpy array")
    

    GOES_fp_list = [] ## list of GOES fire pixels
    VIIRS_rows, VIIRS_cols = np.where(rasterize_VIIRS>0) ## get the row and column indices of the rasterize_VIIRS array
    for VIIRS_fp_row, VIIRS_fp_col in zip(VIIRS_rows, VIIRS_cols): ## loop through the row and column indices
    ## get the values of the 8 neighbors of the current location for the Fire index array
        GOES_FI_values = get_my_neighbores(array = GOES_Fire_Index_array, row_i = VIIRS_fp_row,
                                        col_j = VIIRS_fp_col, distance=1, value_or_index="value")
    ## get the row and column indices of the 8 neighbors of the current location
        VIIRS_fp_locations = get_my_neighbores(array = GOES_Fire_Index_array, row_i = VIIRS_fp_row,
                                           col_j = VIIRS_fp_col, distance=1, value_or_index="index")
        
        condition_list = [] ## create a list of conditions [True, False, True, ...]
        for loc in VIIRS_fp_locations: ## loop through the VIIRS locations for example [[1,2], [2,3], [3,4]]
            tuple_loc = tuple(loc) ## convert the list to a tuple for example (1,2)
            GOES_fp_set = {tuple(fp) for fp in GOES_fp_list} ## create a set of the GOES fire pixels
            condition_list.append(tuple_loc not in GOES_fp_set) ## append True if the tuple_loc is not in the GOES_fp_set
        
        GOES_FI_values = np.array(GOES_FI_values) ## convert the GOES_FI_values to a numpy array
        VIIRS_fp_locations = np.array(VIIRS_fp_locations) ## convert the VIIRS_fp_locations to a numpy array
        
        GOES_FI_filter = GOES_FI_values[condition_list] ## filter the GOES_FI_values
        VIIRS_fp_filter = VIIRS_fp_locations[condition_list] ## filter the VIIRS_fp_locations
        
        con_len = len(GOES_FI_filter)>0 ## check if the length of the GOES_FI_filter is greater than 0
        con_zero = np.any(GOES_FI_filter!=0) ## check if any of the values of the GOES_FI_filter is not 0
        con_nan = np.any(~np.isnan(GOES_FI_filter)) ## check if any of the values of the GOES_FI_filter is not nan
        if (con_len) & (con_zero) & (con_nan): ## check if the conditions are met
            max_FI_value = np.max(GOES_FI_filter) ## get the maximum value of the 8 neighbors
            max_FI_locations = np.where(GOES_FI_filter == max_FI_value)[0] ## get the indices of the maximum value
    
            for loc in max_FI_locations: ## loop through the indices of the maximum value
                GOES_fp_row = VIIRS_fp_filter[loc][0] ## get the row index of the maximum value
                GOES_fp_col = VIIRS_fp_filter[loc][1] ## get the column index of the maximum value
                GOES_fp_list.append([GOES_fp_row, GOES_fp_col])
            
    corrected_list = list(GOES_fp_list for GOES_fp_list,_ in itertools.groupby(GOES_fp_list)) ## remove duplicates
    return corrected_list ## return the GOES_fp_list

# %%
def GOES_locations_to_kill(GOES_fp_list, GOES_Fire_Index_array):
    """This function get a GOES fire pixel list and a GOES array and return the locations that can't be used for sampling non-fire pixels

    Args:
        GOES_fp_list (list): GOES fire pixel list
        GOES_Fire_Index_array (array): GOES Fire Index array
    """
    if isinstance(GOES_fp_list, list) == False:
        raise ValueError("GOES_fp_list should be a list")
    if isinstance(GOES_Fire_Index_array, np.ndarray) == False:
        raise ValueError("GOES_Fire_Index_array should be a numpy array")
    
    GOES_locations_to_kill = [] ## list of locations we can't use anymore for sampling
    for fp in GOES_fp_list: ## loop through the GOES fire pixel locations
        GOES_row_loc = fp[0] ## get the row location of the GOES fire pixel
        GOES_col_loc = fp[1] ## get the column location of the GOES fire pixel
        GOES_neighbor_locations = get_my_neighbores(array=GOES_Fire_Index_array, row_i=GOES_row_loc,
                                                 col_j=GOES_col_loc, distance=1, value_or_index="index") ## get the neighbors of the GOES fire pixel
        for GOES_neighbor_loc in GOES_neighbor_locations: ## loop through the neighbors of the GOES fire pixel
            GOES_locations_to_kill.append(GOES_neighbor_loc) ## append the neighbors to the GOES_locations_to_kill
    corrected_list = list(GOES_locations_to_kill for GOES_locations_to_kill,_ in itertools.groupby(GOES_locations_to_kill)) ## remove duplicates
    return corrected_list ## return the GOES_locations_to

# %%
def nan_locations_to_kill(GOES_Fire_Index_array):
    """Get a GOES array and return the locations that can't be used for sampling non-fire pixels

    Args:
        GOES_Fire_Index_array (array): GOES Fire Index array
    """
    if isinstance(GOES_Fire_Index_array, np.ndarray) == False:
        raise ValueError("GOES_Fire_Index_array should be a numpy array")
    
    nan_loc_list = [] ## list to store the locations of the nan values in the Fire Index array
    nan_row, nan_col = np.where(np.isnan(GOES_Fire_Index_array))
    for row,col in zip(nan_row, nan_col): ## loop through the nan values locations
        nan_loc_list.append([row,col]) ## append the location to the nan_loc_list
    return nan_loc_list ## return the nan_loc_list

# %%
def get_random_non_fire_pixels(GOES_Fire_Index_array, corrected_kill_list, number_of_non_fire_pixels):
    """This function gets the GOES Fire Index array to get image shape, get the corrected kill (pixels that can't be used for non-fire pixels), and the number of non-fire pixels to sample and return the non-fire pixels 

    Args:
        GOES_Fire_Index_array (array): GOES Fire Index array
        corrected_kill_list (list): list of locations that can't be used for sampling non-fire pixels for example [[1,2], [3,4]]
        number_of_non_fire_pixels (int): number of fire pixels to sample for example 10
    """
    # Get the row and column shapes of the array
    Image_row_shape = GOES_Fire_Index_array.shape[0] - 1
    Image_col_shape = GOES_Fire_Index_array.shape[1] - 1

    # Use a set for fast lookup
    corrected_kill_set = {tuple(loc) for loc in corrected_kill_list}
    random_loc_list = []
    iteration = number_of_non_fire_pixels
    while len(random_loc_list) < iteration:
        # Generate a random location
        random_row = random.randint(0, Image_row_shape)
        random_col = random.randint(0, Image_col_shape)
        random_loc = (random_row, random_col)  # Use tuple for compatibility with set

        # Check if the location is not in the set
        if random_loc not in corrected_kill_set:
            random_loc_list.append(random_loc)
            corrected_kill_set.add(random_loc)  # Add to set for fast lookup

    return random_loc_list

# %%
def remove_cloud_neighbores(band_array, cloud_mask_array, row_i, col_j, distance,cloud_probability_list, statistic):
    """This function gets the band array, cloud mask array, row index, column index, distance, and statistic and return a statistic of the pixel neighbore withot clouds and without the pixel itself

    Args:
        band_array (array): The band array
        cloud_mask_array (array): ACM array
        row_i (int): fire pixel row index
        col_j (int): fire pixel column index
        distance (int): the buffer distance 1 for 3x3 and 2 for 5x5
        cloud_probability_list (list): list of cloud probabilities of ACM to be excluded for example [2,3]
        statistic (string): the statistic to calculate for example "mean". Avilable statistics are "mean", "median, "std, "max", "min"
    """
    if isinstance(band_array, np.ndarray) == False:
        raise ValueError("band_array should be a numpy array")
    if isinstance(cloud_mask_array, np.ndarray) == False:
        raise ValueError("cloud_mask_array should be a numpy array")

    all_clouds = -999
    all_nan = -888
    if statistic == "value": ## check if the statistic is value
        return band_array[row_i, col_j] ## return the value of the pixel
    
    else: ## if the statistic is not value
    
        band_values = get_my_neighbores(array=band_array, row_i=row_i, col_j=col_j, distance=distance, value_or_index="value") ## get the band values
        cloud_values = get_my_neighbores(array=cloud_mask_array, row_i=row_i, col_j=col_j, distance=distance, value_or_index="value") ## get the cloud values
    
        if distance == 1:
            band_values.pop(4) ## remove the center pixel
            cloud_values.pop(4) ## remove the center pixel
        elif distance == 2:
            band_values.pop(12) ## remove the center pixel
            cloud_values.pop(12) ## remove the center pixel
        
        band_values = np.array(band_values) ## convert the band values to a numpy array
        cloud_values = np.array(cloud_values) ## convert the cloud values to a numpy array
    
        is_cloud = np.isin(cloud_values, cloud_probability_list) ## check if the cloud values are in the cloud_probability_list
        filter_band_values = band_values[~is_cloud] ## filter the band values. Take only the values that are not clouds
    
        if len(filter_band_values) == 0: ## check if the filter_band_values is empty
            #print(f"in pixel {row_i}, {col_j} all of the neighbors are clouds") ## print a message
            return all_clouds ## return -999
    
        else: ## if the filter_band_values is not empty
            with warnings.catch_warnings(): ## catch the warnings
                warnings.simplefilter("ignore", category=RuntimeWarning) ## ignore the runtime warnings of nan
                if statistic == "mean": ## check if the statistic is mean
                    mean = np.nanmean(filter_band_values) ## calculate the mean of the filter_band_values
                    if np.isnan(mean): ## check if the mean is nan
                        return all_nan ## return -888
                    else: ## if the mean is not nan
                        return mean ## return the mean
                elif statistic == "median": ## check if the statistic is median
                    median = np.nanmedian(filter_band_values)
                    if np.isnan(median):
                        return all_nan
                    else:
                        return median
                elif statistic == "std": ## check if the statistic is std
                    std = np.nanstd(filter_band_values)
                    if np.isnan(std):
                        return all_nan
                    else:
                        return std
                elif statistic == "max": ## check if the statistic is max
                    max_value = np.nanmax(filter_band_values)
                    if np.isnan(max_value):
                        return all_nan
                    else:
                        return max_value
                elif statistic == "min": ## check if the statistic is min
                    min_value = np.nanmin(filter_band_values)
                    if np.isnan(min_value):
                        return all_nan
                    else:
                        return min_value

# %%
def get_fire_pixel_values_in_all_bands(pixel_location_list, MCMI_path, FDC_path, ACM_path, VIIRS_path, GOES_date_time, rasterize_VIIRS, cloud_probability_list=[2,3]):
    """This function gets the pixel location list and the MCMI and VIIRS paths and return a df with pixel values

    Args:
        pixel_location_list (list): pixel location list for example [[1,2], [3,4]]
        MCMI_path (str): MCMI full path for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'
        FDC_path (str): FDC full path for example 'F:\\ML_project\\GOES_16\\FDC\\OR_ABI-L2-FDCC-M6_G16_s202301010751.nc'
        ACM_path (str): ACM full path for example 'F:\\ML_project\\GOES_16\\ACM\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc'
        VIIRS_path (str): VIIRS image path for example 'F:\\ML_project\\east_us\\VIIRS\\VIIRS_fire\\VNP14IMG.nc'
        GOES_date_time (str): GOES date time for example '2023-01-01 07:51'
        rasterize_VIIRS (array): rasterized VIIRS image
        cloud_probability_list (list): list of cloud probabilities of ACM to be excluded for example [3,4]
    """
    if isinstance(pixel_location_list, list) == False:
        raise ValueError("pixel_location_list should be a list")
    if isinstance(MCMI_path, str) == False:
        raise ValueError("MCMI_path should be a string")
    if isinstance(FDC_path, str) == False:
        raise ValueError("FDC_path should be a string")
    if isinstance(ACM_path, str) == False:
        raise ValueError("ACM_path should be a string")
    if isinstance(VIIRS_path, str) == False:
        raise ValueError("VIIRS_path should be a string")
    if isinstance(GOES_date_time, str) == False:
        raise ValueError("GOES_date_time should be a string")
    if isinstance(rasterize_VIIRS, np.ndarray) == False:
        raise ValueError("rasterize_VIIRS should be a numpy array")
    
    ## Now we will open the VIIRS file for day/night
    try:
        VIIRS_file = xr.open_dataset(VIIRS_path) ## open the MCMI file
        VIIRS_day_night = VIIRS_file.attrs['Day/Night/Both']
        if VIIRS_day_night == "Day": ## check if the VIIRS file is for day
            is_day = 1 ## set the is_day to 1
            is_night = 0 ## set the is_night to 0
            is_day_night = 0 ## set the is_day_night to 0
        elif VIIRS_day_night == "Night": ## check if the VIIRS file is for night
            is_day = 0 ## set the is_day to 0
            is_night = 1 ## set the is_night to 1
            is_day_night = 0 ## set the is_day_night to 0
        elif VIIRS_day_night == "Both": ## check if the VIIRS file is for both day and night
            is_day = 0 ## set the is_day to 0
            is_night = 0 ## set the is_night to 0
            is_day_night = 1 ## set the is_day_night to 1
    except: ## if there is an error
        print(f"Error in opening the VIIRS file: {VIIRS_path}") ## print an error message
        ## set the is_day, is_night, and is_day_night to -999
        is_day = -999
        is_night = -999
        is_day_night = -999
    t = "t0" ## set the time to t0
    ACM = crop_GOES_using_VIIRS(GOES_path=ACM_path, GOES_band="ACM", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
    ACM_values = ACM.values[0] ## get the values of the ACM
    FDC = crop_GOES_using_VIIRS(GOES_path=FDC_path, GOES_band="Mask", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
    FDC_values = FDC.values[0] ## get the values of the FDC
    MCMI = crop_GOES_using_VIIRS(GOES_path=MCMI_path, GOES_band="all", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
    B7_values = MCMI["CMI_C07"].values[0] ## get the values of the band
    B14_values = MCMI["CMI_C14"].values[0] ## get the values of the band
    FI = (B7_values - B14_values) / (B7_values + B14_values) ## calculate the fire index
    band_list = [f"CMI_C{band:02d}" for band in range(1, 17)] ## Get the CMI bands
    indices_list = ["FI"] ## list of indices for example fire index (FI)
    band_iteration_list = band_list + indices_list ## combine the band_list and indices_list
    statistics_list = ["value", "mean", "median", "std","min","max"] ## list of statistics for example value, mean, median, std
    
    row_list = [] ## list to store the row values
    col_list = [] ## list to store the column values
    ACM_list = [] ## list to store the ACM values
    FDC_list = [] ## list to store the FDC values
    VIIRS_max_list = [] ## list to store the VIIRS max value
    VIIRS_sum_list = [] ## list to store the VIIRS sum value
    
    for loc in pixel_location_list: ## loop through the pixel location list
        row = loc[0] ## get the row location
        col = loc[1] ## get the column location
        row_list.append(row) ## append the row to the row_list
        col_list.append(col) ## append the column to the col_list
        ACM_value = ACM_values[row, col] ## get the value of the ACM
        FDC_value = FDC_values[row, col] ## get the value of the FDC
        ACM_list.append(ACM_value) ## append the ACM value to the ACM_list
        FDC_list.append(FDC_value) ## append the FDC value to the FDC_list
        ## get the neighbors of the pixel including the pixel itself in VIIRS rasterized image
        VIIRS_n = get_my_neighbores(array=rasterize_VIIRS, row_i=row, col_j=col, distance=1, value_or_index="value")
        max_VIIRS_n = np.nanmax(VIIRS_n) ## get the max value of the neighbors
        sum_VIIRS_n = np.nansum(VIIRS_n) ## get the sum value of the neighbors
        VIIRS_max_list.append(max_VIIRS_n) ## append the max value to the VIIRS_values_list
        VIIRS_sum_list.append(sum_VIIRS_n) ## append the sum value to the VIIRS_values_list
        
    d = {} ## dictionary to store the values
    d["row"] = row_list ## add the row_list to the dictionary
    d["col"] = col_list ## add the col_list to the dictionary
    d["VIIRS_fp_max"] = VIIRS_max_list ## add the VIIRS_values_list to the dictionary
    d["VIIRS_fp_sum"] = VIIRS_sum_list ## add the VIIRS_values_list to the dictionary
    for band in band_iteration_list: ## loop through the band_iteration_list
        if band == "FI": ## check if the band is FI
            FI_value_list = [] ## list to store the fire index values
            FI_n_mean_list = [] ## list to store the fire index mean values
            FI_n_median_list = [] ## list to store the fire index median values
            FI_n_std_list = [] ## list to store the fire index std values
            FI_n_min_list = [] ## list to store the fire index min values
            FI_n_max_list = [] ## list to store the fire index max values
            for loc in pixel_location_list: ## loop through the pixel location list
                row = loc[0] ## get the row location
                col = loc[1] ## get the column location
                for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                    stat_value = remove_cloud_neighbores(band_array=FI, 
                                                              cloud_mask_array=ACM_values,
                                                              row_i=row,
                                                              col_j=col,
                                                              distance=1,
                                                              cloud_probability_list=cloud_probability_list,
                                                              statistic=stat) ## get the neighbors of the pixel
                    if stat == "value":
                        FI_value_list.append(stat_value) ## append the value to the FI_value_list
                    elif stat == "mean": ## check if the stat is mean
                        FI_n_mean_list.append(stat_value) ## append the mean to the FI_n_mean_list
                    elif stat == "median": ## check if the stat is median
                        FI_n_median_list.append(stat_value) ## append the median to the FI_n_median_list
                    elif stat == "std": ## check if the stat is std
                        FI_n_std_list.append(stat_value) ## append the std to the FI_n_std_list
                    elif stat == "min": ## check if the stat is min
                        FI_n_min_list.append(stat_value) ## append the min to the FI_n_min_list
                    elif stat == "max": ## check if the stat is max
                        FI_n_max_list.append(stat_value) ## append the max to the FI_n_max_list
            d[f"{t}_FI_value"] = FI_value_list ## add the FI_value_list to the dictionary
            d[f"{t}_tFI_mean"] = FI_n_mean_list ## add the FI_n_mean_list to the dictionary
            d[f"{t}_FI_median"] = FI_n_median_list ## add the FI_n_median_list to the dictionary
            d[f"{t}_FI_std"] = FI_n_std_list ## add the FI_n_std_list to the dictionary
            d[f"{t}_FI_min"] = FI_n_min_list ## add the FI_n_min_list to the dictionary
            d[f"{t}_FI_max"] = FI_n_max_list ## add the FI_n_max_list to the dictionary
        else: ## if the band is not FI
            band_number = f'B{band.split("_")[-1][1:]}' ## get the band number for example B01
            ## crop the GOES image using the VIIRS image
            B = MCMI[band] ## get the band
            band_array = B.values[0] ## get the values of the band
            band_value_list = [] ## list to store the band values
            band_n_mean_list = [] ## list to store the band mean values
            band_n_median_list = [] ## list to store the band median values
            band_n_std_list = [] ## list to store the band std values
            band_n_min_list = [] ## list to store the band min values
            band_n_max_list = [] ## list to store the band max values
            for loc in pixel_location_list: ## loop through the pixel location list
                row = loc[0] ## get the row location
                col = loc[1] ## get the column location
                for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                    stat_value = remove_cloud_neighbores(band_array=band_array, 
                                                              cloud_mask_array=ACM_values,
                                                              row_i=row,
                                                              col_j=col,
                                                              distance=1,
                                                              cloud_probability_list=cloud_probability_list,
                                                              statistic=stat) ## get the neighbors of the pixel
                    if stat == "value": ## check if the stat is value
                        band_value_list.append(stat_value) ## append the value to the band_value_list
                    elif stat == "mean": ## check if the stat is mean
                        band_n_mean_list.append(stat_value) ## append the mean to the band_n_mean_list
                    elif stat == "median": ## check if the stat is median
                        band_n_median_list.append(stat_value) ## append the median to the band_n_median_list
                    elif stat == "std": ## check if the stat is std
                        band_n_std_list.append(stat_value) ## append the std to the band_n_std_list
                    elif stat == "min": ## check if the stat is min
                        band_n_min_list.append(stat_value)
                    elif stat == "max": ## check if the stat is max
                        band_n_max_list.append(stat_value)
            d[f"{t}_{band_number}_value"] = band_value_list ## add the band_value_list to the dictionary
            d[f"{t}_{band_number}_mean"] = band_n_mean_list ## add the band_n_mean_list to the dictionary
            d[f"{t}_{band_number}_median"] = band_n_median_list ## add the band_n_median_list to the dictionary
            d[f"{t}_{band_number}_std"] = band_n_std_list ## add the band_n_std_list to the dictionary
            d[f"{t}_{band_number}_min"] = band_n_min_list ## add the band_n_min_list to the dictionary
            d[f"{t}_{band_number}_max"] = band_n_max_list ## add the band_n_max_list to the dictionary
        
    df = pd.DataFrame(d) ## create a DataFrame from the dictionary
    df[f"{t}_FDC_value"] = FDC_list ## add the FDC_list to the DataFrame
    df[f"{t}_ACM_value"] = ACM_list ## add the ACM_list to the DataFrame
    day_list = np.repeat(is_day, len(df)) ## repeat the is_day for the length of the DataFrame
    night_list = np.repeat(is_night, len(df)) ## repeat the is_night for the length of the DataFrame
    day_night_list = np.repeat(is_day_night, len(df)) ## repeat the is_day_night for the length of the DataFrame
    df["is_day"] = day_list ## add the is_day to the DataFrame
    df["is_night"] = night_list ## add the is_night to the DataFrame
    df["is_day_night"] = day_night_list ## add the is_day_night to the DataFrame
    file_name = os.path.basename(MCMI_path).split("_")[-1] ## get the base name of the MCMI file 
    file_name_list = np.repeat(file_name, len(df)) ## repeat the file name for the length of the DataFrame
    date_time_list = np.repeat(GOES_date_time, len(df)) ## repeat the date time for the length of the DataFrame
    df.insert(0, f"{t}_MCMI_file", file_name_list) ## insert the file name to the first column
    df.insert(1, f"{t}_GOES_date_time", date_time_list) ## insert the date time to the second column
    return df ## return the DataFrame

# %%
def get_temporal_fire_pixel_values_in_all_bands(temporal_df, pixel_location_list, VIIRS_path ,GOES_date_time ,temporal_images=4, cloud_probability_list=[3,4]):
    """This function gets the pixel location list and the MCMI and VIIRS paths and return a df with pixel values

    Args:
        temporal_df (DataFrame): DataFrame with the temporal values
        pixel_location_list (list): pixel location list for example [[1,2], [3,4]]
        VIIRS_path (str): VIIRS image path for example 'F:\\ML_project\\east_us\\VIIRS\\VIIRS_fire\\VNP14IMG.nc'
        GOES_date_time (str): GOES date time for example '2023-01-01 07:51'
        temporal_images (int, optional): number of temporal images to get. Defaults to 4. It should be between 1 and 4
        cloud_probability_list (list, optional): list of cloud probabilities of ACM to be excluded for example [3,4]. Defaults to [3,4].
    """
    
    if isinstance(temporal_df, pd.DataFrame) == False:
        raise ValueError("temporal_df should be a DataFrame")
    if isinstance(pixel_location_list, list) == False:
        raise ValueError("pixel_location_list should be a list")
    if isinstance(VIIRS_path, str) == False:
        raise ValueError("VIIRS_path should be a string")
    if temporal_images not in [1,2,3,4]:
        raise ValueError("temporal should be between 1 and 4")
    
    filter_temporal_df = temporal_df[temporal_df["GOES_date_time"] == GOES_date_time] ## filter the temporal_df for the GOES_date_time
    band_list = list(range(1,17)) ## list of the MCMI bands
    CMI_list = [f"CMI_C{band:02d}" for band in band_list] ## list of the CMI bands
    indices_list = ["FI"] ## list of indices for example fire index (FI)
    band_iteration_list = CMI_list + indices_list ## combine the band_list and indices_list
    statistics_list = ["value", "mean", "median", "std", "min", "max"] ## list of statistics for example value, mean, median, std
    df_list = [] ## list to store the DataFrames
    for t in range(temporal_images):
        full_file_name = os.path.basename(filter_temporal_df["MCMI"].iloc[t]) ## get the base name of the MCMI file
        file_date_name = full_file_name.split("_")[-1] ## get the file name
        file_date = file_date_name.split(".")[0][1:] ## get the date of the file
        Year = file_date[:4] ## get the year of the file
        Month = file_date[4:6] ## get the month of the file
        Day = file_date[6:8] ## get the day of the file
        Hour = file_date[8:10] ## get the hour of the file
        Minute = file_date[10:12]
        string_date = f"{Year}-{Month}-{Day} {Hour}:{Minute}" ## create a string date
        ACM_path = filter_temporal_df["ACM"].iloc[t] ## get the ACM path
        FDC_path = filter_temporal_df["FDC"].iloc[t] ## get the FDC path
        MCMI_path = filter_temporal_df["MCMI"].iloc[t] ## get the MCMI path
        ACM = crop_GOES_using_VIIRS(GOES_path=ACM_path, GOES_band="ACM", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
        ACM_values = ACM.values[0] ## get the values of the ACM
        FDC = crop_GOES_using_VIIRS(GOES_path=FDC_path, GOES_band="Mask", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
        FDC_values = FDC.values[0] ## get the values of the FDC
        MCMI = crop_GOES_using_VIIRS(GOES_path=MCMI_path, GOES_band="all", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
        B7_values = MCMI["CMI_C07"].values[0] ## get the values of the band
        B14_values = MCMI["CMI_C14"].values[0] ## get the values of the band
        FI = (B7_values - B14_values) / (B7_values + B14_values) ## calculate the fire index
        ACM_list = [] ## list to store the ACM values
        FDC_list = [] ## list to store the FDC values
        d = {} ## dictionary to store the values
        for loc in pixel_location_list: ## loop through the pixel location list
            row = loc[0] ## get the row location
            col = loc[1] ## get the column location
            ACM_value = ACM_values[row, col] ## get the value of the ACM
            FDC_value = FDC_values[row, col] ## get the value of the FDC
            ACM_list.append(ACM_value) ## append the ACM value to the ACM_list
            FDC_list.append(FDC_value) ## append the FDC value to the FDC_list
        for band in band_iteration_list: ## loop through the band_iteration_list
            if band == "FI": ## check if the band is FI
                FI_value_list = [] ## list to store the fire index values
                FI_n_mean_list = [] ## list to store the fire index mean values
                FI_n_median_list = [] ## list to store the fire index median values
                FI_n_std_list = [] ## list to store the fire index std values
                FI_n_min_list = [] ## list to store the fire index min values
                FI_n_max_list = [] ## list to store the fire index max values
                for loc in pixel_location_list: ## loop through the pixel location list
                    row = loc[0] ## get the row location
                    col = loc[1] ## get the column location
                    for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                        stat_value = remove_cloud_neighbores(band_array=FI,
                                                             cloud_mask_array=ACM_values,
                                                             row_i=row,
                                                             col_j=col,
                                                             distance=1,
                                                             cloud_probability_list=cloud_probability_list,
                                                             statistic=stat)
                        if stat == "value":
                            FI_value_list.append(stat_value) ## append the value to the FI_value_list
                        elif stat == "mean": ## check if the stat is mean
                            FI_n_mean_list.append(stat_value) ## append the mean to the FI_n_mean_list
                        elif stat == "median": ## check if the stat is median
                            FI_n_median_list.append(stat_value) ## append the median to the FI_n_median_list
                        elif stat == "std": ## check if the stat is std
                            FI_n_std_list.append(stat_value) ## append the std to the FI_n_std_list
                        elif stat == "min": ## check if the stat is min
                            FI_n_min_list.append(stat_value) ## append the min to the FI_n_min_list
                        elif stat == "max": ## check if the stat is max
                            FI_n_max_list.append(stat_value) ## append the max to the FI_n_max_list
                d[f"t{t+1}_FI_value"] = FI_value_list ## add the FI_value_list to the dictionary
                d[f"t{t+1}_FI_mean"] = FI_n_mean_list ## add the FI_n_mean_list to the dictionary
                d[f"t{t+1}_FI_median"] = FI_n_median_list ## add the FI_n_median_list to the dictionary
                d[f"t{t+1}_FI_std"] = FI_n_std_list ## add the FI_n_std_list to the dictionary
                d[f"t{t+1}_FI_min"] = FI_n_min_list ## add the FI_n_min_list to the dictionary
                d[f"t{t+1}_FI_max"] = FI_n_max_list ## add the FI_n_max_list to the dictionary
            else: ## if the band is not FI
                band_number = f'B{band.split("_")[-1][1:]}' ## get the band number for example B01
                    ## crop the GOES image using the VIIRS image
                B = MCMI[band] ## get the band
                band_array = B.values[0] ## get the values of the band
                band_value_list = [] ## list to store the band values
                band_n_mean_list = [] ## list to store the band mean values
                band_n_median_list = [] ## list to store the band median values
                band_n_std_list = [] ## list to store the band std values
                band_n_min_list = [] ## list to store the band min values
                band_n_max_list = [] ## list to store the band max values
                for loc in pixel_location_list: ## loop through the pixel location list
                    row = loc[0] ## get the row location
                    col = loc[1] ## get the column location
                    for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                        stat_value = remove_cloud_neighbores(band_array=band_array,
                                                             cloud_mask_array=ACM_values,
                                                             row_i=row,
                                                             col_j=col,
                                                             distance=1,
                                                             cloud_probability_list=cloud_probability_list,
                                                             statistic=stat)
                        if stat == "value": ## check if the stat is value
                            band_value_list.append(stat_value) ## append the value to the band_value_list
                        elif stat == "mean": ## check if the stat is mean
                            band_n_mean_list.append(stat_value) ## append the mean to the band_n_mean_list
                        elif stat == "median": ## check if the stat is median
                            band_n_median_list.append(stat_value) ## append the median to the band_n_median_list
                        elif stat == "std": ## check if the stat is std
                            band_n_std_list.append(stat_value) ## append the std to the band_n_std_list
                        elif stat == "min": ## check if the stat is min
                            band_n_min_list.append(stat_value) ## append the min to the band_n_min_list
                        elif stat == "max": ## check if the stat is max
                            band_n_max_list.append(stat_value)
                d[f"t{t+1}_{band_number}_value"] = band_value_list ## add the band_value_list to the dictionary
                d[f"t{t+1}_{band_number}_mean"] = band_n_mean_list ## add the band_n_mean_list to the dictionary
                d[f"t{t+1}_{band_number}_median"] = band_n_median_list ## add the band_n_median_list to the dictionary
                d[f"t{t+1}_{band_number}_std"] = band_n_std_list ## add the band_n_std_list to the dictionary
                d[f"t{t+1}_{band_number}_min"] = band_n_min_list
                d[f"t{t+1}_{band_number}_max"] = band_n_max_list
        
        df = pd.DataFrame(d) ## create a DataFrame from the dictionary
        df[f"t{t+1}_FDC_value"] = FDC_list ## add the FDC_list to the DataFrame
        df[f"t{t+1}_ACM_value"] = ACM_list ## add the ACM_list to the DataFrame
        file_name_list = np.repeat(file_date_name, len(df)) ## repeat the file name for the length of the DataFrame
        date_time_list = np.repeat(string_date, len(df)) ## repeat the date time for the length of the DataFrame
        df.insert(0, f"t{t+1}_MCMI_file", file_name_list) ## insert the file name to the first column
        df.insert(1, f"t{t+1}_GOES_date_time", date_time_list) ## insert the date time to the second column
        df_list.append(df) ## append the DataFrame to the df_list
    if temporal_images == 1:
        return df_list[0] ## return the first DataFrame
    else:
        return pd.concat(df_list, axis=1) ## return the concatenated DataFrames

# %%
def create_ML_training_df(MCMI_path, FDC_path,
                          ACM_path ,VIIRS_path,
                          filter_VIIRS,
                          GOES_date_time,
                          temporal_df, VIIRS_threshold=1,
                          number_of_temporal_GOES_images=4, 
                          number_of_non_fire_pixels=500,
                          cloud_probability_list=[2,3]):
    """This function gets the MCMI, FDC, ACM, VIIRS paths and the temporal_df and return the ML training DataFrame

    Args:
        MCMI_path (str): path to the MCMI file for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'
        FDC_path (str):path to the FDC file for example 'F:\\ML_project\\GOES_16\\FDC\\OR_ABI-L2-FDCC-M6_G16_s202301010751.nc'
        ACM_path (str): path of the ACM file for example 'F:\\ML_project\\GOES_16\\ACM\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc'
        VIIRS_path (str): path of VIIRS image for example 'F:\\ML_project\\east_us\\VIIRS\\VIIRS_fire\\VNP14IMG.nc'
        filter_VIIRS (GeoDataFrame): a GeoDataFrame with the VIIRS fire pixels
        GOES_date_time (str): GOES date time for example '2023-01-01 07:51'
        temporal_df (DataFrame): GOES temporal DataFrame
        VIIRS_threshold (int): VIIRS threshold for rasterazing for example 1
        number_of_temporal_GOES_images (int): number of temporal GOES. needs to be between 0 and 4
        number_of_non_fire_pixels (int): number of random non-fire pixels to sample for each image. for example 10
        cloud_probability_list (list): list of cloud probabilities of ACM to be excluded for example [3,4]
    """
    ## check the input types
    if isinstance(MCMI_path, str) == False:
        raise ValueError("MCMI_path should be a string")
    if isinstance(FDC_path, str) == False:
        raise ValueError("FDC_path should be a string")
    if isinstance(ACM_path, str) == False:
        raise ValueError("ACM_path should be a string")
    if isinstance(VIIRS_path, str) == False:
        raise ValueError("VIIRS_path should be a string")
    if isinstance(GOES_date_time, str) == False:
        raise ValueError("GOES_date_time should be a string")
    if isinstance(temporal_df, pd.DataFrame) == False:
        raise ValueError("temporal_df should be a DataFrame")
    if VIIRS_threshold < 0:
        raise ValueError("VIIRS_threshold should be greater than 0")
    if number_of_temporal_GOES_images not in [0,1,2,3,4]:
        raise ValueError("number_of_temporal_GOES_images should be between 1 and 4")
    if number_of_non_fire_pixels < 0:
        raise ValueError("number_of_non_fire_pixels should be greater than 0")
    if isinstance(filter_VIIRS, gpd.GeoDataFrame) == False:
        raise ValueError("filter_VIIRS should be a GeoDataFrame")
    
    
    print(f"Now working of GOES time stamp: {GOES_date_time}") ## print the GOES date time
    ## Open GOES bands
    MCMI = crop_GOES_using_VIIRS(GOES_path=MCMI_path, GOES_band="all", VIIRS_path=VIIRS_path) ## crop the GOES image using the VIIRS image
    B7 = MCMI["CMI_C07"] ## get the band 7
    B14 = MCMI["CMI_C14"] ## get the band 14
    FI = (B7.values[0] - B14.values[0])/(B7.values[0] + B14.values[0]) ## calculate the fire index
    
    ## Rasterize VIIRS image
    rasterized_VIIRS_image = rasterize_VIIRS(cropped_GOES_image=B7, filter_VIIRS_gdf=filter_VIIRS,
                                      rasterize_type="count", number_of_VIIRS_points=VIIRS_threshold, VIIRS_band = None)
    if np.any(rasterized_VIIRS_image>0) == False: ## check if the rasterized VIIRS image has no fire pixels
        print(f"No VIIRS fire pixels for GOES time stamp: {GOES_date_time}") ## print a message
        raise ValueError("No VIIRS fire pixels") ## raise an error
    else: ## if there are fire pixels
        print(f"rasterize VIIRS is done for GOES time stamp: {GOES_date_time}") ## print a message
    
    GOES_fp_pixel_list = get_GOES_actual_fire_pixel_locations(GOES_Fire_Index_array=FI, rasterize_VIIRS=rasterized_VIIRS_image) ## get the GOES fire pixel list
    print(f"GOES fire pixel list is done for GOES time stamp: {GOES_date_time}") ## print a message
    
    VIIRS_to_kill = VIIRS_locations_to_kill(rasterize_VIIRS=rasterized_VIIRS_image) ## get the VIIRS locations to kill
    GOES_to_kill = GOES_locations_to_kill(GOES_fp_list=GOES_fp_pixel_list,GOES_Fire_Index_array=FI) ## get the GOES locations to kill
    nan_kill_list = nan_locations_to_kill(GOES_Fire_Index_array=FI) ## get the nan locations to kill
    kill_list = GOES_to_kill + VIIRS_to_kill + nan_kill_list ## combine the GOES_to_kill, VIIRS_to_kill and nan_loc_list
    corrected_kill  = list(kill_list for kill_list,_ in itertools.groupby(kill_list)) ## remove duplicates
    print(f"list of locations not to sample is done for GOES time stamp: {GOES_date_time}") ## print a message
    
    ## Get the random non-fire pixels
    print(f"Starting to genrate {number_of_non_fire_pixels} random non-fire pixels")
    non_fire_pixels = get_random_non_fire_pixels(GOES_Fire_Index_array=FI,
                                                 number_of_non_fire_pixels=number_of_non_fire_pixels,
                                                 corrected_kill_list=corrected_kill)
    print(f"Genrated {number_of_non_fire_pixels} random non-fire pixels for GOES time stamp: {GOES_date_time}") ## print a message
    
    print(f"staring to genrate fire pixel values for GOES time stamp: {GOES_date_time}")
    ## Get the fire pixel values
    if number_of_temporal_GOES_images == 0: ## If We only use the current GOES image
        print(f"Starting to get the fire pixel values for GOES time stamp: {GOES_date_time}")
        df_fire_pixels = get_fire_pixel_values_in_all_bands(pixel_location_list=GOES_fp_pixel_list,
                                            MCMI_path=MCMI_path,
                                            FDC_path=FDC_path,
                                            ACM_path=ACM_path,
                                            VIIRS_path=VIIRS_path,
                                            GOES_date_time=GOES_date_time,
                                            rasterize_VIIRS=rasterized_VIIRS_image,
                                            cloud_probability_list=cloud_probability_list)
        df_fire_pixels["fire_label"] = 1 ## add a column called "fire_label" and set it to 1
        print(f"done. Now starting working on the non-fire pixels")
        df_non_fire_pixels = get_fire_pixel_values_in_all_bands(pixel_location_list=non_fire_pixels,
                                            MCMI_path=MCMI_path,
                                            FDC_path=FDC_path,
                                            ACM_path=ACM_path,
                                            VIIRS_path=VIIRS_path,
                                            GOES_date_time=GOES_date_time,
                                            rasterize_VIIRS=rasterized_VIIRS_image,
                                            cloud_probability_list=cloud_probability_list)
        df_non_fire_pixels["fire_label"] = 0 ## add a column called "fire_label" and set it to 0
        print(f"done. df is ready for GOES time stamp: {GOES_date_time}")
        train_df = pd.concat([df_fire_pixels, df_non_fire_pixels]).reset_index(drop=True) ## concatenate the fire and non-fire DataFrames
        return train_df ## return the train_df
        
    else: ## If we use the temporal GOES images
        print(f"Starting to get the fire pixel values for GOES time stamp: {GOES_date_time}")   
        df_fire_pixels = get_fire_pixel_values_in_all_bands(pixel_location_list=GOES_fp_pixel_list,
                                            MCMI_path=MCMI_path,
                                            FDC_path=FDC_path,
                                            ACM_path=ACM_path,
                                            VIIRS_path=VIIRS_path,
                                            GOES_date_time=GOES_date_time,
                                            rasterize_VIIRS=rasterized_VIIRS_image,
                                            cloud_probability_list=cloud_probability_list)
        print(f"done. Now starting working on the temporal data")
        ## Get the temporal fire pixel values
        df_temporal_fire_pixels = get_temporal_fire_pixel_values_in_all_bands(temporal_df=temporal_df,
                                                                          pixel_location_list=GOES_fp_pixel_list,
                                                                          VIIRS_path=VIIRS_path,
                                                                          GOES_date_time=GOES_date_time,
                                                                          temporal_images=number_of_temporal_GOES_images,
                                                                          cloud_probability_list=cloud_probability_list)
        ## concatenate the fire pixel values and the temporal fire pixel values
        df_fire_pixels_concat_temporal = pd.concat([df_fire_pixels, df_temporal_fire_pixels], axis=1)
        ## add the fire label to the DataFrame
        df_fire_pixels_concat_temporal["fire_label"] = 1
    
        print(f"done. Now starting working on the non-fire pixels")
    
        df_non_fire_pixels = get_fire_pixel_values_in_all_bands(pixel_location_list=non_fire_pixels,
                                            MCMI_path=MCMI_path,
                                            FDC_path=FDC_path,
                                            ACM_path=ACM_path,
                                            VIIRS_path=VIIRS_path,
                                            GOES_date_time=GOES_date_time,
                                            rasterize_VIIRS=rasterized_VIIRS_image,
                                            cloud_probability_list=cloud_probability_list)
        print(f"done. Now starting working on the temporal data")
        df_temporal_non_fire_pixels = get_temporal_fire_pixel_values_in_all_bands(temporal_df=temporal_df,
                                                                          pixel_location_list=non_fire_pixels,
                                                                          VIIRS_path=VIIRS_path,
                                                                          GOES_date_time=GOES_date_time,
                                                                          temporal_images=number_of_temporal_GOES_images,
                                                                          cloud_probability_list=cloud_probability_list)
        ## concatenate the non-fire pixel values and the temporal non-fire pixel values
        df_non_fire_pixels_concat_temporal = pd.concat([df_non_fire_pixels, df_temporal_non_fire_pixels], axis=1)
        ## add the non-fire label to the DataFrame
        df_non_fire_pixels_concat_temporal["fire_label"] = 0 ## add a column called "fire_label" and set it to 0
    
        print(f"done. df is ready for GOES time stamp: {GOES_date_time}")
        train_df = pd.concat([df_fire_pixels_concat_temporal, df_non_fire_pixels_concat_temporal]).reset_index(drop=True) ## concatenate the fire and non-fire DataFrames
        return train_df ## return the train_df
    

# %%



