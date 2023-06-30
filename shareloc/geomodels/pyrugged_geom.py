#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Shareloc
# (see https://github.com/CNES/shareloc).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains the Pyrugged_geom class corresponding to the Pyrugged models.

"""
# pylint: disable=no-member

# Standard imports
import logging
import os
from xml.dom import minidom
import json

# Third party imports
import numpy as np
import rasterio as rio
from numba import config, njit, prange

# Shareloc imports
from shareloc.location_pydimap import Location
from pydimaprugged.compute.localization import direct_loc as direct_loc_pydimap
from pydimaprugged.compute.localization import inverse_loc as inverse_loc_pydimap

# Pyrugged import
# from pyrugged.configuration.init_orekit import init_orekit
# from pyrugged.location.optical import OpticalLocation

# Pydimaprugged imports
#from pydimaprugged.compute.grid_computation import COLOCATION, DIRECT, DIRECT_MULTI_ALT, INVERSE, compute_grid

# Set numba type of threading layer before parallel target compilation
config.THREADING_LAYER = "omp"



class Pyrugged_geom:
    """
    Pyrugged_geom class inspired by RPC class
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, model, light_time=None,aberration_light=None,atmospheric_refraction=None,transforms=None):
        self.epsg = None
        self.datum = None
        self.dimap = model

        self.type = "pyrugged_geom"
        if self.epsg is None:
            self.epsg = 4326
        if self.datum is None:
            self.datum = "ellipsoid"


        xmldoc = minidom.parse(self.dimap)
        mtd = xmldoc.getElementsByTagName("Dataset_Content")
        mtd_format = mtd[0].getElementsByTagName("Component")[2]
        diap_rpc = mtd_format.getElementsByTagName("COMPONENT_PATH")[0].attributes.items()[0][1]

        
        dir_dimap = os.path.dirname(self.dimap)
        xmldoc = minidom.parse(os.path.join(dir_dimap,diap_rpc))
        global_rfm = xmldoc.getElementsByTagName("Global_RFM")[0]
        normalisation_coeffs = global_rfm.getElementsByTagName("RFM_Validity")[0]

        #assuming it is top left convention
        self.offset_col = float(normalisation_coeffs.getElementsByTagName("SAMP_OFF")[0].firstChild.data)-0.5
        self.scale_col = float(normalisation_coeffs.getElementsByTagName("SAMP_SCALE")[0].firstChild.data)
        self.offset_row  = float(normalisation_coeffs.getElementsByTagName("LINE_OFF")[0].firstChild.data)-0.5
        self.scale_row = float(normalisation_coeffs.getElementsByTagName("LINE_SCALE")[0].firstChild.data)
        self.offset_alt = float(normalisation_coeffs.getElementsByTagName("HEIGHT_OFF")[0].firstChild.data)
        self.scale_alt = float(normalisation_coeffs.getElementsByTagName("HEIGHT_SCALE")[0].firstChild.data)

        self.col0 = self.offset_col - self.scale_col
        self.colmax = self.offset_col + self.scale_col
        self.row0 = self.offset_row - self.scale_row
        self.rowmax = self.offset_row + self.scale_row
        self.light_time = light_time
        self.aberration_light = aberration_light
        self.atmospheric_refraction = atmospheric_refraction
        self.transforms = transforms

 

    def direct_loc(self, row, col, alt:list = None, location_dimap:Location = None):
        """
        direct localization at constant altitude

        :param row:  line sensor position
        :type row: float or 1D numpy.ndarray dtype=float64
        :param col:  column sensor position
        :type col: float or 1D numpy.ndarray dtype=float64
        :param location_dimap: Location object from location_pydimap.py used to initialize pyrugged
        :type location_dimap: Location
        :return: ground position (lon,lat,h)
        :rtype: numpy.ndarray 2D dimension with (N,3) shape, where N is number of input coordinates
        """

        if location_dimap is None:
            raise Exception("location_dimap is None cannot perfom direct_loc")
        
        location = location_dimap.location
        parser = location_dimap.parser
        sensor_name = location_dimap.sensor_name
        lon,lat,alt = direct_loc_pydimap(
            location=location,
            parser=parser,
            sensor_name=sensor_name,
            lines = np.array(row),
            pixels = np.array(col),
            altitudes = alt,
        )
        return np.array([lon,lat,alt]).transpose()

    
    def inverse_loc(self,location_dimap:Location, lon, lat, alt=None):
        """
        Inverse localization

        :param location_dimap: Location object from location_pydimap.py used to initialize pyrugged
        :type location_dimap: Location
        :param lon: longitude position
        :type lon: float or 1D numpy.ndarray dtype=float64
        :param lat: latitude position
        :type lat: float or 1D numpy.ndarray dtype=float64
        :param alt: altitude
        :type alt: float
        :return: sensor position (row, col, alt)
        :rtype: tuple(1D np.array row position, 1D np.array col position, 1D np.array alt)
        """
        parser = location_dimap.parser
        location = location_dimap.location
        sensor_name = location_dimap.sensor_name

        if alt is not None:
            altitudes=np.array(alt)
        else:
            altitudes=alt

        lines, pixels=inverse_loc_pydimap(
            location=location,
            sensor_name=sensor_name,
            parser=parser,
            longitudes=np.array(lon),
            latitudes=np.array(lat),
            altitudes=altitudes,
        )

        lines = np.array(lines)
        pixels = np.array(pixels)

        try:
            lines = np.squeeze(lines,axis=0)
            pixels = np.squeeze(pixels,axis=0)
        except:
            lines = np.squeeze(lines, axis=1)
            pixels = np.squeeze(pixels, axis=1)

        return (lines, pixels, alt)

    def get_alt_min_max(self):
        """
        returns altitudes min and max layers

        :return: alt_min,lat_max
        :rtype: list
        """
        return [self.offset_alt - self.scale_alt / 2.0, self.offset_alt + self.scale_alt / 2.0]

  
