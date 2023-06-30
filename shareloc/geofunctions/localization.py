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
Localization class for localization functions.
"""

# Standard imports
import logging
import numbers

# Third party imports
import numpy as np

# Shareloc imports
from pyrugged.configuration.init_orekit import init_orekit
from shareloc.proj_utils import coordinates_conversion
from shareloc.location_pydimap import Location

init_orekit()


class Localization:
    """
    Base class for localization function.
    Underlying model can be both multi layer localization grids or RPCs models
    """

    def __init__(self, model, elevation=None, image=None, epsg=None):
        """
        Localization constructor

        :param model: geometric model
        :type model: shareloc.grid or  shareloc.rpc or pyrugged_geom
        :param elevation: dtm or default elevation over ellipsoid if None elevation is set to 0
        :type elevation: shareloc.dtm or float or np.ndarray
        :param image: image class to handle geotransform
        :type image: shareloc.image.Image
        :param epsg: coordinate system of world points, if None model coordiante system will be used
        :type epsg: int
        """
        self.use_rpc = model.type == "rpc"
        self.model = model
        self.default_elevation = 0.0
        self.dtm = None
        if isinstance(elevation, (numbers.Number, list, np.ndarray)):
            self.default_elevation = elevation
        else:
            self.dtm = elevation
        self.image = image
        self.epsg = epsg

    def direct(self, row, col, h=None, using_geotransform=False):

        #diff localization entre rpc et pyrugged de 1 npour les colles -> bizarre
        """
        direct localization

        :param row: sensor row
        :type row: float or 1D np.ndarray
        :param col: sensor col
        :type col: float or 1D np.ndarray
        :param h: altitude, if none DTM is used
        :type h: float or 1D np.ndarray
        :param using_geotransform: using_geotransform
        :type using_geotransform: boolean
        :return coordinates: [lon,lat,h] (2D np.array)
        :rtype: np.ndarray of 2D dimension
        """
        if using_geotransform and self.image is not None:
            row, col = self.image.transform_index_to_physical_point(row, col)

        if self.model.__class__.__name__!="Pyrugged_geom":
            if h is not None:
                coords = self.model.direct_loc_h(row, col, h)
                epsg = self.model.epsg
            elif self.dtm is not None:
                coords = self.model.direct_loc_dtm(row, col, self.dtm)
                epsg = self.dtm.epsg
            else:
                coords = self.model.direct_loc_h(row, col, self.default_elevation)
                epsg = self.model.epsg
        else:#pyrugged geom used
            if self.dtm is not None:
                coords = self.model.direct_loc(row=row, col=col, alt=h, location_dimap=self.dtm)
                epsg = self.model.epsg
            elif h is not None:
                location_dimap = Location(
                    self.model.dimap,
                    sensor_name="sensor_a",
                    physical_data_dir=None,
                    dem_path=None,
                    geoid_path=None,
                    alti_over_ellipsoid=0.0,
                    light_time = self.model.light_time,
                    aberration_light = self.model.aberration_light,
                    atmospheric_refraction = self.model.atmospheric_refraction,
                    transforms = self.model.transforms,
                    )
                self.dtm = location_dimap
                coords = self.model.direct_loc(row, col, h, self.dtm)
                epsg = self.model.epsg


            else:#no location_dimap so we create one
                location_dimap = Location(
                    self.model.dimap,
                    sensor_name="sensor_a",
                    physical_data_dir=None,
                    dem_path=None,
                    geoid_path=None,
                    alti_over_ellipsoid=0.0,
                    light_time = self.model.light_time,
                    aberration_light = self.model.aberration_light,
                    atmospheric_refraction = self.model.atmospheric_refraction,
                    transforms = self.model.transforms,
                    )
                self.dtm = location_dimap
                coords = self.model.direct_loc(row, col, self.default_elevation, self.dtm)
                epsg = self.model.epsg
        if self.epsg is not None and self.epsg != epsg:
            return coordinates_conversion(coords, epsg, self.epsg)
        return coords

    def extent(self, margin=0.0):
        """
        returns model extent:
            * whole validity domains if image is not given
            * image footprint if image is set
            * epipolar footprint if right_model is set

        :param margin: footprint margin (in degrees)
        :type margin: float
        :return: extent [lon_min,lat_min,lon max,lat max] (2D np.array)
        :rtype: numpy.array
        """
        footprint = np.zeros([2, 2])
        if self.image is not None:
            logging.debug("image extent")
            footprint[0, :] = [-0.5, -0.5]
            footprint[1, :] = [-0.5 + self.image.nb_rows, -0.5 + self.image.nb_columns]
            using_geotransform = True
        else:
            logging.debug("model extent")
            footprint[0, :] = [self.model.row0, self.model.col0]
            footprint[1, :] = [self.model.rowmax, self.model.colmax]
            using_geotransform = False
        on_ground_pos = self.direct(footprint[:, 0], footprint[:, 1], 0, using_geotransform=using_geotransform)
        [lon_min, lat_min, __] = np.min(on_ground_pos, 0)
        [lon_max, lat_max, __] = np.max(on_ground_pos, 0)
        return np.array([lat_min - margin, lon_min - margin, lat_max + margin, lon_max + margin])

    def inverse(self, lon, lat, h=None, using_geotransform=False):
        """
        inverse localization

        :param lat:  latitude (or y)
        :param lon: longitude (or x)
        :param h: altitude
        :param using_geotransform: using_geotransform
        :type using_geotransform: boolean
        :return: coordinates [row,col,h] (1D np.ndarray)
        :rtype: Tuple(1D np.ndarray row position, 1D np.ndarray col position, 1D np.ndarray alt)
        """

        if not self.use_rpc and not hasattr(self.model, "pred_ofset_scale_lon") and self.model.__class__.__name__!="Pyrugged_geom":#grid geom
            self.model.estimate_inverse_loc_predictor()
        if h is None:
            h = self.default_elevation

        if self.epsg is not None and self.model.epsg != self.epsg:
            if isinstance(lon, np.ndarray) and isinstance(lat, np.ndarray):
                coords = np.full([lon.shape[0], 3], fill_value=0.0)
            else:
                coords = np.full([1, 3], fill_value=0.0)
            coords[:, 0] = lon
            coords[:, 1] = lat
            coords[:, 2] = h

            converted_coords = coordinates_conversion(coords, self.epsg, self.model.epsg)
            lon = converted_coords[:, 0]
            lat = converted_coords[:, 1]
            h = converted_coords[:, 2] 
        
        if self.model.__class__.__name__!="Pyrugged_geom":
            row, col, __ = self.model.inverse_loc(lon, lat, h)
        else:#Pyrugged_geom is used
            if self.dtm is not None:
                row, col, _ = self.model.inverse_loc(self.dtm, lon,lat, h)
            else:# no Location object so we create one
                location_dimap = Location(
                    self.model.dimap,
                    sensor_name="sensor_a",
                    physical_data_dir=None,
                    dem_path=None,
                    geoid_path=None,
                    alti_over_ellipsoid=0.0,
                    light_time = self.model.light_time,
                    aberration_light = self.model.aberration_light,
                    atmospheric_refraction = self.model.atmospheric_refraction,
                    transforms = self.model.transforms,
                    )
                row, col, _ = self.model.inverse_loc(location_dimap, lon, lat, h)

   
        if using_geotransform and self.image is not None:
            row, col = self.image.transform_physical_point_to_index(np.array(row), np.array(col))
        return row, col, h


def coloc(model1, model2, row, col, elevation=None, image1=None, image2=None, using_geotransform=False,elevation2=None,altitude=None):
    """
    Colocalization : direct localization with model1, then inverse localization with model2

    :param model1: geometric model 1
    :type model1: shareloc.grid or  shareloc.rpc or shareloc.Pyrugged_geom
    :param model2: geometric model 2
    :type model2: shareloc.grid or  shareloc.rpc or shareloc.Pyrugged_geom
    :param row: sensor row
    :type row: int or 1D numpy array
    :param col: sensor col
    :type col: int or 1D numpy array
    :param elevation: elevation
    :type elevation: shareloc.dtm or float or 1D numpy array or shareloc.location_dimap
    :param elevation2: elevation2 (Location class linked to the rigth image dimap)
    :type elevation2: shareloc.location_pydimap
    :param image1: image class to handle geotransform
    :type image1: shareloc.image.Image
    :param image2: image class to handle geotransform
    :type image2: shareloc.image.Image
    :param using_geotransform: using_geotransform
    :type using_geotransform: boolean
    :return: Corresponding sensor position [row, col, True] in the geometric model 2
    :rtype: Tuple(1D np.array row position, 1D np.array col position, 1D np.array alt)
    """

    if not isinstance(row, (list, np.ndarray)):
        row = np.array([row])
        col = np.array([col])
        if altitude is not None:
            altitude = np.array([altitude])

    if model1.__class__.__name__=="Pyrugged_geom" and elevation2 is not None:
        geometric_model1 = Localization(model1, elevation, image=image1)
        geometric_model2 = Localization(model2, elevation2, image=image2)
        #print("altitude : ",altitude)
        ground_coord = geometric_model1.direct(row, col, altitude, using_geotransform=using_geotransform)

    else:
        geometric_model1 = Localization(model1, elevation, image=image1)
        geometric_model2 = Localization(model2, elevation, image=image2)
        ground_coord = geometric_model1.direct(row, col, using_geotransform=using_geotransform)

    # Estimate sensor position (row, col, altitude) using inverse localization with model2
    sensor_coord = np.zeros((row.shape[0], 3), dtype=np.float64)
    sensor_coord[:, 0], sensor_coord[:, 1], sensor_coord[:, 2] = geometric_model2.inverse(
        ground_coord[:, 0], ground_coord[:, 1], ground_coord[:, 2], using_geotransform
    )

    return sensor_coord[:, 0], sensor_coord[:, 1], sensor_coord[:, 2]
  