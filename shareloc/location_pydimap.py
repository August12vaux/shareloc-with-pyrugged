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
This module contains the DTMIntersection class to handle DTM intersection.
"""
import numpy as np

from pyrugged.configuration.init_orekit import init_orekit
from pydimaprugged.process.pyrugged_initialization import initialize_pyrugged
from pyrugged.location.optical import CorrectionsParams
from pydimaprugged.static_configuration import load_config
from pyrugged.intersection.constant_elevation_algorithm import ConstantElevationAlgorithm



init_orekit()


class Location:
    
    def __init__(
            self,
            dimap:str,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid =0.0,
            light_time: bool = None,
            aberration_light: bool = None,
            atmospheric_refraction:bool = None,
            transforms= None,
            ) -> None:

        conf = {
                "col_margin": 0,
                "time_line_margin": False,
                "line_margin": 0,
        }
        load_config(conf)
        self.dimap = dimap
        self.physical_data_dir = physical_data_dir
        self.dem_path = dem_path
        self.geoid_path = geoid_path
        self.alti_over_ellipsoid = alti_over_ellipsoid
        self.light_time = light_time
        self.aberration_light = aberration_light
        self.transforms = transforms
        self.corrections_params = CorrectionsParams(light_time, aberration_light, atmospheric_refraction)
        self.sensor_name = sensor_name
        self.location, self.parser = initialize_pyrugged(
                dimap_path=dimap,
                sensor_name=sensor_name,
                corrections_params=self.corrections_params,
                physical_data_dir=physical_data_dir,
                dem_path=dem_path,
                geoid_path=geoid_path,
                alti_over_ellipsoid=alti_over_ellipsoid,
                transforms=transforms,
                )
        
        self.atmospheric_refraction = self.location.atmospheric_refraction
        if self.location.algorithm.__class__.__name__ == "IgnoreDEMAlgorithm":
            self.location.algorithm = ConstantElevationAlgorithm(0.0)
        
  
    def set_alti_over_ellipsoid(self, alti:float):

        self.alti_over_ellipsoid = alti
        self.location, self.parser = initialize_pyrugged(
                dimap_path=self.dimap,
                sensor_name=self.sensor_name,
                corrections_params=self.corrections_params,
                physical_data_dir=self.physical_data_dir,
                dem_path=self.dem_path,
                geoid_path=self.geoid_path,
                alti_over_ellipsoid=alti,
                transforms= self.transforms,
                )
        return
        

    def set_location_dem_geoid(self, dem, geoid):# unused but could be useful
        self.dem_path = dem
        self.geoid_path = geoid
        self.location, self.parser = initialize_pyrugged(
                dimap_path=self.dimap,
                sensor_name=self.sensor_name,
                corrections_params=self.corrections_params,
                physical_data_dir=self.physical_data_dir,
                dem_path=dem,
                geoid_path=geoid,
                alti_over_ellipsoid=self.alti_over_ellipsoid,
                transforms= self.transforms,
                )
        return
    
    def set_algorithm(self,algo):
        self.location.algorithm = algo