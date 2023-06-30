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
Module to test functions that use Pyrugged
"""
# pylint: disable=no-member

# Third party imports
import numpy as np
import pytest

# Shareloc imports
from pyrugged.configuration.init_orekit import init_orekit
from org.hipparchus.geometry.euclidean.threed import Vector3D
from shareloc.geomodels.pyrugged_geom import Pyrugged_geom
from pyrugged.refraction.multi_layer_model import MultiLayerModel
from pyrugged.bodies.extended_ellipsoid import ExtendedEllipsoid
from pyrugged.bodies.body_rotating_frame_id import BodyRotatingFrameId
from pyrugged.model.pyrugged_builder import select_ellipsoid, select_body_rotating_frame
from pyrugged.bodies.ellipsoid_id import EllipsoidId
from pyrugged.intersection.constant_elevation_algorithm import ConstantElevationAlgorithm
from pydimaprugged.dimap_parser.phr_parser import PHRParser
from pyrugged.los.sinusoidal_rotation import SinusoidalRotation

# Shareloc test imports
from shareloc.location_pydimap import Location

init_orekit()


@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_loc(lon, lat, alt):
    """
    test inverse and direct localization using dimap files
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    location = location_dimap.location

    elevation = location.algorithm.get_elevation(np.array(float(np.radians(lat))), np.array(float(np.radians(lon))))
    print('elevation : ', elevation)
    (row, col, __) = geom.inverse_loc(location_dimap, [float(lon)], [float(lat)], [float(elevation)])
    print("(row, col)",(row, col))

    pyrugged = location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)

    print("(max_line,min_line)",(max_line,min_line))

    (row_ref, col_ref) = location.inverse_location(min_line, max_line, [float(np.radians(lat))], [float(np.radians(lon))], [float(elevation)], "sensor_a")
    print("(row_ref, col_ref)",(row_ref, col_ref))

    #assert inverse loc
    assert col == col_ref
    assert row == row_ref


    lonlatalt = geom.direct_loc(row, col, elevation, location_dimap)
    lonlatalt_ref = location.direct_location(row, col, elevation,"sensor_a")
    print("lonlatalt : ",lonlatalt)

    print('lonlatalt_ref : ',lonlatalt_ref)
    print('np.shape(lonlatalt) : ', np.shape(lonlatalt))
    
    #assert direct loc
    assert lonlatalt[:,0] == np.degrees(lonlatalt_ref[0][0])
    assert lonlatalt[:,1] == np.degrees(lonlatalt_ref[1][0])
    assert lonlatalt[:,2] == lonlatalt_ref[2][0]


    #assert identity

    print(abs(np.squeeze(lonlatalt)-np.array([lon, lat, alt])))

    assert lonlatalt[:,0] == lon
    assert lonlatalt[:,1] == pytest.approx(lat,abs=1e-14)
    assert lonlatalt[:,2] == pytest.approx(alt,abs=1e-9)


@pytest.mark.parametrize(
    "row,col,alt",
    [(120,120, 25.6)],
)
def test_direct_loc_alt(row, col, alt):
    """
    test inverse and direct localization using dimap files
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    location = location_dimap.location
    location.algorithm =ConstantElevationAlgorithm(0.0)
    
    lonlatalt = geom.direct_loc([row], [col], [alt], location_dimap)
    lonlatalt_ref = location.direct_location([row], [col], [alt],"sensor_a")

    #assert direct loc
    assert lonlatalt[:,0] == np.degrees(lonlatalt_ref[0][0])
    assert lonlatalt[:,1] == np.degrees(lonlatalt_ref[1][0])
    assert lonlatalt[:,2] == lonlatalt_ref[2][0]



@pytest.mark.parametrize(
    "lon,lat",
    [(144.9446261,-37.8232639)],
)
def test_loc_dtm(lon, lat):
    """
    test inverse and direct localization using dimap files and dem
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90_only_s38"
    geoid_file = "/new_cars/shareloc/tests/data/pyrugged/rectification/egm96.grd"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    location = location_dimap.location

    
    (row, col, __) = geom.inverse_loc(location_dimap, [float(lon)], [float(lat)])
    print("(row, col)",(row, col))

    pyrugged = location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)

    print("(max_line,min_line)",(max_line,min_line))

    (row_ref, col_ref) = location.inverse_location(min_line, max_line, np.array([float(np.radians(lat))]), np.array([float(np.radians(lon))]), None, "sensor_a")
    print("(row_ref, col_ref)",(row_ref, col_ref),end="\n\n")

    #assert inverse loc
    assert col == col_ref
    assert row == row_ref


    lonlatalt = geom.direct_loc(row, col, None, location_dimap)
    lonlatalt_ref = location.direct_location(row, col, None,"sensor_a")

    print("lonlatalt : ",lonlatalt)
    print('lonlatalt_ref : ',lonlatalt_ref)
    print('np.shape(lonlatalt) : ', np.shape(lonlatalt))
    
    #assert direct loc
    assert lonlatalt[:,0] == np.degrees(lonlatalt_ref[0][0])
    assert lonlatalt[:,1] == np.degrees(lonlatalt_ref[1][0])
    assert lonlatalt[:,2] == lonlatalt_ref[2][0]

    #assert identity

    elevation = location.algorithm.get_elevation(np.array([float(np.radians(lat))]), np.array([float(np.radians(lon))]))[0]
    
    print(abs(np.squeeze(lonlatalt)-np.array([lon, lat, elevation])))
    #assert np.squeeze(lonlatalt) == pytest.approx(np.array([lon, lat, elevation]),abs=1e-14)
    assert lonlatalt[:,0] == lon
    assert lonlatalt[:,1] == pytest.approx(lat,abs=1e-12)
    assert lonlatalt[:,2] == pytest.approx(elevation,abs=1e-9)




@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_inv_loc_multi_alt(lon, lat, alt):
    """
    test inverse and direct localization using dimap files multiples altitudes
    """

    lon = [lon]*5
    lat = [lat]*5
    alt = [0.0,10.0,25.2,80.3,251.6]


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
 
    (row, col, _) = geom.inverse_loc(location_dimap, lon, lat, alt)
    

    location = location_dimap.location
    pyrugged = location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)

    (row_ref, col_ref) = location.inverse_location(min_line, max_line, np.radians(lat), np.radians(lon), alt, "sensor_a")
    
    print(abs(col-col_ref))
    print(abs(row-row_ref))
    #assert inverse loc
    assert col == pytest.approx(col_ref,abs=1e-8)
    assert row == pytest.approx(row_ref,abs=1e-4)


@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_direct_loc_multi_alt(lon, lat, alt):
    """
    test inverse and direct localization using dimap files multiples altitudes
    """

    lon = [lon]*5
    lat = [lat]*5
    alt = [0.0,10.0,25.2,80.3,251.6]


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    location = location_dimap.location

    (row, col, alt) = geom.inverse_loc(location_dimap, lon, lat, alt)

    lonlatalt = geom.direct_loc(row, col, alt, location_dimap)
    lon_ref, lat_ref, alt_ref = location.direct_location(row, col,alt,"sensor_a")

    lon_ref = np.degrees(lon_ref)
    lat_ref = np.degrees(lat_ref)
    alt_ref = alt_ref

    lon = lonlatalt[:,0]
    lat = lonlatalt[:,1]
    alt = lonlatalt[:,2]

    diff_lon = abs(lon-lon_ref)
    diff_lat = abs(lat-lat_ref)
    diff_alt = abs(alt-alt_ref)

    print("abs(lon-lon_ref) : ",diff_lon)
    print("abs(lat-lat_ref) : ",diff_lat)
    print("abs(alt-alt_ref) : ",diff_alt)

    assert lon == pytest.approx(lon_ref, abs=1e-12)
    assert lat == pytest.approx(lat_ref, abs=1e-12)
    assert alt == pytest.approx(alt_ref, abs=1e-12)





@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_loc_correction(lon, lat, alt):
    """
    test direct localization using dimap files
    """

    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)

    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"

    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )

    location = location_dimap.location
    elevation = location.algorithm.get_elevation(np.array((np.radians(lat))), np.array(float(np.radians(lon))))


    (row, col, __) = geom.inverse_loc(location_dimap, [float(lon)], [float(lat)], [float(elevation)])
 
    pyrugged = location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)

    (row_ref, col_ref) = location.inverse_location(min_line, max_line, [float(np.radians(lat))], [float(np.radians(lon))], [float(elevation)], "sensor_a")

    #assert inverse loc

    print("abs(col-col_ref) : ",abs(col-col_ref))
    print("abs(row-row_ref) : ",abs(row-row_ref))

    assert col == pytest.approx(col_ref, abs=1e-7)
    assert row == pytest.approx(row_ref, abs=1e-7)


    lonlatalt = geom.direct_loc(row, col, elevation, location_dimap)
    lonlatalt_ref = location.direct_location(row, col, elevation,"sensor_a")

    print("lon : ",abs(lonlatalt[:,0] - np.degrees(lonlatalt_ref[0][0])))
    print("lat : ",abs(lonlatalt[:,1] - np.degrees(lonlatalt_ref[1][0])))
    print("alt : ",abs(lonlatalt[:,2] - lonlatalt_ref[2][0]))
    
    #assert direct loc
    assert lonlatalt[:,0] == np.degrees(lonlatalt_ref[0][0])
    assert lonlatalt[:,1] == np.degrees(lonlatalt_ref[1][0])
    assert lonlatalt[:,2] == lonlatalt_ref[2][0]


    #assert identity

    print(abs(np.squeeze(lonlatalt)-np.array([lon, lat, alt])))

    assert lonlatalt[:,0] == pytest.approx(lon,abs=1e-10)
    assert lonlatalt[:,1] == pytest.approx(lat,abs=1e-10)
    assert lonlatalt[:,2] == alt


@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_direct_loc_multi_alt_correction(lon, lat, alt):
    """
    test inverse and direct localization using dimap files multiples altitudes
    """

    lon = [lon]*5
    lat = [lat]*5
    alt = [0.0,10.0,25.2,80.3,251.6]



    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    px_step = 1000
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)


    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    location = location_dimap.location

    (row, col, alt) = geom.inverse_loc(location_dimap, lon, lat, alt)

    lonlatalt = geom.direct_loc(row, col, alt, location_dimap)
    lon_ref, lat_ref, alt_ref = location.direct_location(row, col, alt,"sensor_a")

    lon_ref = np.degrees(lon_ref)
    lat_ref = np.degrees(lat_ref)

    lon = lonlatalt[:,0]
    lat = lonlatalt[:,1]
    alt = lonlatalt[:,2]

    diff_lon = abs(lon-lon_ref)
    diff_lat = abs(lat-lat_ref)
    diff_alt = abs(alt-alt_ref)

    print("abs(lon-lon_ref) : ",diff_lon)
    print("abs(lat-lat_ref) : ",diff_lat)
    print("abs(alt-alt_ref) : ",diff_alt)


    assert lon == pytest.approx(lon_ref,abs=1e-12)
    assert lat == pytest.approx(lat_ref,abs=1e-12)
    assert alt == pytest.approx(alt_ref,abs=1e-12)


@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_inv_loc_multi_alt_correction(lon, lat, alt):
    """
    test inverse and direct localization using dimap files multiples altitudes
    """

    lon = [lon]*5
    lat = [lat]*5
    alt = [0.0,10.0,25.2,80.3,251.6]

    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )

    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
 
    (row, col, _) = geom.inverse_loc(location_dimap, lon, lat, alt)
    

    location = location_dimap.location
    pyrugged = location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)

    (row_ref, col_ref) = location.inverse_location(min_line, max_line, np.radians(lat), np.radians(lon), alt, "sensor_a")
    
    print(abs(col-col_ref))
    print(abs(row-row_ref))
    #assert inverse loc
    assert col == pytest.approx(col_ref,abs=1e-8)
    assert row == pytest.approx(row_ref,abs=8e-4)



@pytest.mark.parametrize(
    "lon,lat,alt",
    [(144.9446261,-37.8232639, 0.0)],
)
def test_direct_loc_multi_alt_correction_sinusoid(lon, lat, alt):
    """
    test inverse and direct localization using dimap files multiples altitudes
    """

    lon = [lon]*5
    lat = [lat]*5
    alt = [0.0,10.0,25.2,80.3,251.6]


    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    px_step = 1000
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    abs_date = PHRParser(file_dimap,"just_for_time_ref").start_time
    amp = 1e-4
    freq = 30
    phase = np.pi/2
    location_dimap = Location(
            geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            transforms=[SinusoidalRotation("sinRot", Vector3D.PLUS_J, abs_date, amp, freq, phase)],
            )
    location = location_dimap.location

    (row, col, alt) = geom.inverse_loc(location_dimap, lon, lat, alt)

    lonlatalt = geom.direct_loc(row, col, alt, location_dimap)
    lon_ref, lat_ref, alt_ref = location.direct_location(row, col, alt,"sensor_a")

    lon_ref = np.degrees(lon_ref)
    lat_ref = np.degrees(lat_ref)

    lon = lonlatalt[:,0]
    lat = lonlatalt[:,1]
    alt = lonlatalt[:,2]

    diff_lon = abs(lon-lon_ref)
    diff_lat = abs(lat-lat_ref)
    diff_alt = abs(alt-alt_ref)

    print("abs(lon-lon_ref) : ",diff_lon)
    print("abs(lat-lat_ref) : ",diff_lat)
    print("abs(alt-alt_ref) : ",diff_alt)

    assert lon == pytest.approx(lon_ref,abs=0)
    assert lat == pytest.approx(lat_ref,abs=0)
    assert alt == pytest.approx(alt_ref,abs=0)







def test_h_minmax():
    """
    test get min and max altitude
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)
    [h_min,h_max] = geom.get_alt_min_max()
    print("h_min : ",h_min)
    print("h_max : ",h_max)
    assert h_min == 30.0
    assert h_max == 90.0

