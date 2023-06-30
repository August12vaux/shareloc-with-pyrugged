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
Test module for triangulation class shareloc/geofunctions/triangulation.py

Netcdf4 historical ref data converted to pickle to avoid netcdf4 dependency :
pickle.dump(xarray_data, open('xarray_data_file.pickle','wb'))
To be changed with CARS output when interface is stable
"""

import logging

# Standard imports
import os
import pickle

# Third party imports
import numpy as np
import pytest
import xarray as xr
import tifffile as tiff
import rasterio
import otbApplication
import matplotlib.pyplot as plt

from pyrugged.configuration.init_orekit import init_orekit
from org.hipparchus.geometry.euclidean.threed import Vector3D
from pydimaprugged.dimap_parser.phr_parser import PHRParser
from pyrugged.los.sinusoidal_rotation import SinusoidalRotation

# Shareloc imports
from shareloc.geofunctions.triangulation import distance_point_los, epipolar_triangulation, sensor_triangulation, transform_disp_to_matches
from shareloc.geomodels.grid import Grid
from shareloc.geomodels.rpc import RPC
from shareloc.geomodels.pyrugged_geom import Pyrugged_geom
from shareloc.proj_utils import coordinates_conversion

from pyrugged.refraction.multi_layer_model import MultiLayerModel
from pyrugged.bodies.extended_ellipsoid import ExtendedEllipsoid
from pyrugged.bodies.body_rotating_frame_id import BodyRotatingFrameId
from pyrugged.model.pyrugged_builder import select_ellipsoid, select_body_rotating_frame
from pyrugged.bodies.ellipsoid_id import EllipsoidId

# Shareloc test imports
from ..helpers import data_path


@pytest.mark.parametrize("col,row,h", [(1000.5, 1500.5, 10.0)])
@pytest.mark.unit_tests
def test_sensor_triangulation(row, col, h):
    """
    Test sensor triangulation
    """

    # First read the left and right geometric models (here Grids)
    id_scene_right = "P1BP--2017092838319324CP"
    grid_right = prepare_loc("ellipsoide", id_scene_right)
    id_scene_left = "P1BP--2017092838284574CP"
    grid_left = prepare_loc("ellipsoide", id_scene_left)

    # We need matches between left and right image. In real case use correlator or SIFT points.
    # In this test example we create a match by colocalization of one point.
    grid_right.estimate_inverse_loc_predictor()
    lonlatalt = grid_left.direct_loc_h(row, col, h)
    inv_row, inv_col, __ = grid_right.inverse_loc(lonlatalt[0][0], lonlatalt[0][1], lonlatalt[0][2])
    # matches are defined as Nx4 array, here N=1
    matches = np.zeros([1, 4])
    matches[0, :] = [col, row, inv_col[0], inv_row[0]]

    # compute triangulation with residues (see sensor_triangulation docstring for further details),
    point_ecef, point_wgs84, distance = sensor_triangulation(matches, grid_left, grid_right, residues=True)

    logging.info("cartesian coordinates :")
    logging.info(point_ecef)

    assert lonlatalt[0][0] == pytest.approx(point_wgs84[0, 0], abs=1e-8)
    assert lonlatalt[0][1] == pytest.approx(point_wgs84[0, 1], abs=1e-8)
    assert lonlatalt[0][2] == pytest.approx(point_wgs84[0, 2], abs=8e-3)
    # residues is approx 0.0 meter here since los intersection is ensured by colocalization
    assert distance == pytest.approx(0.0, abs=1e-3)


def prepare_loc(alti="geoide", id_scene="P1BP--2017030824934340CP"):
    """
    Read multiH grid

    :param alti: alti validation dir
    :param id_scene: scene ID
    :return: multi H grid
    :rtype: str
    """
    data_folder = data_path(alti, id_scene)
    # load grid
    gld = os.path.join(data_folder, f"GRID_{id_scene}.tif")
    gri = Grid(gld)

    return gri


@pytest.mark.unit_tests
def test_triangulation_residues():
    """
    Test triangulation residues on simulated LOS
    """

    class SimulatedLOS:
        """line of sight class"""

        def __init__(self):
            self.sis = np.array([[100.0, 10.0, 200.0], [100.0, 10.0, 200.0]])
            self.vis = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        def print_sis(self):
            """
            print los hat
            """
            print(self.sis)

        def print_vis(self):
            """
            print los viewing vector
            """
            print(self.vis)

    los = SimulatedLOS()

    distance = 10.0
    point = los.sis + 100.0 * los.vis + distance * np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    residue = distance_point_los(los, point)
    assert distance == pytest.approx(residue, abs=1e-9)


@pytest.mark.unit_tests
def test_epi_triangulation_sift():
    """
    Test epipolar triangulation
    """
    id_scene_right = "P1BP--2017092838319324CP"
    grid_right = prepare_loc("ellipsoide", id_scene_right)
    id_scene_left = "P1BP--2017092838284574CP"
    grid_left = prepare_loc("ellipsoide", id_scene_left)

    grid_left_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid.tif")
    grid_right_filename = os.path.join(data_path(), "rectification_grids", "right_epipolar_grid.tif")

    matches_filename = os.path.join(data_path(), "triangulation", "matches-crop.npy")
    matches = np.load(matches_filename)

    point_ecef, __, __ = epipolar_triangulation(
        matches, None, "sift", grid_left, grid_right, grid_left_filename, grid_right_filename
    )
    valid = [4584341.37359843123704195022583, 572313.675204274943098425865173, 4382784.51356450468301773071289]
    assert valid == pytest.approx(point_ecef[0, :], abs=0.5)


@pytest.mark.unit_tests
def test_epi_triangulation_sift_rpc():
    """
    Test epipolar triangulation
    """

    data_folder = data_path()
    id_scene = "PHR1B_P_201709281038045_SEN_PRG_FC_178608-001"
    file_geom = os.path.join(data_folder, f"rpc/{id_scene}.geom")
    geom_model_left = RPC.from_any(file_geom, topleftconvention=True)
    id_scene = "PHR1B_P_201709281038393_SEN_PRG_FC_178609-001"
    file_geom = os.path.join(data_folder, f"rpc/{id_scene}.geom")
    geom_model_right = RPC.from_any(file_geom, topleftconvention=True)

    grid_left_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid.tif")
    grid_right_filename = os.path.join(data_path(), "rectification_grids", "right_epipolar_grid.tif")

    matches_filename = os.path.join(data_path(), "triangulation", "matches-crop.npy")
    matches = np.load(matches_filename)

    point_ecef, __, __ = epipolar_triangulation(
        matches, None, "sift", geom_model_left, geom_model_right, grid_left_filename, grid_right_filename
    )
    valid = [4584341.37359843123704195022583, 572313.675204274943098425865173, 4382784.51356450468301773071289]
    # print(valid - point_ecef[0,:])
    assert valid == pytest.approx(point_ecef[0, :], abs=1e-3)


def stats_diff(cloud, array_epi):
    """
    compute difference statistics between dataset and shareloc results
    :param cloud :  CARS dataset
    :type cloud : xarray dataset
    :param array_epi :  shareloc array
    :type array_epi : numpy.array
    :return stats [mean,min, max]
    :rtype numpy.array
    """
    coords_x = cloud.x.values
    coords_y = cloud.y.values
    coords_z = cloud.z.values
    diff_x = abs(coords_x - array_epi[:, :, 0])
    diff_y = abs(coords_y - array_epi[:, :, 1])
    diff_z = abs(coords_z - array_epi[:, :, 2])

    stats = np.zeros([3, 3])
    mean_x = np.mean(diff_x)
    min_x = np.min(diff_x)
    max_x = np.max(diff_x)
    stats[0, :] = [mean_x, min_x, max_x]

    mean_y = np.mean(diff_y)
    min_y = np.min(diff_y)
    max_y = np.max(diff_y)
    stats[1, :] = [mean_y, min_y, max_y]
    mean_z = np.mean(diff_z)
    min_z = np.min(diff_z)
    max_z = np.max(diff_z)
    stats[2, :] = [mean_z, min_z, max_z]
    return stats


@pytest.mark.unit_tests
def test_epi_triangulation_disp_rpc():
    """
    Test epipolar triangulation
    """
    data_folder = data_path()
    id_scene = "PHR1B_P_201709281038045_SEN_PRG_FC_178608-001"
    file_geom = os.path.join(data_folder, f"rpc/{id_scene}.geom")
    geom_model_left = RPC.from_any(file_geom, topleftconvention=True)
    id_scene = "PHR1B_P_201709281038393_SEN_PRG_FC_178609-001"
    file_geom = os.path.join(data_folder, f"rpc/{id_scene}.geom")
    geom_model_right = RPC.from_any(file_geom, topleftconvention=True)

    # grid_left_filename = os.path.join(data_path(), "rectification_grids",
    #                                  "grid_{}.tif".format(id_scene_left))
    # grid_right_filename = os.path.join(data_path(), "rectification_grids",
    #                                   "grid_{}.tif".format(id_scene_right))

    grid_left_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid.tif")
    grid_right_filename = os.path.join(data_path(), "rectification_grids", "right_epipolar_grid.tif")

    disp_filename = os.path.join(data_path(), "triangulation", "disparity-crop.pickle")
    with open(disp_filename, "rb") as disp_file:
        disp = pickle.load(disp_file)
    point_ecef, _, _ = epipolar_triangulation(
        disp, None, "disp", geom_model_left, geom_model_right, grid_left_filename, grid_right_filename, residues=True
    )

    # open cloud
    cloud_filename = os.path.join(data_path(), "triangulation", "cloud_ECEF.pickle")
    with open(cloud_filename, "rb") as cloud_file:
        cloud = pickle.load(cloud_file)
    array_shape = disp.disp.values.shape
    array_epi_ecef = point_ecef.reshape((array_shape[0], array_shape[1], 3))
    stats = stats_diff(cloud, array_epi_ecef)
    index = 1492
    assert point_ecef[index, 0] == pytest.approx(cloud.x.values.flatten()[index], abs=1e-3)
    assert point_ecef[index, 1] == pytest.approx(cloud.y.values.flatten()[index], abs=1e-3)
    assert point_ecef[index, 2] == pytest.approx(cloud.z.values.flatten()[index], abs=1e-3)
    assert stats[:, 2] == pytest.approx([0, 0, 0], abs=6e-4)


@pytest.mark.unit_tests
def test_epi_triangulation_disp_rpc_roi():
    """
    Test epipolar triangulation
    """
    data_folder = data_path()
    file_geom = os.path.join(data_folder, "rpc/phr_ventoux/left_image.geom")
    geom_model_left = RPC.from_any(file_geom, topleftconvention=True)
    file_geom = os.path.join(data_folder, "rpc/phr_ventoux/right_image.geom")
    geom_model_right = RPC.from_any(file_geom, topleftconvention=True)

    grid_left_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid_ventoux.tif")
    grid_right_filename = os.path.join(data_path(), "rectification_grids", "right_epipolar_grid_ventoux.tif")

    disp_filename = os.path.join(data_path(), "triangulation", "disp1_ref.pickle")
    with open(disp_filename, "rb") as disp_file:
        disp = pickle.load(disp_file)
    __, point_wgs84, __ = epipolar_triangulation(
        disp,
        None,
        "disp",
        geom_model_left,
        geom_model_right,
        grid_left_filename,
        grid_right_filename,
        residues=True,
        fill_nan=True,
    )

    # open cloud
    cloud_filename = os.path.join(data_path(), "triangulation", "triangulation1_ref.pickle")
    with open(cloud_filename, "rb") as cloud_file:
        cloud = pickle.load(cloud_file)
    array_shape = disp.disp.values.shape
    array_epi_wgs84 = point_wgs84.reshape((array_shape[0], array_shape[1], 3))
    stats = stats_diff(cloud, array_epi_wgs84)
    # 1492 first non masked index
    index = 100
    assert point_wgs84[index, 0] == pytest.approx(cloud.x.values.flatten()[index], abs=1e-8)
    assert point_wgs84[index, 1] == pytest.approx(cloud.y.values.flatten()[index], abs=1e-8)
    assert point_wgs84[index, 2] == pytest.approx(cloud.z.values.flatten()[index], abs=1e-3)
    assert stats[:, 2] == pytest.approx([0, 0, 0], abs=6e-4)


@pytest.mark.unit_tests
def test_epi_triangulation_disp_grid():
    """
    Test epipolar triangulation
    """
    id_scene_left = "P1BP--2017092838284574CP"
    id_scene_right = "P1BP--2017092838319324CP"
    gri_right = prepare_loc("ellipsoide", id_scene_right)

    gri_left = prepare_loc("ellipsoide", id_scene_left)

    # grid_left_filename = os.path.join(data_path(), "rectification_grids",
    #                                  "grid_{}.tif".format(id_scene_left))
    # grid_right_filename = os.path.join(data_path(), "rectification_grids",
    #                                   "grid_{}.tif".format(id_scene_right))

    grid_left_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid.tif")
    grid_right_filename = os.path.join(data_path(), "rectification_grids", "right_epipolar_grid.tif")

    disp_filename = os.path.join(data_path(), "triangulation", "disparity-crop.pickle")
    with open(disp_filename, "rb") as disp_file:
        disp = pickle.load(disp_file)

    point_ecef, _, _ = epipolar_triangulation(
        disp, None, "disp", gri_left, gri_right, grid_left_filename, grid_right_filename, residues=True
    )

    # open cloud
    cloud_filename = os.path.join(data_path(), "triangulation", "cloud_ECEF.pickle")
    with open(cloud_filename, "rb") as cloud_file:
        cloud = pickle.load(cloud_file)
    array_shape = disp.disp.values.shape
    array_epi_ecef = point_ecef.reshape((array_shape[0], array_shape[1], 3))

    stats = stats_diff(cloud, array_epi_ecef)
    # 1492 first non masked index
    index = 1492
    assert point_ecef[index, 0] == pytest.approx(cloud.x.values.flatten()[index], abs=0.3)
    assert point_ecef[index, 1] == pytest.approx(cloud.y.values.flatten()[index], abs=0.3)
    assert point_ecef[index, 2] == pytest.approx(cloud.z.values.flatten()[index], abs=0.3)
    assert stats[:, 2] == pytest.approx([0, 0, 0], abs=0.5)


@pytest.mark.unit_tests
def test_epi_triangulation_disp_grid_masked():
    """
    Test epipolar triangulation
    """
    id_scene_left = "P1BP--2017092838284574CP"
    id_scene_right = "P1BP--2017092838319324CP"
    gri_right = prepare_loc("ellipsoide", id_scene_right)

    gri_left = prepare_loc("ellipsoide", id_scene_left)

    grid_left_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid.tif")
    grid_right_filename = os.path.join(data_path(), "rectification_grids", "right_epipolar_grid.tif")

    disp_filename = os.path.join(data_path(), "triangulation", "disparity-crop.pickle")
    with open(disp_filename, "rb") as disp_file:
        disp = pickle.load(disp_file)
    mask_array = disp.msk.values
    point_ecef, __, __ = epipolar_triangulation(
        disp, mask_array, "disp", gri_left, gri_right, grid_left_filename, grid_right_filename, residues=True
    )

    assert np.array_equal(point_ecef[0, :], [0, 0, 0])

@pytest.mark.unit_tests
def test_epi_triangulation_sift_pyrugged():
    """
    Test epipolar triangulation
    """

    init_orekit()

    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )

    atmospheric_refraction = MultiLayerModel(ellipsoid)

    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/triangulation/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/triangulation/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    geom_left_no_effects = Pyrugged_geom(file_dimap_left,None,None,None)
    geom_right_no_effects = Pyrugged_geom(file_dimap_right,None,None,None)

    geom_left_lightTime = Pyrugged_geom(file_dimap_left,True,None,None)
    geom_right_lightTime = Pyrugged_geom(file_dimap_right,True,None,None)

    geom_left_aber = Pyrugged_geom(file_dimap_left,None,True,None)
    geom_right_aber = Pyrugged_geom(file_dimap_right,None,True,None)

    geom_left_refra = Pyrugged_geom(file_dimap_left,None,None,atmospheric_refraction)
    geom_right_refra = Pyrugged_geom(file_dimap_right,None,None,atmospheric_refraction)

    geom_left_all = Pyrugged_geom(file_dimap_left,True,True,atmospheric_refraction)
    geom_right_all = Pyrugged_geom(file_dimap_right,True,True,atmospheric_refraction)

    abs_date = PHRParser(file_dimap_right,"just_for_time_ref").start_time
    amp = 0.2e-6
    freq = 30
    phase = 0
    transforms=[SinusoidalRotation("sinRot", Vector3D.PLUS_J, abs_date, amp, freq, phase)]
    #même transfo donc même referentiel de temps
    geom_left_sinusoid = Pyrugged_geom(file_dimap_left,None,None,None,transforms)
    geom_right_sinusoid = Pyrugged_geom(file_dimap_right,None,None,None,transforms)


    grid_left_filename = "/new_cars/shareloc/tests/data/pyrugged/triangulation/left_el0_st60.tif"
    grid_right_filename = "/new_cars/shareloc/tests/data/pyrugged/triangulation/right_el0_st60.tif"

    #grilles générées avec l'OTB 7.4 qui génére des artéfacts dans les grilles de stréréorectification
    #  voir (issue gitlab pydimaprugged).
    #Il est donc quasiment certain de la grille de droite soit imprécise. Cependant cela n'a aucun effets sur la
    #comparaison des résultats car elle a été passée en argument des 2 fonctions que l'on compare.
 

    im = tiff.imread('/new_cars/shareloc/tests/data/pyrugged/triangulation/disp_trian_otb_size.tif')
    disp_map = np.squeeze(np.array(im)[:,:,0])#take only horizonatal disparity

    #disp2matches
    col, row = np.meshgrid(list(range(np.shape(disp_map)[1])), list(range(np.shape(disp_map)[0])))
    epi_left_pos = np.vstack((col.flatten(), row.flatten()))
    epi_right_pos = np.vstack((col.flatten() + disp_map.flatten(), row.flatten()))
    matches = np.vstack((epi_left_pos,epi_right_pos)).transpose()

    tiff.imsave("/new_cars/shareloc/tests/data/pyrugged/triangulation/matches.tif",matches)

    _, point_wsg84_no_effects, _ = epipolar_triangulation(
        matches, None, "sift", geom_left_no_effects, geom_right_no_effects, grid_left_filename, grid_right_filename
    )
    #print("\n\n LIGHT TIME")
    _, point_wsg84_lightTime, _ = epipolar_triangulation(
        matches, None, "sift", geom_left_lightTime, geom_right_lightTime, grid_left_filename, grid_right_filename
    )
    #print("\n\n ABERRATION")
    _, point_wsg84_aber, _ = epipolar_triangulation(
        matches, None, "sift", geom_left_aber, geom_right_aber, grid_left_filename, grid_right_filename
    )
    #print("\n\n REFRACTION")
    _, point_wsg84_refra, _ = epipolar_triangulation(
        matches, None, "sift", geom_left_refra, geom_right_refra, grid_left_filename, grid_right_filename
    )
    #print("\n\n ALL ")
    _, point_wsg84_all, _ = epipolar_triangulation(
        matches, None, "sift", geom_left_all, geom_right_all, grid_left_filename, grid_right_filename
    )

    _, point_wsg84_sinusoid, _ = epipolar_triangulation(
        matches, None, "sift", geom_left_sinusoid, geom_right_sinusoid, grid_left_filename, grid_right_filename
    )

    ref = tiff.imread('/new_cars/shareloc/tests/data/pyrugged/triangulation/out_trian_otb_macthes.tif')
    ref = np.squeeze(np.array(ref)) #(1156,3)


    print("\n\n\nNO EFFECTS : ")
    print("diff lon lat : ",np.amax(abs(point_wsg84_no_effects[:,:2]-ref[:,:2])))
    print("diff alt : ",np.amax(abs(point_wsg84_no_effects[:,2]-ref[:,2])))
    print("diff moy lon lat : ",np.mean(abs(point_wsg84_no_effects[:,:2]-ref[:,:2])))
    print("diff moy alt : ",np.mean(abs(point_wsg84_no_effects[:,2]-ref[:,2])))

    print("\nlight time : ")
    print("diff lon lat : ",np.amax(abs(point_wsg84_lightTime[:,:2]-ref[:,:2])))
    print("diff alt : ",np.amax(abs(point_wsg84_lightTime[:,2]-ref[:,2])))
    print("diff moy lon lat : ",np.mean(abs(point_wsg84_lightTime[:,:2]-ref[:,:2])))
    print("diff moy alt : ",np.mean(abs(point_wsg84_lightTime[:,2]-ref[:,2])))

    print("\n aberation : ")
    print("diff lon lat : ",np.amax(abs(point_wsg84_aber[:,:2]-ref[:,:2])))
    print("diff alt : ",np.amax(abs(point_wsg84_aber[:,2]-ref[:,2])))
    print("diff moy lon lat : ",np.mean(abs(point_wsg84_aber[:,:2]-ref[:,:2])))
    print("diff moy alt : ",np.mean(abs(point_wsg84_aber[:,2]-ref[:,2])))

    print("\n refraction : ")
    print("diff lon lat : ",np.amax(abs(point_wsg84_refra[:,:2]-ref[:,:2])))
    print("diff alt : ",np.amax(abs(point_wsg84_refra[:,2]-ref[:,2])))
    print("diff moy lon lat : ",np.mean(abs(point_wsg84_refra[:,:2]-ref[:,:2])))
    print("diff moy alt : ",np.mean(abs(point_wsg84_refra[:,2]-ref[:,2]))) #ref>point_wsg84_refra

    print("\n ALL EFFECTS : ")
    print("diff lon lat : ",np.amax(abs(point_wsg84_all[:,:2]-ref[:,:2])))
    print("diff alt : ",np.amax(abs(point_wsg84_all[:,2]-ref[:,2])))
    print("diff moy lon lat : ",np.mean(abs(point_wsg84_all[:,:2]-ref[:,:2])))
    print("diff moy alt : ",np.mean(abs(point_wsg84_all[:,2]-ref[:,2])))

    print("\n SINSOIDE : ")
    print("diff lon lat : ",np.amax(abs(point_wsg84_sinusoid[:,:2]-ref[:,:2])))
    print("diff alt : ",np.amax(abs(point_wsg84_sinusoid[:,2]-ref[:,2])))
    print("diff moy lon lat : ",np.mean(abs(point_wsg84_sinusoid[:,:2]-ref[:,:2])))
    print("diff moy alt : ",np.mean(abs(point_wsg84_sinusoid[:,2]-ref[:,2])))

    # NO EFFECTS : 
    # diff lon lat :  1.0416779360866713e-05
    # diff alt :  0.07731775846332312
    # diff moy lon lat :  3.428182015879545e-06
    # diff moy alt :  0.07730821765005383

    # light time : 
    # diff lon lat :  1.5072462446141799e-05
    # diff alt :  0.07662536483258009
    # diff moy lon lat :  5.047817216657328e-06
    # diff moy alt :  0.07661597216727412

    # aberation : 
    # diff lon lat :  0.00016958894725149776
    # diff alt :  0.07987968903034925
    # diff moy lon lat :  0.0001015626917113774
    # diff moy alt :  0.07983300366332581

    # refraction : 
    # diff lon lat :  1.041670068957501e-05
    # diff alt :  3.0431363340467215
    # diff moy lon lat :  3.4283000181334193e-06
    # diff moy alt :  3.042824337484515

    # ALL EFFECTS : 
    # diff lon lat :  0.00016958907256992006
    # diff alt :  3.044966835528612
    # diff moy lon lat :  0.00010671462768416442
    # diff moy alt :  3.0446522487198555

    # SINSOIDE : 
    # diff lon lat :  1.0412989865926647e-05
    # diff alt :  0.21486869174987078
    # diff moy lon lat :  3.661427923192322e-06
    # diff moy alt :  0.15304525217821235



    assert point_wsg84_no_effects[:,:2] == pytest.approx(ref[:,:2], abs=2e-5)
    assert point_wsg84_no_effects[:,2] == pytest.approx(ref[:,2], abs=0.1)

    assert point_wsg84_lightTime[:,:2] == pytest.approx(ref[:,:2], abs=2e-5)
    assert point_wsg84_lightTime[:,2] == pytest.approx(ref[:,2], abs=0.1)

    assert point_wsg84_aber[:,:2] == pytest.approx(ref[:,:2], abs=2e-4)
    assert point_wsg84_aber[:,2] == pytest.approx(ref[:,2], abs=0.1)

    assert point_wsg84_refra[:,:2] == pytest.approx(ref[:,:2], abs=2e-5)
    assert point_wsg84_refra[:,2] == pytest.approx(ref[:,2], abs=5)

    assert point_wsg84_all[:,:2] == pytest.approx(ref[:,:2], abs=2e-4)
    assert point_wsg84_all[:,2] == pytest.approx(ref[:,2], abs=5)

    assert point_wsg84_sinusoid[:,:2] == pytest.approx(point_wsg84_no_effects[:,:2], abs=2e-5)
    assert point_wsg84_sinusoid[:,2] == pytest.approx(point_wsg84_no_effects[:,2], abs=0.3)






    