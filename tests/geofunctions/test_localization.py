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
Test module for localisation class shareloc/geofunctions/localisation.py
"""
# TODO: refactor to disable no-member in Grid class
# pylint: disable=no-member

# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# import cProfile
# import io
# import pstats
# from pstats import SortKey

from pyrugged.configuration.init_orekit import init_orekit
from org.hipparchus.geometry.euclidean.threed import Vector3D
from pydimaprugged.dimap_parser.phr_parser import PHRParser
from pyrugged.los.sinusoidal_rotation import SinusoidalRotation

from shareloc.geofunctions.dtm_intersection import DTMIntersection
from shareloc.geofunctions.localization import Localization
from shareloc.geomodels.pyrugged_geom import Pyrugged_geom
from shareloc.geofunctions.localization import coloc as coloc_rpc

# Shareloc imports
from shareloc.geomodels.grid import Grid, coloc
from shareloc.geomodels.rpc import RPC
from shareloc.image import Image
from shareloc.proj_utils import coordinates_conversion
from shareloc.location_pydimap import Location
from pyrugged.refraction.multi_layer_model import MultiLayerModel
from pyrugged.bodies.extended_ellipsoid import ExtendedEllipsoid
from pyrugged.bodies.body_rotating_frame_id import BodyRotatingFrameId
from pyrugged.model.pyrugged_builder import select_ellipsoid, select_body_rotating_frame
from pyrugged.bodies.ellipsoid_id import EllipsoidId
import pydimaprugged.static_configuration as static_cfg

# Shareloc test imports
from ..helpers import data_path


@pytest.mark.unit_tests
def test_localize_direct_rpc():
    """
    Test direct localization using image indexing and RPC model
    """

    # we want to localize the first pixel center of phr_ventoux left image
    row = 0
    col = 0

    # first instanciate the RPC geometric model
    # data = os.path.join(data_path(), "rpc/phr_ventoux/", "left_image.geom")
    # geom_model = RPC.from_any(data)
    data = os.path.join(data_path(), "rpc/phr_ventoux/", "RPC_PHR1B_P_201308051042194_SEN_690908101-001.XML")
    geom_model = RPC.from_any(data)
    # then read the Image to retrieve its geotransform
    image_filename = os.path.join(data_path(), "image/phr_ventoux/", "left_image.tif")
    image_left = Image(image_filename)

    # read the SRTM tile and Geoid
    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)

    # Localization class is then used using WGS84 output
    loc = Localization(geom_model, elevation=dtm_ventoux, image=image_left, epsg=4326)

    # use direct localisation of the first image pixel
    lonlatalt = loc.direct(row, col, using_geotransform=True)
    # [5.19340615  44.20805808 503.51202179]
    # print(lonlatalt)
    # print(geom_model_1.inverse_loc(lonlatalt[0][0], lonlatalt[0][1], lonlatalt[0][2]))
    assert lonlatalt[0][0] == pytest.approx(5.193406151946084, abs=1e-8)
    assert lonlatalt[0][1] == pytest.approx(44.20805807814395, abs=1e-8)
    assert lonlatalt[0][2] == pytest.approx(503.51202179, abs=1e-4)


@pytest.mark.unit_tests
def test_localize_direct_grid():
    """
    Test direct localization using image indexing and grid model
    """

    # we want to localize the first pixel center of phr_ventoux left image
    row = 0
    col = 0

    # first instanciate the Grid geometric model
    # data = os.path.join(data_path(), "rpc/phr_ventoux/", "left_image.geom")
    # geom_model_1 = RPC.from_any(data)
    data = os.path.join(data_path(), "grid/phr_ventoux/", "GRID_PHR1B_P_201308051042194_SEN_690908101-001.tif")
    geom_model = Grid(data)
    # then read the Image to retrieve its geotransform
    image_filename = os.path.join(data_path(), "image/phr_ventoux/", "left_image.tif")
    image_left = Image(image_filename)

    # read the SRTM tile and Geoid
    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file, fill_nodata="min")

    # Localization class is then used using WGS84 output
    loc = Localization(geom_model, elevation=dtm_ventoux, image=image_left, epsg=4326)

    # use direct localisation of the first image pixel
    # TODO: refacto interfaces to avoid np.squeeze
    lonlatalt = np.squeeze(loc.direct(row, col, using_geotransform=True))

    assert lonlatalt[0] == pytest.approx(5.193406151946084, abs=1e-7)
    assert lonlatalt[1] == pytest.approx(44.20805807814395, abs=1e-7)
    assert lonlatalt[2] == pytest.approx(503.51202179, abs=1e-3)


def prepare_loc(alti="geoide", id_scene="P1BP--2017030824934340CP"):
    """
    Read multiH grid
    :param alti: alti validation dir
    :param id_scene: scene ID
    :return: (mnt, grid)
    :rtype: list(shareloc.geofunctions.dtmIntersection.DTMIntersection,
            shareloc.grid.Grid)
    """
    data_folder = data_path(alti, id_scene)

    mnt_name = f"MNT_{id_scene}.tif"
    grid_name = f"GRID_{id_scene}.tif"
    fic = os.path.join(data_folder, mnt_name)
    # load mnt
    dtm = DTMIntersection(fic)
    # load grid model
    gld = os.path.join(data_folder, grid_name)
    gri = Grid(gld)
    return dtm, gri


@pytest.mark.parametrize("idxcol,idxrow", [(20, 10)])
@pytest.mark.parametrize("row0,col0,steprow,stepcol,nbrow,nbcol", [(100.5, 150.5, 60, 50, 200, 200)])
@pytest.mark.unit_tests
def test_gld_dtm(idxrow, idxcol, row0, col0, steprow, stepcol, nbrow, nbcol):
    """
    Test loc direct grid on dtm function
    """
    dtmbsq, gri = prepare_loc()

    gri_gld = gri.direct_loc_grid_dtm(row0, col0, steprow, stepcol, nbrow, nbcol, dtmbsq)

    lonlatalt = gri_gld[:, idxrow, idxcol]

    row = row0 + steprow * idxrow
    col = col0 + stepcol * idxcol

    # TODO: refacto interfaces to avoid np.squeeze
    valid_lonlatalt = np.squeeze(gri.direct_loc_dtm(row, col, dtmbsq))
    assert lonlatalt == pytest.approx(valid_lonlatalt, abs=1e-12)


@pytest.mark.parametrize("col,row", [(50.5, 100.5)])
@pytest.mark.parametrize("valid_lon,valid_lat,valid_alt", [(57.21700176041541, 21.959197148974, 238.0)])
@pytest.mark.unit_tests
def test_loc_dir_instersect_cube_dtm(col, row, valid_lon, valid_lat, valid_alt):
    """
    Test direct localization check dtm cube
    """
    dtmbsq, gri = prepare_loc()

    visee = np.zeros((3, gri.nbalt))
    vislonlat = gri.interpolate_grid_in_plani(row, col)
    visee[0, :] = vislonlat[0]
    visee[1, :] = vislonlat[1]
    visee[2, :] = gri.alts_down
    visee = visee.T
    (__, __, position, __) = dtmbsq.intersect_dtm_cube(visee)
    assert position[0] == pytest.approx(valid_lon, abs=1e-12)
    assert position[1] == pytest.approx(valid_lat, abs=1e-12)
    assert position[2] == pytest.approx(valid_alt, abs=1e-12)


@pytest.mark.parametrize("col,row", [(50.5, 100.5)])
@pytest.mark.parametrize("valid_lon,valid_lat", [(57.2169100597702, 21.96277930762832)])
@pytest.mark.unit_tests
def test_loc_dir_interp_visee_unitaire_gld(row, col, valid_lon, valid_lat):
    """
    Test los interpolation
    """
    ___, gri = prepare_loc()
    visee = gri.interpolate_grid_in_plani(row, col)
    assert visee[0][1] == pytest.approx(valid_lon, abs=1e-12)
    assert visee[1][1] == pytest.approx(valid_lat, abs=1e-12)


@pytest.mark.parametrize("col,row,h", [(50.5, 100.5, 100.0)])
@pytest.mark.parametrize("valid_lon,valid_lat,valid_alt", [(57.2170054518422, 21.9590529453258, 100.0)])
@pytest.mark.unit_tests
def test_sensor_loc_dir_h(col, row, h, valid_lon, valid_lat, valid_alt):
    """
    Test direct localization at constant altitude
    """
    ___, gri = prepare_loc()
    loc = Localization(gri, elevation=None)
    lonlatalt = loc.direct(row, col, h)

    assert gri.epsg == 4269
    assert valid_lon == pytest.approx(lonlatalt[0][0], abs=1e-12)
    assert valid_lat == pytest.approx(lonlatalt[0][1], abs=1e-12)
    assert valid_alt == pytest.approx(lonlatalt[0][2], abs=1e-8)


@pytest.mark.unit_tests
@pytest.mark.skip(reason="check must to be done #90")
def test_grid_extent():
    """
    Test grid extent
    """
    ___, gri = prepare_loc()
    loc = Localization(gri, elevation=None)
    np.testing.assert_allclose(loc.extent(), [21.958719, 57.216475, 22.17099, 57.529534], atol=1e-8)


@pytest.mark.unit_tests
def test_extent():
    """
    Test  extent
    """
    data_left = os.path.join(data_path(), "rectification", "left_image")
    geom_model = RPC.from_any(data_left + ".geom", topleftconvention=True)
    image_filename = os.path.join(data_path(), "image/phr_ventoux/", "left_image_pixsize_0_5.tif")
    image = Image(image_filename)
    loc_rpc_image = Localization(geom_model, elevation=None, image=image)
    np.testing.assert_allclose(loc_rpc_image.extent(), [44.20518231, 5.19307549, 44.20739814, 5.19629785], atol=1e-8)
    loc_rpc = Localization(geom_model)
    np.testing.assert_allclose(loc_rpc.extent(), [44.041678, 5.155808, 44.229592, 5.412923], atol=1e-8)


@pytest.mark.parametrize("col,row,valid_coord", [(1999.5, 999.5, (5.17388499778903, 44.2257233720898, 376.86))])
@pytest.mark.unit_tests
def test_sensor_loc_dir_dtm_geoid(col, row, valid_coord):
    """
    Test direct localization using image geotransform
    """
    data = os.path.join(data_path(), "rectification", "left_image")
    geom_model_left = RPC.from_any(data + ".geom", topleftconvention=True)
    image_filename = os.path.join(data_path(), "image/phr_ventoux/", "left_image.tif")
    image_left = Image(image_filename)

    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)
    loc = Localization(geom_model_left, elevation=dtm_ventoux, image=image_left)
    lonlatalt = loc.direct(row, col, using_geotransform=False)
    assert valid_coord[0] == pytest.approx(lonlatalt[0, 0], abs=3.0 * 1e-5)
    assert valid_coord[1] == pytest.approx(lonlatalt[0, 1], abs=2.0 * 1e-4)
    assert valid_coord[2] == pytest.approx(lonlatalt[0, 2], abs=15.0)


@pytest.mark.parametrize("col,row,valid_coord", [(1999.5, 999.5, (5.17388499778903, 44.2257233720898, 376.86))])
@pytest.mark.unit_tests
def test_sensor_loc_dir_dtm_geoid_utm(col, row, valid_coord):
    """
    Test direct localization using image geotransform
    """
    data = os.path.join(data_path(), "rectification", "left_image")
    geom_model_left = RPC.from_any(data + ".geom", topleftconvention=True)
    image_filename = os.path.join(data_path(), "image/phr_ventoux/", "left_image.tif")
    image_left = Image(image_filename)

    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_resampled_UTM31", "N44E005_UTM.tif")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)
    loc = Localization(geom_model_left, elevation=dtm_ventoux, image=image_left, epsg=4326)
    lonlatalt = loc.direct(row, col, using_geotransform=False)
    assert valid_coord[0] == pytest.approx(lonlatalt[0, 0], abs=3.0 * 1e-5)
    assert valid_coord[1] == pytest.approx(lonlatalt[0, 1], abs=2.0 * 1e-4)
    assert valid_coord[2] == pytest.approx(lonlatalt[0, 2], abs=15.0)


@pytest.mark.parametrize("col,row,h", [(150.5, 100, 100.0)])
@pytest.mark.unit_tests
def test_sensor_loc_dir_vs_loc_rpc(row, col, h):
    """
    Test direct localization coherence with grid and RPC
    """
    id_scene = "P1BP--2018122638935449CP"
    ___, gri = prepare_loc("ellipsoide", id_scene)
    assert gri.epsg == 4269
    loc_grid = Localization(gri)
    # init des predicteurs
    lonlatalt = loc_grid.direct(row, col, h)

    data_folder = data_path()
    fichier_dimap = os.path.join(data_folder, f"rpc/PHRDIMAP_{id_scene}.XML")

    fctrat = RPC.from_any(fichier_dimap)

    loc_rpc = Localization(fctrat)
    lonlatalt_rpc = loc_rpc.direct(row, col, h)

    diff_lon = lonlatalt[0][0] - lonlatalt_rpc[0][0]
    diff_lat = lonlatalt[0][1] - lonlatalt_rpc[0][1]
    diff_alt = lonlatalt[0][2] - lonlatalt_rpc[0][2]
    assert diff_lon == pytest.approx(0.0, abs=1e-7)
    assert diff_lat == pytest.approx(0.0, abs=1e-7)
    assert diff_alt == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("index_x,index_y", [(10.5, 20.5)])
@pytest.mark.unit_tests
def test_sensor_loc_dir_dtm(index_x, index_y):
    """
    Test direct localization on DTMIntersection
    """
    dtmbsq, gri = prepare_loc()
    loc = Localization(gri, elevation=dtmbsq)

    vect_index = [index_x, index_y]
    [lon, lat] = dtmbsq.index_to_ter(vect_index)
    alt = dtmbsq.interpolate(index_x, index_y)

    row, col, alt = loc.inverse(lon, lat, alt)
    # TODO: refacto interfaces to avoid np.squeeze
    lonlath = np.squeeze(loc.direct(row, col))
    assert lon == pytest.approx(lonlath[0], abs=1e-8)
    assert lat == pytest.approx(lonlath[1], abs=1e-8)
    assert alt == pytest.approx(lonlath[2], abs=1e-4)


@pytest.mark.parametrize(
    "image_name, row,col, origin_row, origin_col, pixel_size_row, pixel_size_col",
    [
        ("right_image.tif", 100, 200.5, 5162, 4915.0, 1.0, 1.0),
        ("right_image_resample.tif", 100.0, 200.0, 5162.0, 4915.0, 2.0, 0.5),
    ],
)
@pytest.mark.unit_tests
def test_image_metadata(image_name, row, col, origin_row, origin_col, pixel_size_row, pixel_size_col):
    """
    Test image class
    """
    data_folder = data_path()
    image_filename = os.path.join(data_folder, "image/phr_ventoux/", image_name)

    my_image = Image(image_filename)
    assert my_image.origin_row == origin_row
    assert my_image.origin_col == origin_col
    assert my_image.pixel_size_row == pixel_size_row
    assert my_image.pixel_size_col == pixel_size_col

    [phys_row, phys_col] = my_image.transform_index_to_physical_point(row, col)
    assert phys_row == origin_row + (row + 0.5) * pixel_size_row
    assert phys_col == origin_col + (col + 0.5) * pixel_size_col

    row_index, col_index = my_image.transform_physical_point_to_index(phys_row, phys_col)
    assert row == row_index
    assert col == col_index


@pytest.mark.parametrize("valid_row,valid_col", [(50.5, 10.0)])
@pytest.mark.parametrize("lon,lat,alt", [(57.2167252772905, 21.9587514585812, 10.0)])
@pytest.mark.unit_tests
def test_sensor_loc_inv(lon, lat, alt, valid_col, valid_row):
    """
    Test inverse localization
    """

    ___, gri = prepare_loc()

    loc = Localization(gri)
    inv_row, inv_col, h = loc.inverse(lon, lat, alt)
    assert inv_row == pytest.approx(valid_row, abs=1e-2)
    assert inv_col == pytest.approx(valid_col, abs=1e-2)
    assert h == alt


@pytest.mark.parametrize("lon,lat,alt", [(2.12026631, 31.11245154, 10.0)])
@pytest.mark.unit_tests
def test_sensor_loc_inv_vs_loc_rpc(lon, lat, alt):
    """
    Test direct localization coherence with grid and RPC
    """
    id_scene = "P1BP--2018122638935449CP"
    ___, gri = prepare_loc("ellipsoide", id_scene)
    loc_grid = Localization(gri)

    [row, col, __] = loc_grid.inverse(lon, lat, alt)
    data_folder = data_path()
    fichier_dimap = os.path.join(data_folder, f"rpc/PHRDIMAP_{id_scene}.XML")

    fctrat = RPC.from_any(fichier_dimap, topleftconvention=True)

    loc_rpc = Localization(fctrat)
    [row_rpc, col_rpc, __] = loc_rpc.inverse(lon, lat, alt)
    diff_row = row_rpc - row
    diff_col = col_rpc - col

    assert diff_row == pytest.approx(0.0, abs=1e-2)
    assert diff_col == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize(
    "col_min_valid",
    [([4.15161251e-02, 1.95057636e-01, 1.10977819e00, -8.35016563e-04, -3.50772271e-02, -9.46432481e-03])],
)
@pytest.mark.parametrize(
    "row_min_valid", [([0.05440845, 1.26513831, -0.36737151, -0.00229532, -0.07459378, -0.02558954])]
)
@pytest.mark.parametrize(
    "col_max_valid",
    [([1.76451389e-02, 2.05533045e-01, 1.11758291e00, -9.50086076e-04, -3.59923603e-02, -1.03291594e-02])],
)
@pytest.mark.parametrize(
    "row_max_valid", [([0.07565692, 1.27499912, -0.36677813, -0.00252395, -0.07539624, -0.0270914])]
)
@pytest.mark.parametrize("valid_offset_lon", [([57.37295223744326, 0.15660032225072484])])
@pytest.mark.parametrize("valid_offset_lat", [([22.066877016445275, 0.14641205050748773])])
@pytest.mark.parametrize("valid_offset_row", [([24913.0, 24912.5])])
@pytest.mark.parametrize("valid_offset_col", [([19975.5, 19975.0])])
@pytest.mark.unit_tests
def test_pred_loc_inv(
    col_min_valid,
    row_min_valid,
    col_max_valid,
    row_max_valid,
    valid_offset_lon,
    valid_offset_lat,
    valid_offset_row,
    valid_offset_col,
):
    """
    Test inverse localization
    """
    # init predictors
    ___, gri = prepare_loc()
    gri.estimate_inverse_loc_predictor()

    assert gri.pred_col_min.flatten() == pytest.approx(col_min_valid, abs=1e-6)
    assert gri.pred_row_min.flatten() == pytest.approx(row_min_valid, abs=1e-6)
    assert gri.pred_col_max.flatten() == pytest.approx(col_max_valid, abs=1e-6)
    assert gri.pred_row_max.flatten() == pytest.approx(row_max_valid, abs=1e-6)
    assert gri.pred_ofset_scale_lon == pytest.approx(valid_offset_lon, abs=1e-12)
    assert gri.pred_ofset_scale_lat == pytest.approx(valid_offset_lat, abs=1e-12)
    assert gri.pred_ofset_scale_row == pytest.approx(valid_offset_row, abs=1e-6)
    assert gri.pred_ofset_scale_col == pytest.approx(valid_offset_col, abs=1e-6)


@pytest.mark.parametrize("col,row", [(50.5, 100.5)])
@pytest.mark.parametrize("valid_lon,valid_lat,valid_alt", [(57.21700367698209, 21.95912227930429, 166.351227229112)])
@pytest.mark.unit_tests
def test_loc_intersection(row, col, valid_lon, valid_lat, valid_alt):
    """
    Test direct localization intersection function
    """
    dtmbsq, gri = prepare_loc()

    visee = np.zeros((3, gri.nbalt))
    vislonlat = gri.interpolate_grid_in_plani(row, col)
    visee[0, :] = vislonlat[0]
    visee[1, :] = vislonlat[1]
    visee[2, :] = gri.alts_down
    visee = visee.T
    (__, __, point_b, alti) = dtmbsq.intersect_dtm_cube(visee)
    (__, __, point_dtm) = dtmbsq.intersection(visee, point_b, alti)
    assert point_dtm[0] == pytest.approx(valid_lon, abs=1e-12)
    assert point_dtm[1] == pytest.approx(valid_lat, abs=1e-12)
    assert point_dtm[2] == pytest.approx(valid_alt, abs=1e-10)


@pytest.mark.parametrize("col,row,h", [(20.5, 150.5, 10.0)])
@pytest.mark.unit_tests
def test_loc_dir_loc_inv(row, col, h):
    """
    Test direct localization followed by inverse one
    """
    ___, gri = prepare_loc()
    # init predictors
    gri.estimate_inverse_loc_predictor()
    lonlatalt = gri.direct_loc_h(row, col, h)
    inv_row, inv_col, h = gri.inverse_loc(lonlatalt[0][0], lonlatalt[0][1], lonlatalt[0][2])

    assert row == pytest.approx(inv_row, abs=1e-2)
    assert col == pytest.approx(inv_col, abs=1e-2)


# delta vt 0.5 pixel shift between physical model and rpc OTB
@pytest.mark.parametrize(
    "id_scene, rpc, col,row, h",
    [
        ("P1BP--2018122638935449CP", "PHRDIMAP_P1BP--2018122638935449CP.XML", 150.5, 20.5, 10.0),
        ("P1BP--2017092838284574CP", "RPC_PHR1B_P_201709281038045_SEN_PRG_FC_178608-001.XML", 150.5, 20.5, 10.0),
    ],
)
@pytest.mark.unit_tests
def test_loc_dir_loc_inv_rpc(id_scene, rpc, row, col, h):
    """
    Test direct localization followed by inverse one
    """
    ___, gri = prepare_loc("ellipsoide", id_scene)
    # init predictors
    gri.estimate_inverse_loc_predictor()
    lonlatalt = gri.direct_loc_h(row, col, h)

    data_folder = data_path()
    fichier_dimap = os.path.join(data_folder, "rpc", rpc)

    fctrat = RPC.from_any(fichier_dimap, topleftconvention=True)
    (inv_row, inv_col, __) = fctrat.inverse_loc(lonlatalt[0][0], lonlatalt[0][1], lonlatalt[0][2])
    assert row == pytest.approx(inv_row, abs=1e-2)
    assert col == pytest.approx(inv_col, abs=1e-2)


@pytest.mark.parametrize("l0_src,c0_src, steprow_src, stepcol_src,nbrow_src,nbcol_src", [(0.5, 1.5, 10, 100, 20, 20)])
@pytest.mark.parametrize("col,row", [(1, 3)])
@pytest.mark.unit_tests
def test_coloc(l0_src, c0_src, steprow_src, stepcol_src, nbrow_src, nbcol_src, row, col):
    """
    Test coloc function
    """
    dtmbsq, gri = prepare_loc()
    gri.estimate_inverse_loc_predictor()

    gricol = coloc(gri, gri, dtmbsq, [l0_src, c0_src], [steprow_src, stepcol_src], [nbrow_src, nbcol_src])

    assert gricol[0, row, col] == pytest.approx(row * steprow_src + l0_src, 1e-6)
    assert gricol[1, row, col] == pytest.approx(col * stepcol_src + c0_src, 1e-6)


@pytest.mark.parametrize("col,lig,h", [(1000.5, 1500.5, 10.0)])
@pytest.mark.unit_tests
def test_loc_dir_loc_inv_couple(lig, col, h):
    """
    Test direct localization followed by inverse one for phr couple
    """
    id_scene_right = "P1BP--2017092838319324CP"
    ___, gri_right = prepare_loc("ellipsoide", id_scene_right)
    id_scene_left = "P1BP--2017092838284574CP"
    ___, gri_left = prepare_loc("ellipsoide", id_scene_left)
    # init predictors
    gri_right.estimate_inverse_loc_predictor()
    lonlatalt = gri_left.direct_loc_h(lig, col, h)
    inv_lig, inv_col, __ = gri_right.inverse_loc(lonlatalt[0][0], lonlatalt[0][1], lonlatalt[0][2])

    print(f"lig {inv_lig} col {inv_col}")
    # assert(lig == pytest.approx(inv_lig,abs=1e-2))
    # assert(col == pytest.approx(inv_col,abs=1e-2))
    # assert(valid == 1)


@pytest.mark.parametrize("col,row,alt", [(600, 200, 125)])
def test_colocalization(col, row, alt):
    """
    Test colocalization function using rpc
    """

    data_folder = data_path()
    id_scene = "P1BP--2018122638935449CP"
    file_dimap = os.path.join(data_folder, f"rpc/PHRDIMAP_{id_scene}.XML")
    fctrat = RPC.from_dimap_v1(file_dimap)

    row_coloc, col_coloc, _ = coloc_rpc(fctrat, fctrat, row, col, alt)

    assert row == pytest.approx(row_coloc, abs=1e-1)
    assert col == pytest.approx(col_coloc, abs=1e-1)


@pytest.mark.parametrize("col,row,h", [(500.0, 200.0, 100.0)])
@pytest.mark.unit_tests
def test_sensor_coloc_using_geotransform(col, row, h):
    """
    Test direct localization using image geotransform
    """
    data_left = os.path.join(data_path(), "rectification", "left_image")
    geom_model_left = RPC.from_any(data_left + ".geom", topleftconvention=True)
    image_filename_left = os.path.join(data_path(), "image/phr_ventoux/", "left_image_pixsize_0_5.tif")
    image_left = Image(image_filename_left)

    data_right = os.path.join(data_path(), "rectification", "right_image")
    geom_model_right = RPC.from_any(data_right + ".geom", topleftconvention=True)
    image_filename_right = os.path.join(data_path(), "image/phr_ventoux/", "right_image_pixsize_0_5.tif")
    image_right = Image(image_filename_right)

    row_coloc, col_coloc, _ = coloc_rpc(
        geom_model_left, geom_model_right, row, col, h, image_left, image_right, using_geotransform=True
    )
    origin_left = [5000.0, 5000.0]
    pix_size_left = [0.5, 0.5]
    row_phys = origin_left[0] + (row + 0.5) * pix_size_left[0]
    col_phys = origin_left[1] + (col + 0.5) * pix_size_left[1]
    loc_left = Localization(geom_model_left, elevation=None, image=image_left)
    lonlath = loc_left.direct(row_phys, col_phys, h)
    loc_right = Localization(geom_model_right, elevation=None, image=image_right)
    row_inv, col_inv, h = loc_right.inverse(lonlath[0][0], lonlath[0][1], lonlath[0][2])
    origin_right = [5162.0, 4915.0]
    pix_size_right = [0.5, 0.5]
    row_index = (row_inv - origin_right[0]) / pix_size_right[0] - 0.5
    col_index = (col_inv - origin_right[1]) / pix_size_right[1] - 0.5
    assert row_coloc == row_index
    assert col_coloc == col_index


@pytest.mark.parametrize("col,row", [(500.0, 200.0)])
@pytest.mark.unit_tests
def test_sensor_loc_utm(col, row):
    """
    Test direct localization using image geotransform
    """
    data_left = os.path.join(data_path(), "rectification", "left_image")
    geom_model = RPC.from_any(data_left + ".geom", topleftconvention=True)
    epsg = 32631
    loc_wgs = Localization(geom_model)
    loc_utm = Localization(geom_model, epsg=epsg)
    lonlath = loc_wgs.direct(np.array([row, row]), np.array([col, col]))
    coord_utm = coordinates_conversion(lonlath, geom_model.epsg, epsg)
    inv_row, inv_col, __ = loc_utm.inverse(coord_utm[:, 0], coord_utm[:, 1])
    assert row == pytest.approx(inv_row[0], abs=1e-8)
    assert col == pytest.approx(inv_col[0], abs=1e-8)

    xyh = loc_utm.direct(np.array([row]), np.array([col]), np.array([10.0]))
    assert xyh[0][0] == pytest.approx(6.72832643e05, abs=1e-3)
    assert xyh[0][1] == pytest.approx(4.899552978865e06, abs=1e-3)


@pytest.mark.unit_tests
def test_sensor_loc_dir_dtm_multi_points():
    """
    Test direct localization on DTMIntersection
    """

    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))

    geom_model = RPC.from_any(os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True)

    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)

    loc = Localization(geom_model, image=left_im, elevation=dtm_ventoux)

    row = np.array([100.0, 200.0])
    col = np.array([10.0, 20.5])
    points = loc.direct(row, col)
    print(points)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_loc_direct_pyrugged(col,row):
    """
    Test direct localization with pyrugged_geom
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.direct([row], [col], h=None, using_geotransform=False).transpose()

    coords_ref =location_dimap.location.direct_location([row], [col],None,"sensor_a")
    print("coords : ",coords)
    print("coords_ref : ",np.degrees(np.array(coords_ref)))

    assert coords[0] == pytest.approx(np.degrees(np.array(coords_ref[0])), abs=0)
    assert coords[1] == pytest.approx(np.degrees(np.array(coords_ref[1])), abs=0)
    assert coords[2] == pytest.approx(np.array(coords_ref[2]), abs=0)


@pytest.mark.parametrize("col,row,alt", [(5000.0, 5000.0, 10)])
@pytest.mark.unit_tests
def test_loc_direct_pyrugged_alt(col,row,alt):
    """
    Test direct localization with pyrugged_geom with altitude
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=alt,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.direct([row], [col], h=alt, using_geotransform=False).transpose()

    coords_ref =location_dimap.location.direct_location([row], [col], [alt],"sensor_a")

    print(abs(np.degrees(np.array(coords_ref[0:2]))-coords[0:2]))
    assert coords[0:2] == pytest.approx(np.degrees(np.array(coords_ref[0:2])), abs=0)


@pytest.mark.parametrize("lon,lat,alt", [(144.86952644029716, -37.78510620778207, 0.0)]) #degrees
@pytest.mark.unit_tests
def test_loc_inverse_pyrugged(lon,lat,alt):
    """
    Test inverse localization with pyrugged_geom
    """
    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    
    image_left = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image_left, epsg=4326)
    coords = localization.inverse([lon],[lat],[alt], using_geotransform=False)

    pyrugged = location_dimap.location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)
    coords_ref = location_dimap.location.inverse_location(min_line, max_line, np.radians([lat]), np.radians([lon]), [alt], "sensor_a")

    print("coords[0:2] : ",coords[0:2])
    print("coords_ref : ",coords_ref)

    assert coords[0] == pytest.approx(coords_ref[0], abs=0)
    assert coords[1] == pytest.approx(coords_ref[1], abs=0)

@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_colocalization_pyrugged_identity(col, row):
    """
    Test colocalization function using Pyrurgged ( "coloc_rpc" -> use also pyrugged)
    """

    file_dimap="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    geom = Pyrugged_geom(file_dimap,None,None,None)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )

    row_coloc, col_coloc, _ = coloc_rpc(geom, geom, row, col, location_dimap)
    print("row",np.amax(abs(np.array(row_coloc)-np.array(row))))
    print("col",np.amax(abs(np.array(col_coloc)-np.array(col))))
    assert row == pytest.approx(row_coloc, abs=1e-6) #coloc précise à 1e-6
    assert col == pytest.approx(col_coloc, abs=1e-8)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_sensor_coloc_pyrugged(col, row):
    """
    Test direct colocalization using image geotransform
    """

    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,None,None,None)
    geom_right = Pyrugged_geom(file_dimap_right,None,None,None)

    elev_left = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom_left.light_time,
            aberration_light = geom_left.aberration_light,
            atmospheric_refraction = geom_left.atmospheric_refraction,
            )

    elev_right = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )

    #called "coloc_rpc" but it use the shareloc colo function workibng with rpc and pyrugged

    row_coloc, col_coloc, _ = coloc_rpc(
        geom_left, geom_right, row, col, elev_left, left_im, right_im, using_geotransform=True,elevation2=elev_right
    )


    row, col = left_im.transform_index_to_physical_point(row, col)
    direct_ref =elev_left.location.direct_location([row], [col],None,"sensor_a") #line,pix -> lon,lat,alt

    pyrugged = elev_right.location.rugged
    line_sensor = pyrugged.sensors["sensor_b"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)
    coloc_ref = elev_right.location.inverse_location(min_line, max_line, direct_ref[1], direct_ref[0], direct_ref[2], "sensor_b")#lat,lon,alt -> line,pix
    row_ref, col_ref = right_im.transform_physical_point_to_index(np.array(coloc_ref[0]), np.array(coloc_ref[1]))


    print('row_coloc',row_coloc)
    print('type(row_coloc)',type(row_coloc))
    print('row_ref',row_ref)
    print('type(row_ref)',type(row_ref))

    print("row_coloc-row_ref",row_coloc-row_ref)
    print("col_coloc-col_ref",col_coloc-col_ref)

    assert row_coloc == pytest.approx(row_ref, abs=1e-8)
    assert col_coloc == pytest.approx(col_ref, abs=1e-8)



@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_sensor_coloc_pyrugged_dem(col, row):
    """
    Test direct colocalization using image geotransform
    """

    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,None,None,None)
    geom_right = Pyrugged_geom(file_dimap_right,None,None,None)

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90"
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")

    elev_left = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_left.light_time,
            aberration_light = geom_left.aberration_light,
            atmospheric_refraction = geom_left.atmospheric_refraction,
            )

    elev_right = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )

    #called "coloc_rpc" but it use the shareloc coloc function workibng with rpc and pyrugged
    row_coloc, col_coloc, _ = coloc_rpc(
        geom_left, geom_right, row, col, elev_left, left_im, right_im, using_geotransform=True,elevation2=elev_right
    )

    row, col = left_im.transform_index_to_physical_point(row, col)
    direct_ref =elev_left.location.direct_location([row], [col], None,"sensor_a") #line,pix -> lon,lat,alt

    pyrugged = elev_right.location.rugged
    line_sensor = pyrugged.sensors["sensor_b"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)
    coloc_ref = elev_right.location.inverse_location(min_line, max_line, direct_ref[1], direct_ref[0], direct_ref[2], "sensor_b")#lat,lon,alt -> line,pix
    row_ref, col_ref = right_im.transform_physical_point_to_index(np.array(coloc_ref[0]), np.array(coloc_ref[1]))

    assert row_coloc == row_ref
    assert col_coloc == col_ref


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_loc_direct_pyrugged_with_correction(col,row):
    """
    Test direct localization with pyrugged_geom
    """


    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    atmospheric_refraction = MultiLayerModel(ellipsoid)


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.direct([row], [col], h=None, using_geotransform=False).transpose()

    coords_ref =location_dimap.location.direct_location([row], [col], None,"sensor_a")

    assert coords[0] == pytest.approx(np.degrees(np.array(coords_ref[0])), abs=0)
    assert coords[1] == pytest.approx(np.degrees(np.array(coords_ref[1])), abs=0)
    assert coords[2] == pytest.approx(np.array(coords_ref[2]), abs=0)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_loc_direct_pyrugged_multi_alti_without_correction(col,row):
    """
    Test direct localization with pyrugged_geom
    """

    row = np.array([row]*5)
    col = np.array([col]*5)
    alt = np.array([0.0,10.0,25.3,86.5,400.2])


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,False,False,None)
    # geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.direct(row, col, h=alt, using_geotransform=False).transpose()

    coords_ref =location_dimap.location.direct_location(row, col, alt,"sensor_a")

    geom_model_RPC = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/localization/RPC_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    )
    localizationRPC = Localization(geom_model_RPC, elevation=None, image=image, epsg=4326)
    coords_ref_RPC = localizationRPC.direct(row, col, h=alt, using_geotransform=False)

    coords = coords.T
    coords_ref = np.array(coords_ref).T
    coords_ref[:,:2] = np.degrees(coords_ref[:,:2])

    diffRPC = abs(coords-coords_ref_RPC)

    print("\n\nabs(coords-coord_ref_RPC) : \n",diffRPC)
    print("abs(coords-coord_ref) : \n",abs(coords-coords_ref))

    assert coords[:,:2] == pytest.approx(coords_ref[:,:2],abs=0)
    assert coords[:,2] == pytest.approx(coords_ref[:,2],abs=0)
    assert coords[:,:2] == pytest.approx(coords_ref_RPC[:,:2],abs=1e-7)
    assert coords[:,2] == pytest.approx(coords_ref_RPC[:,2],abs=1e-3)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_loc_direct_pyrugged_multi_alti_with_correction(col,row):
    """
    Test direct localization with pyrugged_geom
    """

    row = np.array([row]*5)
    col = np.array([col]*5)
    alt = np.array([0.0,10.0,25.3,86.5,400.2])


    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    atmospheric_refraction = MultiLayerModel(ellipsoid)


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.direct(row, col, h=alt, using_geotransform=False).transpose()

    coords_ref =location_dimap.location.direct_location(row, col, alt,"sensor_a")

    geom_model_RPC = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/localization/RPC_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    )
    localizationRPC = Localization(geom_model_RPC, elevation=None, image=image, epsg=4326)
    coords_ref_RPC = localizationRPC.direct(row, col, h=alt, using_geotransform=False)

    coords = coords.T
    coords_ref = np.array(coords_ref).T
    coords_ref[:,:2] = np.degrees(coords_ref[:,:2])

    diffRPC = abs(coords-coords_ref_RPC)

    print("coords : ",coords)
    print("coords_ref_RPC : ",coords_ref_RPC)
    print("abs(coords-coord_ref_RPC) : \n",diffRPC)
    #print("abs(coords-coord_ref) : \n",abs(coords-coords_ref))

    ecart_absolu_moyen = np.sum(diffRPC,axis=0)/len(alt)
    print("écart absolui moyen : ",ecart_absolu_moyen)

    assert coords[:,:2] == pytest.approx(coords_ref[:,:2],abs=0)
    assert coords[:,2] == pytest.approx(coords_ref[:,2],abs=0)
    assert coords[:,:2] == pytest.approx(coords_ref_RPC[:,:2],abs=2e-4)
    assert coords[:,2] == pytest.approx(coords_ref_RPC[:,2],abs=1e-3)




@pytest.mark.parametrize("lon,lat", [(144.86952644029716, -37.78510620778207)]) #degrees
@pytest.mark.unit_tests
def test_loc_inverse_pyrugged_multi_alti_with_correction(lon,lat):
    """
    Test inverse localization with pyrugged_geom
    """

    lon = np.array([lon]*5)
    lat = np.array([lat]*5)
    alt = np.array([0.0,10.0,25.3,86.5,400.2])

    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    #conf = static_cfg.load_config()
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)

    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    location_dimap = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    
    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.inverse(lon,lat,alt, using_geotransform=False)

    pyrugged = location_dimap.location.rugged
    line_sensor = pyrugged.sensors["sensor_a"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)
    coords_ref = location_dimap.location.inverse_location(min_line, max_line, np.radians(lat), np.radians(lon), alt, "sensor_a")

    geom_model_RPC = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/localization/RPC_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    )
    localizationRPC = Localization(geom_model_RPC, elevation=None, image=image, epsg=4326)
    coords_ref_RPC = localizationRPC.inverse(lon, lat, h=alt, using_geotransform=False)

    print("coords[0] : ",abs(coords[0]-coords_ref[0]))
    print("coords[1] : ",abs(coords[1]-coords_ref[1]))

    print("coordsRPC[0] : ",abs(coords[0]-coords_ref_RPC[0]))
    print("coordsRPC[1] : ",abs(coords[1]-coords_ref_RPC[1]))

    #pyrugged or orekit cache make this tiny differences
    assert coords[0] == pytest.approx(coords_ref[0], abs=1e-3)
    assert coords[1] == pytest.approx(coords_ref[1], abs=1e-8)

    # 1pix =~ 0.7m ->42*0.7=29,4m -> relevant with corrections
    assert coords[0] == pytest.approx(coords_ref_RPC[0], abs=42)
    assert coords[1] == pytest.approx(coords_ref_RPC[1], abs=10)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_sensor_coloc_pyrugged_with_correction(col, row):
    """
    Test direct colocalization using image geotransform
    """


    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    #conf = static_cfg.load_config()
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)



    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,True,True,atmospheric_refraction)
    geom_right = Pyrugged_geom(file_dimap_right,True,True,atmospheric_refraction)

    elev_left = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom_left.light_time,
            aberration_light = geom_left.aberration_light,
            atmospheric_refraction = geom_left.atmospheric_refraction,
            )

    elev_right = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )
    #called "coloc_rpc" but it use the shareloc colo function workibng with rpc and pyrugged
    row_coloc, col_coloc, _ = coloc_rpc(
        geom_left, geom_right, row, col, elev_left, left_im, right_im, using_geotransform=True,elevation2=elev_right
    )


    file_dimap_left_RPC="/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right_RPC="/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    geom_left = RPC.from_any(file_dimap_left_RPC)
    geom_right = RPC.from_any(file_dimap_right_RPC)
    row_coloc_RPC, col_coloc_RPC, _ = coloc_rpc(
        geom_left, geom_right, row, col, 0.0, left_im, right_im, using_geotransform=True,elevation2=None
    )

    row, col = left_im.transform_index_to_physical_point(row, col)
    direct_ref =elev_left.location.direct_location([row], [col],None,"sensor_a") #line,pix -> lon,lat,alt
    pyrugged = elev_right.location.rugged
    line_sensor = pyrugged.sensors["sensor_b"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)
    coloc_ref = elev_right.location.inverse_location(min_line, max_line, direct_ref[1], direct_ref[0], direct_ref[2], "sensor_b")#lat,lon,alt -> line,pix
    row_ref, col_ref = right_im.transform_physical_point_to_index(np.array(coloc_ref[0]), np.array(coloc_ref[1]))


    print("diff row ref : ",np.amax(abs(np.array(row_coloc)-np.array(row_ref))))
    print("diff col ref: ",np.amax(abs(np.array(col_coloc)-np.array(col_ref))))
    print("diff row RPC: ",np.amax(abs(np.array(row_coloc)-np.array(row_coloc_RPC))))
    print("diff col RPC: ",np.amax(abs(np.array(col_coloc)-np.array(col_coloc_RPC))))

    assert row_coloc == pytest.approx(row_ref, abs=1e-5)
    assert col_coloc == pytest.approx(col_ref, abs=1e-8)
    assert row_coloc == pytest.approx(row_coloc_RPC, abs=30)
    assert col_coloc == pytest.approx(col_coloc_RPC, abs=10)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_sensor_coloc_pyrugged_dem_with_correction(col, row):
    """
    Test direct colocalization using image geotransform
    """

    # pr = cProfile.Profile()
    # pr.enable()

    row = np.array([row])
    col = np.array([col])


    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    
    #conf = static_cfg.load_config()
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)


    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,True,True,atmospheric_refraction)
    geom_right = Pyrugged_geom(file_dimap_right,True,True,atmospheric_refraction)

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90"
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")

    elev_left = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_left.light_time,
            aberration_light = geom_left.aberration_light,
            atmospheric_refraction = geom_left.atmospheric_refraction,
            )

    elev_right = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )

    #called "coloc_rpc" but it use the shareloc coloc function workibng with rpc and pyrugged
    row_coloc, col_coloc, _ = coloc_rpc(
        geom_left, geom_right, row, col, elev_left, left_im, right_im, using_geotransform=True,elevation2=elev_right
    )

    file_dimap_left_RPC="/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right_RPC="/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    geom_left = RPC.from_any(file_dimap_left_RPC)
    geom_right = RPC.from_any(file_dimap_right_RPC)
    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90/e144/s38.dt1"
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)
    row_coloc_RPC, col_coloc_RPC, _ = coloc_rpc(
        geom_left, geom_right, row, col, dtm_ventoux, left_im, right_im, using_geotransform=True,elevation2=None
    )

    row, col = left_im.transform_index_to_physical_point(row, col)
    direct_ref =elev_left.location.direct_location(row, col, None,"sensor_a") #line,pix -> lon,lat,alt
    pyrugged = elev_right.location.rugged
    line_sensor = pyrugged.sensors["sensor_b"]
    min_line = line_sensor.get_line(pyrugged.min_date)
    max_line = line_sensor.get_line(pyrugged.max_date)
    coloc_ref = elev_right.location.inverse_location(min_line, max_line, direct_ref[1], direct_ref[0], direct_ref[2], "sensor_b")#lat,lon,alt -> line,pix
    row_ref, col_ref = right_im.transform_physical_point_to_index(np.array(coloc_ref[0]), np.array(coloc_ref[1]))



    print("row_coloc : ",row_coloc)
    print("col_coloc : ",col_coloc)
    print("row_ref : ",row_ref)
    print("col_ref : ",col_ref)
    print("row_coloc_RPC : ",row_coloc_RPC)
    print("col_coloc_RPC : ",col_coloc_RPC)
    print("row : ",np.amax(abs(np.array(row_coloc)-np.array(row_ref))))
    print("col : ",np.amax(abs(np.array(col_coloc)-np.array(col_ref))))
    print("row RPC : ",np.amax(abs(np.array(row_coloc)-np.array(row_coloc_RPC))))
    print("col RPC : ",np.amax(abs(np.array(col_coloc)-np.array(col_coloc_RPC))))

    assert row_coloc == pytest.approx(row_ref, abs=1e-5)
    assert col_coloc == pytest.approx(col_ref, abs=4e-9)
    assert row_coloc == pytest.approx(row_coloc_RPC, abs=30)
    assert col_coloc == pytest.approx(col_coloc_RPC, abs=10)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_loc_pyrugged_multi_alti_with_correction_sinusoid(col,row):
    """
    Test direct localization with pyrugged_geom
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
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)


    file_dimap="/new_cars/shareloc/tests/data/pyrugged/localization/DIM_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001.XML"
    geom = Pyrugged_geom(file_dimap,True,True,atmospheric_refraction)

    abs_date = PHRParser(file_dimap,"just_for_time_ref").start_time
    amp = 0.2e-6
    freq = 30
    phase = 0

    location_dimap = Location(
            dimap=geom.dimap,
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
    
    # Same parametre as pydimap test localization
    t_pertub_max = 1 / (4 * freq) + 1 / freq  # 1 period margin
    rugged = location_dimap.location.rugged
    rate = rugged.sensors["sensor_a"].get_rate()
    line_pert_max = t_pertub_max * rate
    line_pert_max = int(line_pert_max)
    lines = np.array(list(range(line_pert_max - 500, line_pert_max + 501)))  # len=1001
    pixels = np.array(list(range(0, 1001)))
    alt = np.array(list(range(0, 1001)))

    image = Image("/new_cars/shareloc/tests/data/pyrugged/localization/IMG_PHR1A_P_201202250025599_SEN_PRG_FC_5847-001_R1C1.TIFF")
    localization = Localization(geom, elevation=location_dimap, image=image, epsg=4326)
    coords = localization.direct(lines, pixels, h=alt, using_geotransform=False)

    location_dimap_ref = Location(
            dimap=geom.dimap,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=0.0,
            light_time = geom.light_time,
            aberration_light = geom.aberration_light,
            atmospheric_refraction = geom.atmospheric_refraction,
            )
    
    localization_ref = Localization(geom, elevation=location_dimap_ref, image=image, epsg=4326)
    coords_ref = localization_ref.direct(lines, pixels, h=alt, using_geotransform=False)



    diffref = abs(coords-coords_ref)
    # print("diff lon m : ",np.amax(diffref[:,0])/8.9e-6)#only with PHR
    # print("diff lat m : ",np.amax(diffref[:,1])/8.9e-6)
    # print("diff alt m : ",np.amax(diffref[:,2]))

    # PHR 1 m lon/lat ->8.9e-3 deg
    # 0.2e-6rad satellite LOS -> 0.139m= decalage max (doit être atteint)
    # tout ça ne sont que des approximation grossières
    assert np.amax(diffref[:,1])/8.9e-6 == pytest.approx(0.139,abs=0.01)# tanguage donc assert pertinent que sur les lignes/lat
    assert coords[:,2] == pytest.approx(coords_ref[:,2],abs=1e-5)



    # ASSERT INVERSE LOC
    # pert max at index 501
    lines,pixels,altitudes = localization.inverse(coords_ref[450:551,0],coords_ref[450:551,1],coords_ref[450:551,2], using_geotransform=False)
    lines_ref,pixels_ref,altitudes_ref = localization_ref.inverse(coords_ref[450:551,0],coords_ref[450:551,1],coords_ref[450:551,2], using_geotransform=False)

    # print("diff inv lines : ",np.amax(abs(lines_ref-lines)))
    # print("diff inv pixels : ",np.amax(abs(pixels_ref-pixels)))
    # print("diff inv altitudes : ",np.amax(abs(altitudes_ref-altitudes)))

    # diff inv lines :  0.28407875711582165
    # diff inv pixels :  0.0002224332015998698
    # diff inv altitudes :  0.0

    #same tolerance as compute coloc grid test pydimap
    assert lines_ref == pytest.approx(lines_ref,abs=0.29)
    assert pixels == pytest.approx(pixels_ref,abs=1e-3)
    assert altitudes == pytest.approx(altitudes_ref,abs=0)


@pytest.mark.parametrize("col,row", [(5000.0, 5000.0)])
@pytest.mark.unit_tests
def test_sensor_coloc_pyrugged_dem_with_correction_sinusoid(col, row):
    """
    Test direct colocalization using image geotransform
    """

    init_orekit()

    row = np.array([row]*5)
    col = np.array([col]*5)
    alt = np.array([0.0,10.2,23.6,55.9,289.2])


    ellipsoid_id=EllipsoidId.WGS84
    body_rotating_frame_id=BodyRotatingFrameId.ITRF
    new_ellipsoid = select_ellipsoid(ellipsoid_id, select_body_rotating_frame(body_rotating_frame_id))

    ellipsoid = ExtendedEllipsoid(
        new_ellipsoid.equatorial_radius,
        new_ellipsoid.flattening,
        new_ellipsoid.body_frame,
        )
    
    #conf = static_cfg.load_config()
    px_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_PIXEL_STEP]=100
    line_step = 1000#conf[static_cfg.ATMOSPHERIC_GRID_LINE_STEP]
    atmospheric_refraction = MultiLayerModel(ellipsoid)
    atmospheric_refraction.set_grid_steps(px_step, line_step)


    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,True,True,atmospheric_refraction)
    geom_right = Pyrugged_geom(file_dimap_right,True,True,atmospheric_refraction)

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90"
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")



    elev_left = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_left.light_time,
            aberration_light = geom_left.aberration_light,
            atmospheric_refraction = geom_left.atmospheric_refraction,
            )


    abs_date = PHRParser(file_dimap_right,"just_for_time_ref").start_time
    amp = 0.2e-6
    freq = 30
    phase = 0
    elev_right = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            transforms=[SinusoidalRotation("sinRot", Vector3D.PLUS_J, abs_date, amp, freq, phase)],
            )

    #called "coloc_rpc" but it use the shareloc coloc function workibng with rpc and pyrugged
    row_coloc, col_coloc, _ = coloc_rpc(
        geom_left, geom_right, row, col, elev_left, left_im, right_im, using_geotransform=True,elevation2=elev_right,altitude=alt
    )

    elev_right_ref = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            transforms=[SinusoidalRotation("sinRot", Vector3D.PLUS_J, abs_date, 0, freq, phase)],
            )

    row_coloc_ref, col_coloc_ref, _ = coloc_rpc(
        geom_left, geom_right, row, col, elev_left, left_im, right_im, using_geotransform=True,elevation2=elev_right_ref,altitude=alt
    )

    print("row_coloc : ",row_coloc)
    print("col_coloc : ",col_coloc)
    print("row_coloc_ref : ",row_coloc_ref)
    print("col_coloc_ref : ",col_coloc_ref)

    print("row : ",np.amax(abs(np.array(row_coloc)-np.array(row_coloc_ref))))
    print("col : ",np.amax(abs(np.array(col_coloc)-np.array(col_coloc_ref))))
 

    assert row_coloc == pytest.approx(row_coloc_ref, abs=0.29)
    assert col_coloc == pytest.approx(col_coloc_ref, abs=1e-3)