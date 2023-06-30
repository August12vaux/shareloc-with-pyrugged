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
Test module for rectification grid interpolation class shareloc/geofunctions/rectification*.py
Ground truth references (gt_{left/right}_grid*.tif) have been generated using OTB StereoRectificationGridGenerator
application.
"""

# Standard imports
import math
import os

# Third party imports
import numpy as np
import pytest
import rasterio
import pdb

# Shareloc imports

from pyrugged.configuration.init_orekit import init_orekit
from org.hipparchus.geometry.euclidean.threed import Vector3D
from pydimaprugged.dimap_parser.phr_parser import PHRParser
from pyrugged.los.sinusoidal_rotation import SinusoidalRotation


from shareloc.geofunctions.dtm_intersection import DTMIntersection
from shareloc.geofunctions.rectification import (  write_epipolar_grid,
    compute_epipolar_angle,
    compute_stereorectification_epipolar_grids,
    get_epipolar_extent,
    moving_along_lines,
    moving_to_next_line,
    prepare_rectification,
)
from shareloc.geofunctions.rectification_grid import RectificationGrid
from shareloc.geomodels.grid import Grid
from shareloc.geomodels.rpc import RPC
from shareloc.geomodels.pyrugged_geom import Pyrugged_geom
from shareloc.image import Image
from shareloc.location_pydimap import Location
from pyrugged.refraction.multi_layer_model import MultiLayerModel
from pyrugged.bodies.extended_ellipsoid import ExtendedEllipsoid
from pyrugged.bodies.body_rotating_frame_id import BodyRotatingFrameId
from pyrugged.model.pyrugged_builder import select_ellipsoid, select_body_rotating_frame
from pyrugged.bodies.ellipsoid_id import EllipsoidId

# Shareloc test imports
from tests.helpers import data_path


@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_rpc():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: RPC
    Earth elevation: default to 0.0
    """
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    right_im = Image(os.path.join(data_path(), "rectification", "right_image.tif"))

    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    epi_step = 30
    elevation_offset = 50
    default_elev = 0.0
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, default_elev, epi_step, elevation_offset
    )

    # OTB reference
    reference_left_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid.tif")).read()
    reference_right_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_right_grid.tif")).read()

    # update baseline
    # write_epipolar_grid(left_grid, os.path.join(data_path(),'grid_left_elev_0.tif'))
    # write_epipolar_grid(right_grid, os.path.join(data_path(),'grid_right_elev_0.tif'))

    # Check epipolar grids
    # OTB convention is [col, row], shareloc convention is [row, col]
    assert reference_left_grid[1] == pytest.approx(left_grid.data[0, :, :], abs=1e-2)
    assert reference_left_grid[0] == pytest.approx(left_grid.data[1, :, :], abs=1e-2)

    assert reference_right_grid[1] == pytest.approx(right_grid.data[0, :, :], abs=1e-2)
    assert reference_right_grid[0] == pytest.approx(right_grid.data[1, :, :], abs=1e-2)

    # Check size of rectified images
    assert img_size_row == 612
    assert img_size_col == 612

    # Check mean_baseline_ratio
    # ground truth mean baseline ratio from OTB
    referecne_mean_br = 0.704004705
    assert mean_br == pytest.approx(referecne_mean_br, abs=1e-5)


@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_rpc_alti():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: RPC
    Earth elevation: alti=100
    """
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    right_im = Image(os.path.join(data_path(), "rectification", "right_image.tif"))

    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    epi_step = 30
    elevation_offset = 50
    default_elev = 100.0
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, default_elev, epi_step, elevation_offset
    )

    reference_left_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid_100.tif")).read()
    reference_right_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_right_grid_100.tif")).read()

    # Check epipolar grids
    # OTB convention is [col, row], shareloc convention is [row, col]
    assert reference_left_grid[1] == pytest.approx(left_grid.data[0, :, :], abs=1e-2)
    assert reference_left_grid[0] == pytest.approx(left_grid.data[1, :, :], abs=1e-2)

    assert reference_right_grid[1] == pytest.approx(right_grid.data[0, :, :], abs=1e-2)
    assert reference_right_grid[0] == pytest.approx(right_grid.data[1, :, :], abs=1e-2)

    # Check size of rectified images
    assert img_size_row == 612
    assert img_size_col == 612

    # Check mean_baseline_ratio
    # ground truth mean baseline ratio from OTB
    reference_mean_br = 0.7039927244
    assert mean_br == pytest.approx(reference_mean_br, abs=1e-5)


@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_rpc_dtm_geoid():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: RPC
    Earth elevation: SRTM DTM + Geoid egm96_15
    """

    # first instantiate geometric models left and right (here RPC geometrics model)
    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    # read the images
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    right_im = Image(os.path.join(data_path(), "rectification", "right_image.tif"))

    # we use DTM and Geoid: a DTMIntersection class has to be used
    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)

    # compute rectification grid sampled at 30 pixels
    epi_step = 30
    elevation_offset = 50
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, dtm_ventoux, epi_step, elevation_offset
    )

    # evaluate the results by comparison with OTB
    reference_left_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid_dtm.tif")).read()
    reference_right_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_right_grid_dtm.tif")).read()

    # baseline update if necessary
    # write_epipolar_grid(left_grid, os.path.join(data_path(),'grid_left_dtm.tif'))
    # write_epipolar_grid(right_grid, os.path.join(data_path(),'grid_right_dtm.tif'))

    # Check epipolar grids
    # OTB convention is [col, row], shareloc convention is [row, col]
    assert reference_left_grid[1] == pytest.approx(left_grid.data[0, :, :], abs=1e-2)
    assert reference_left_grid[0] == pytest.approx(left_grid.data[1, :, :], abs=1e-2)

    assert reference_right_grid[1] == pytest.approx(right_grid.data[0, :, :], abs=1e-2)
    assert reference_right_grid[0] == pytest.approx(right_grid.data[1, :, :], abs=1e-2)

    # Check size of rectified images
    assert img_size_row == 612
    assert img_size_col == 612

    # Check mean_baseline_ratio
    # ground truth mean baseline ratio from OTB
    reference_mean_br = 0.7039416432
    assert mean_br == pytest.approx(reference_mean_br, abs=1e-5)


@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_rpc_dtm_geoid_roi():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: RPC
    Earth elevation ROI: SRTM DTM + Geoid egm96_15
    """
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    right_im = Image(os.path.join(data_path(), "rectification", "right_image.tif"))

    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    extent = get_epipolar_extent(left_im, geom_model_left, geom_model_right, margin=0.0016667)
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file, roi=extent)

    epi_step = 30
    elevation_offset = 50
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, dtm_ventoux, epi_step, elevation_offset
    )

    reference_left_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid_dtm.tif")).read()
    reference_right_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_right_grid_dtm.tif")).read()

    # update baseline
    # write_epipolar_grid(left_grid, os.path.join(data_path(),'grid_left_dtm_roi.tif'))
    # write_epipolar_grid(right_grid, os.path.join(data_path(),'grid_right_dtm_roi.tif'))

    # Check epipolar grids
    # OTB convention is [col, row], shareloc convention is [row, col]

    assert reference_left_grid[1] == pytest.approx(left_grid.data[0, :, :], abs=1e-2)
    assert reference_left_grid[0] == pytest.approx(left_grid.data[1, :, :], abs=1e-2)
    assert reference_right_grid[1] == pytest.approx(right_grid.data[0, :, :], abs=1e-2)
    assert reference_right_grid[0] == pytest.approx(right_grid.data[1, :, :], abs=1e-2)

    # Check size of rectified images
    assert img_size_row == 612
    assert img_size_col == 612

    # Check mean_baseline_ratio
    # ground truth mean baseline ratio from OTB
    reference_mean_br = 0.7039416432
    assert mean_br == pytest.approx(reference_mean_br, abs=1e-5)


@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_grid():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio


    Input Geomodels: Grid
    Earth elevation: default to 0.0
    """

    # first instantiate geometric models left and right (here Grid geometric model)
    geom_model_left = Grid(
        os.path.join(data_path(), "grid/phr_ventoux/GRID_PHR1B_P_201308051042194_SEN_690908101-001.tif")
    )
    geom_model_right = Grid(
        os.path.join(data_path(), "grid/phr_ventoux/GRID_PHR1B_P_201308051042523_SEN_690908101-002.tif")
    )

    # read the images
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    right_im = Image(os.path.join(data_path(), "rectification", "right_image.tif"))

    default_elev = 0.0

    # compute rectification grid sampled at 30 pixels
    epi_step = 30
    elevation_offset = 50
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, default_elev, epi_step, elevation_offset
    )

    # OTB reference
    reference_left_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid.tif")).read()
    reference_right_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_right_grid.tif")).read()

    # update baseline
    # write_epipolar_grid(left_grid, os.path.join(data_path(),'grid_left_elev_0.tif'))
    # write_epipolar_grid(right_grid, os.path.join(data_path(),'grid_right_elev_0.tif'))

    # Check epipolar grids
    # OTB convention is [col, row], shareloc convention is [row, col]
    assert reference_left_grid[1] == pytest.approx(left_grid.data[0, :, :], abs=1.2e-2)
    assert reference_left_grid[0] == pytest.approx(left_grid.data[1, :, :], abs=1.2e-2)

    assert reference_right_grid[1] == pytest.approx(right_grid.data[0, :, :], abs=1.2e-2)
    assert reference_right_grid[0] == pytest.approx(right_grid.data[1, :, :], abs=1.2e-2)

    # Check size of rectified images
    assert img_size_row == 612
    assert img_size_col == 612

    # Check mean_baseline_ratio
    # ground truth mean baseline ratio from OTB
    referecne_mean_br = 0.704004705
    assert mean_br == pytest.approx(referecne_mean_br, abs=1e-4)


@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_grid_dtm_geoid():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Grids
    Earth elevation: SRTM DTM + Geoid egm96_15
    """

    # first instantiate geometric models left and right (here Grid geometrics model)
    geom_model_left = Grid(
        os.path.join(data_path(), "grid/phr_ventoux/GRID_PHR1B_P_201308051042194_SEN_690908101-001.tif")
    )
    geom_model_right = Grid(
        os.path.join(data_path(), "grid/phr_ventoux/GRID_PHR1B_P_201308051042523_SEN_690908101-002.tif")
    )

    # read the images
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    right_im = Image(os.path.join(data_path(), "rectification", "right_image.tif"))

    # we use DTM and Geoid a DTMIntersection class has to be used
    dtm_file = os.path.join(data_path(), "dtm", "srtm_ventoux", "srtm90_non_void_filled", "N44E005.hgt")
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file, fill_nodata="mean")

    # compute rectification grid sampled at 30 pixels
    epi_step = 30
    elevation_offset = 50
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, dtm_ventoux, epi_step, elevation_offset
    )

    # evaluate the results by comparison with OTB
    reference_left_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid_dtm.tif")).read()
    reference_right_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_right_grid_dtm.tif")).read()

    # baseline update if necessary
    # write_epipolar_grid(left_grid, os.path.join(data_path(),'grid_left_dtm.tif'))
    # write_epipolar_grid(right_grid, os.path.join(data_path(),'grid_right_dtm.tif'))

    # Check epipolar grids
    # OTB convention is [col, row], shareloc convention is [row, col]
    assert reference_left_grid[1] == pytest.approx(left_grid.data[0, :, :], abs=1e-2)
    assert reference_left_grid[0] == pytest.approx(left_grid.data[1, :, :], abs=1e-2)

    assert reference_right_grid[1] == pytest.approx(right_grid.data[0, :, :], abs=1e-2)
    assert reference_right_grid[0] == pytest.approx(right_grid.data[1, :, :], abs=1e-2)

    # Check size of rectified images
    assert img_size_row == 612
    assert img_size_col == 612

    # Check mean_baseline_ratio
    # ground truth mean baseline ratio from OTB
    reference_mean_br = 0.7039416432
    assert mean_br == pytest.approx(reference_mean_br, abs=1e-4)


@pytest.mark.parametrize("row,col", [(15, 0)])
@pytest.mark.unit_tests
def test_rectification_grid_interpolation_one_point(row, col):
    """
    Test interpolation on rectification grid
    """
    id_scene_right = "P1BP--2017092838319324CP"
    grid_filename = os.path.join(data_path(), "rectification_grids", f"grid_{id_scene_right}.tif")
    rectif_grid = RectificationGrid(grid_filename)
    # value at position [15,15]
    value_row = np.sum(rectif_grid.row_positions[0, 0:2]) / 2.0
    value_col = np.sum(rectif_grid.col_positions[0, 0:2]) / 2.0
    coords = rectif_grid.interpolate((col, row))
    assert value_row == pytest.approx(coords[0, 1], abs=1e-4)
    assert value_col == pytest.approx(coords[0, 0], abs=1e-4)


@pytest.mark.unit_tests
def test_rectification_grid_interpolation():
    """
    Test interpolation on rectification grid
    """
    id_scene_right = "P1BP--2017092838319324CP"
    grid_filename = os.path.join(data_path(), "rectification_grids", f"grid_{id_scene_right}.tif")

    rectif_grid = RectificationGrid(grid_filename)
    # value at position [15,15]

    value_row = np.sum(rectif_grid.row_positions[0:2, 0:2]) / 4.0
    value_col = np.sum(rectif_grid.col_positions[0:2, 0:2]) / 4.0
    sensor_positions = np.zeros((2, 2))
    sensor_positions[0, :] = [15.0, 15.0]
    sensor_positions[1, :] = [0.0, 0.0]
    coords = rectif_grid.interpolate(sensor_positions)
    assert value_col == pytest.approx(coords[0, 0], abs=1e-4)
    assert value_row == pytest.approx(coords[0, 1], abs=1e-4)


@pytest.mark.unit_tests
def test_rectification_grid_extrapolation():
    """
    Test interpolation on rectification grid
    """
    grid_filename = os.path.join(data_path(), "rectification_grids", "left_epipolar_grid_ventoux.tif")
    rectif_grid = RectificationGrid(grid_filename)

    sensor_positions = np.zeros((2, 2))
    sensor_positions[0, :] = [30.0, -10.0]
    sensor_positions[1, :] = [30.0, 691.0]
    coords = rectif_grid.interpolate(sensor_positions)
    assert pytest.approx(coords[0, 1], abs=1e-10) == 5065.72347005208303016
    assert pytest.approx(coords[1, 1], abs=1e-10) == 4883.84894205729142413474619389


@pytest.mark.unit_tests
def test_prepare_rectification():
    """
    Test prepare rectification : check grids size, epipolar image size, and left epipolar starting point
    """
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))

    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    epi_step = 30
    elevation_offset = 50
    default_elev = 0.0
    __, grid_size, rectified_image_size, footprint = prepare_rectification(
        left_im, geom_model_left, geom_model_right, default_elev, epi_step, elevation_offset
    )

    # check size of the epipolar grids
    assert grid_size[0] == 22
    assert grid_size[1] == 22

    # check size of rectified images
    assert rectified_image_size[0] == 612
    assert rectified_image_size[1] == 612

    # check the first epipolar point in the left image
    # ground truth values from OTB
    otb_output_origin_in_left_image = [5625.78139593690139008685946465, 5034.15635707952696975553408265, 0]

    # OTB convention is [col, row, altitude], shareloc convention is [row, col, altitude]
    assert otb_output_origin_in_left_image[1] == pytest.approx(footprint[0][0], abs=1e-5)
    assert otb_output_origin_in_left_image[0] == pytest.approx(footprint[0][1], abs=1e-5)

    assert footprint[0][2] == otb_output_origin_in_left_image[2]


@pytest.mark.unit_tests
def test_prepare_rectification_footprint():
    """
    Test prepare rectification : check footprint
    """
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))

    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    epi_step = 30
    elevation_offset = 50
    default_elev = 0.0
    _, _, _, footprint = prepare_rectification(
        left_im, geom_model_left, geom_model_right, default_elev, epi_step, elevation_offset
    )

    ground_truth = np.array(
        [
            [5034.15635485, 5625.78139208, 0.0],
            [5654.75411307, 5459.06023724, 0.0],
            [5488.03295823, 4838.46247902, 0.0],
            [4867.43520001, 5005.18363386, 0.0],
        ]
    )

    assert np.all(ground_truth == pytest.approx(footprint, abs=1e-5))


@pytest.mark.unit_tests
def test_rectification_moving_along_line():
    """
    Test moving along line in epipolar geometry
    """
    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    current_left_coords = np.array([[5000.5, 5000.5, 0.0]], dtype=np.float64)
    mean_spacing = 1
    epi_step = 1
    alphas = 0
    default_elev = 0.0
    # ground truth next pixel
    # col pixel size of the image
    col_pixel_size = 1.0
    reference_next_cords = np.array([[5000.5, 5000.5 + col_pixel_size, 0.0]], dtype=np.float64)

    next_cords, _ = moving_along_lines(
        geom_model_left, geom_model_right, current_left_coords, mean_spacing, default_elev, epi_step, alphas
    )

    np.testing.assert_array_equal(reference_next_cords, next_cords)


@pytest.mark.unit_tests
def test_rectification_moving_to_next_line():
    """
    Test moving to next line in epipolar geometry
    """
    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )

    current_left_coords = np.array([5000.5, 5000.5, 0.0], dtype=np.float64)
    mean_spacing = 1
    epi_step = 1
    alphas = 0
    default_elev = 0.0
    # ground truth next pixel
    # row pixel size of the image
    row_pixel_size = 1.0
    reference_next_cords = np.array([5000.5 + row_pixel_size, 5000.5, 0.0], dtype=np.float64)

    next_cords, _ = moving_to_next_line(
        geom_model_left, geom_model_right, current_left_coords, mean_spacing, default_elev, epi_step, alphas
    )

    np.testing.assert_array_equal(reference_next_cords, next_cords)


@pytest.mark.unit_tests
def test_epipolar_angle():
    """
    test epipolar angle computation
    """
    # First case : same column, positive direction [row, col, alt]
    start_line_1 = np.array([1, 0, 0])
    end_line_1 = np.array([2, 0, 0])

    reference_alpha_1 = math.pi / 2.0
    alpha = compute_epipolar_angle(end_line_1, start_line_1)
    assert alpha == reference_alpha_1

    # Second case : same column, negative direction [row, col, alt]
    start_line_2 = np.array([2, 0, 0])
    end_line_2 = np.array([1, 0, 0])

    reference_alpha_2 = -(math.pi / 2.0)
    alpha = compute_epipolar_angle(end_line_2, start_line_2)
    assert alpha == reference_alpha_2

    # Third case : different column, positive direction [row, col, alt]
    start_line_3 = np.array([2, 0, 0])
    end_line_3 = np.array([1, 1, 0])

    slope = (1 - 2) / (1 - 0)

    reference_alpha_3 = np.arctan(slope)
    alpha = compute_epipolar_angle(end_line_3, start_line_3)
    assert alpha == reference_alpha_3

    # Fourth case : different column, negative direction [row, col, alt]
    start_line_4 = np.array([2, 1, 0])
    end_line_4 = np.array([1, 0, 0])

    slope = (1 - 2) / (0 - 1)
    reference_alpha_4 = math.pi + np.arctan(slope)
    alpha = compute_epipolar_angle(end_line_4, start_line_4)
    assert alpha == reference_alpha_4

    # With multiple point
    start_lines = np.stack((start_line_1, start_line_2, start_line_3, start_line_4))
    end_lines = np.stack((end_line_1, end_line_2, end_line_3, end_line_4))
    reference_alphas = np.stack((reference_alpha_1, reference_alpha_2, reference_alpha_3, reference_alpha_4))

    alphas = compute_epipolar_angle(end_lines, start_lines)
    np.testing.assert_array_equal(alphas, reference_alphas)


@pytest.mark.unit_tests
def test_rectification_grid_pos_inside_prepare_footprint_bounding_box():
    """
    Test that epipolar grid is inside the footprint returned by prepare_rectification
    """
    # Generate epipolar grid and parameters
    # Ground truth generated by GridBasedResampling function from OTB.
    epi_grid = rasterio.open(os.path.join(data_path(), "rectification", "gt_left_grid.tif"))
    width = epi_grid.width
    height = epi_grid.height
    transform = epi_grid.transform
    # Grid origin
    origin_row = transform[5]
    origin_col = transform[2]
    # Pixel size
    pixel_size_row = transform[4]
    pixel_size_col = transform[0]
    # Get array
    grid_col_dep = epi_grid.read(1)
    grid_row_dep = epi_grid.read(2)
    # Origin coordinates
    grid_origin_row = origin_row + pixel_size_row / 2.0
    grid_origin_col = origin_col + pixel_size_col / 2.0
    # Create grid with displacements
    grid_pos_col = np.arange(grid_origin_col, grid_origin_col + width * pixel_size_col, step=pixel_size_col)
    grid_pos_col = np.tile(grid_pos_col, (height, 1))
    grid_pos_row = np.arange(grid_origin_row, grid_origin_row + height * pixel_size_row, step=pixel_size_row)
    grid_pos_row = np.tile(grid_pos_row, (width, 1)).T
    pos_col = grid_col_dep + grid_pos_col
    pos_row = grid_row_dep + grid_pos_row

    # Get positions
    positions = np.stack((pos_col.flatten(), pos_row.flatten()))  # X, Y

    # Get grid footprint
    grid_footprint = np.array(
        [
            positions[:, int(grid_origin_row)],
            positions[:, int(grid_origin_row + grid_pos_row.shape[0] - 1)],
            positions[:, int(grid_origin_row + grid_pos_row.size - 1)],
            positions[:, int(grid_origin_row + grid_pos_row.size - 22)],
        ]
    )
    # OTB convention is [col, row, altitude], shareloc convention is [row, col, altitude]
    grid_footprint = grid_footprint[:, [1, 0]]

    # Compute shareloc epipolar footprint
    left_im = Image(os.path.join(data_path(), "rectification", "left_image.tif"))
    geom_model_left = RPC.from_any(
        os.path.join(data_path(), "rectification", "left_image.geom"), topleftconvention=True
    )
    geom_model_right = RPC.from_any(
        os.path.join(data_path(), "rectification", "right_image.geom"), topleftconvention=True
    )
    epi_step = 30
    elevation_offset = 50
    default_elev = 0.0
    _, _, _, footprint = prepare_rectification(
        left_im, geom_model_left, geom_model_right, default_elev, epi_step, elevation_offset
    )
    footprint = footprint[:, 0:2]

    # Bounding box
    min_row, max_row = min(footprint[:, 0]), max(footprint[:, 0])
    min_col, max_col = min(footprint[:, 1]), max(footprint[:, 1])

    # Test that grid_footprint is in epipolar footprint
    assert np.all(np.logical_and(min_row < grid_footprint[:, 0], grid_footprint[:, 0] < max_row))
    assert np.all(np.logical_and(min_col < grid_footprint[:, 1], grid_footprint[:, 1] < max_col))


















##############################################################################################################################




def write_diff_grid(path,grid):

    row, col = np.shape(grid)

    with rasterio.open(
        path, "w", driver="GTiff", dtype=np.float64, width=col, height=row, count=1
    ) as source_ds:
            source_ds.write(grid, 1)













@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_pyrugged():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Pyrugged_geom
    Earth elevation: default to 0.0
    """
    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,None,None,None)
    geom_right = Pyrugged_geom(file_dimap_right,None,None,None)

    epi_step = 60
    elevation_offset = 50
    default_elev = Location(
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
    elev_2 = Location(
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


    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2
    )


    #print("\n\n\n DEBUT RPC \n\n")

    geom_model_left = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    )
    geom_model_right = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    )
 
    left_grid_rpc, right_grid_rpc, img_size_row_rpc, img_size_col_rpc, mean_br_rpc = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, 0.0, epi_step, elevation_offset
    )

    right_row = abs(right_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])
    left_row = abs(left_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])
    right_col = abs(right_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])
    left_col = abs(left_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])


    # #write grid
    # write_diff_grid( "/new_cars/left_col_0.tif",left_col)
    # write_diff_grid( "/new_cars/left_row_0.tif",left_row)
    # write_diff_grid( "/new_cars/right_col_0.tif",right_col)
    # write_diff_grid( "/new_cars/right_row_0.tif",right_row)



    print("\n\n",np.amax(right_row))
    print(np.amax(right_col))

    assert left_grid.data[0, :, :] == pytest.approx(left_grid_rpc.data[0, :, :], abs=2e-2)
    assert left_grid.data[1, :, :] == pytest.approx(left_grid_rpc.data[1, :, :], abs=2e-2)
    assert right_grid.data[0, :, :] == pytest.approx(right_grid_rpc.data[0, :, :], abs=9e-2)
    assert right_grid.data[1, :, :] == pytest.approx(right_grid_rpc.data[1, :, :], abs=3e-2)
    assert img_size_row == pytest.approx(img_size_row_rpc, abs=1e-8)
    assert img_size_col == pytest.approx(img_size_col_rpc, abs=1e-8)
    assert mean_br == pytest.approx(mean_br_rpc, abs=2e-5)


   



@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_pyrugged_alti():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Pyrugged_geom
    Earth elevation: default to 100.0
    """
    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,None,None,None)
    geom_right = Pyrugged_geom(file_dimap_right,None,None,None)

    epi_step = 60
    elevation_offset = 50
    alti = 100.0
    default_elev = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=alti,
            light_time = geom_left.light_time,
            aberration_light = geom_left.aberration_light,
            atmospheric_refraction = geom_left.atmospheric_refraction,
            )
    elev_2 = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=None,
            geoid_path=None,
            alti_over_ellipsoid=alti,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )


    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2
    )



    geom_model_left = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    )
    geom_model_right = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    )
 
    left_grid_rpc, right_grid_rpc, img_size_row_rpc, img_size_col_rpc, mean_br_rpc = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, alti, epi_step, elevation_offset
    )


    right_row = abs(right_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])
    right_col = abs(right_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])

    print(np.amax(right_row))
    print(np.amax(right_col))

    assert left_grid.data[0, :, :] == pytest.approx(left_grid_rpc.data[0, :, :], abs=1e-2)
    assert left_grid.data[1, :, :] == pytest.approx(left_grid_rpc.data[1, :, :], abs=1e-2)
    assert right_grid.data[0, :, :] == pytest.approx(right_grid_rpc.data[0, :, :], abs=9e-2)
    assert right_grid.data[1, :, :] == pytest.approx(right_grid_rpc.data[1, :, :], abs=3e-2)
    assert img_size_row == pytest.approx(img_size_row_rpc, abs=1e-8)
    assert img_size_col == pytest.approx(img_size_col_rpc, abs=1e-8)
    assert mean_br == pytest.approx(mean_br_rpc, abs=2e-5)






@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_pyrugged_dtm_geoid():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Pyrugged_geom
    Earth elevation: DTM + GEOID
    """
    file_dimap_left="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    file_dimap_right="/new_cars/shareloc/tests/data/pyrugged/rectification/DIM_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"

    left_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001_R1C1.TIFF")
    right_im = Image("/new_cars/shareloc/tests/data/pyrugged/rectification/IMG_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001_R1C1.TIFF")

    geom_left = Pyrugged_geom(file_dimap_left,None,None,None)
    geom_right = Pyrugged_geom(file_dimap_right,None,None,None)

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90"#_only_s38"
    geoid_file = "/new_cars/shareloc/tests/data/pyrugged/rectification/egm96.grd"

    epi_step = 120
    elevation_offset = 50
    default_elev = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = None,
            aberration_light = None,
            atmospheric_refraction = None,
            )
    elev_2 = Location(
            file_dimap_right,
            sensor_name="sensor_b",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = None,
            aberration_light = None,
            atmospheric_refraction = None,
            )
    
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2
    )


    geom_model_left = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    )
    geom_model_right = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    )

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90_only_s38/e144/s38.dt1"
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)

 
    left_grid_rpc, right_grid_rpc, img_size_row_rpc, img_size_col_rpc, mean_br_rpc = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, dtm_ventoux, epi_step, elevation_offset
    )

    # right_row = abs(right_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])
    # right_col = abs(right_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])

    # # write grid
    
    # # write_diff_grid( "/new_cars/left_col_dtm.tif",left_col)
    # # write_diff_grid( "/new_cars/left_row_dtm.tif",left_row)
    # write_diff_grid( "/new_cars/right_diff_col_dtm.tif",right_col)
    # write_diff_grid( "/new_cars/right_diff_row_dtm.tif",right_row)



    assert left_grid.data[0, :, :] == pytest.approx(left_grid_rpc.data[0, :, :], abs=1e-2)
    assert left_grid.data[1, :, :] == pytest.approx(left_grid_rpc.data[1, :, :], abs=1e-2)
    assert right_grid.data[0, :, :] == pytest.approx(right_grid_rpc.data[0, :, :], abs=9e-2)
    assert right_grid.data[1, :, :] == pytest.approx(right_grid_rpc.data[1, :, :], abs=1e-1)
    assert img_size_row == pytest.approx(img_size_row_rpc, abs=1e-8)
    assert img_size_col == pytest.approx(img_size_col_rpc, abs=1e-8)
    assert mean_br == pytest.approx(mean_br_rpc, abs=1e-5)





@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_pyrugged_correction():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Pyrugged_geom
    Earth elevation: default to 100.0
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

    epi_step = 60
    elevation_offset = 50
    default_elev = Location(
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
    elev_2 = Location(
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


    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2
    )


    #print("\n\n\n DEBUT RPC \n\n")

    geom_model_left = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    )
    geom_model_right = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    )
 
    left_grid_rpc, right_grid_rpc, img_size_row_rpc, img_size_col_rpc, mean_br_rpc = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, 0.0, epi_step, elevation_offset
    )

    right_row = abs(right_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])
    left_row = abs(left_grid.data[0, :, :]-left_grid_rpc.data[0, :, :])
    right_col = abs(right_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])
    left_col = abs(left_grid.data[1, :, :]-left_grid_rpc.data[1, :, :])


    # #write grid
    # write_diff_grid( "/new_cars/left_col_0.tif",left_col)
    # write_diff_grid( "/new_cars/left_row_0.tif",left_row)
    # write_diff_grid( "/new_cars/right_col_0.tif",right_col)
    # write_diff_grid( "/new_cars/right_row_0.tif",right_row)


    print("Rectif alt 0.0 with correction")
    print(np.amax(right_row))
    print(np.amax(right_col))
    print(np.amax(left_row))
    print(np.amax(left_col),end="\n\n")
    print(abs(mean_br-mean_br_rpc))

    assert left_grid.data[0, :, :] == pytest.approx(left_grid_rpc.data[0, :, :], abs=2e-2)
    assert left_grid.data[1, :, :] == pytest.approx(left_grid_rpc.data[1, :, :], abs=2e-2)
    assert right_grid.data[0, :, :] == pytest.approx(right_grid_rpc.data[0, :, :], abs=10)
    assert right_grid.data[1, :, :] == pytest.approx(right_grid_rpc.data[1, :, :], abs=10)
    assert img_size_row == pytest.approx(img_size_row_rpc, abs=1e-8)
    assert img_size_col == pytest.approx(img_size_col_rpc, abs=1e-8)
    assert mean_br == pytest.approx(mean_br_rpc, abs=1e-3)








@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_pyrugged_dtm_geoid_correction():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Pyrugged_geom
    Earth elevation: DTM + GEOID
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

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90"
    geoid_file = "/new_cars/shareloc/tests/data/pyrugged/rectification/egm96.grd"

    epi_step = 60
    elevation_offset = 50
    default_elev = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )
    elev_2 = Location(
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
    
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2
    )


    geom_model_left = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250026276_SEN_PRG_FC_5109-001.XML"
    )
    geom_model_right = RPC.from_any(
        "/new_cars/shareloc/tests/data/pyrugged/rectification/RPC_PHR1A_P_201202250025329_SEN_PRG_FC_5110-001.XML"
    )

    dtm_file =  "/new_cars/shareloc/tests/data/pyrugged/rectification/PlanetDEM90_only_s38/e144/s38.dt1"
    geoid_file = os.path.join(data_path(), "dtm", "geoid", "egm96_15.gtx")
    dtm_ventoux = DTMIntersection(dtm_file, geoid_file)

 
    left_grid_rpc, right_grid_rpc, img_size_row_rpc, img_size_col_rpc, mean_br_rpc = compute_stereorectification_epipolar_grids(
        left_im, geom_model_left, right_im, geom_model_right, dtm_ventoux, epi_step, elevation_offset
    )

    right_row = abs(right_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])
    right_col = abs(right_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])
    left_row = abs(left_grid.data[0, :, :]-left_grid_rpc.data[0, :, :])
    left_col = abs(left_grid.data[1, :, :]-left_grid_rpc.data[1, :, :])

    # write grid
    write_diff_grid( "/new_cars/left_col_dtm.tif",left_col)
    write_diff_grid( "/new_cars/left_row_dtm.tif",left_row)
    write_diff_grid( "/new_cars/right_diff_col_dtm.tif",right_col)
    write_diff_grid( "/new_cars/right_diff_row_dtm.tif",right_row)

    print("Rectif DTM with correction")
    print(np.amax(abs(left_grid.data[0, :, :]-left_grid_rpc.data[0, :, :])))
    print(np.amax(abs(left_grid.data[1, :, :]-left_grid_rpc.data[1, :, :])))
    print(np.amax(abs(right_grid.data[0, :, :]-right_grid_rpc.data[0, :, :])))
    print(np.amax(abs(right_grid.data[1, :, :]-right_grid_rpc.data[1, :, :])),end="\n\n")

    # Rectif DTM with correction
    # 0.0029982045818996994
    # 0.008901000728201325
    # 5.80162125891502
    # 1.5637583463671945


    assert left_grid.data[0, :, :] == pytest.approx(left_grid_rpc.data[0, :, :], abs=1e-2)
    assert left_grid.data[1, :, :] == pytest.approx(left_grid_rpc.data[1, :, :], abs=1e-2)
    assert right_grid.data[0, :, :] == pytest.approx(right_grid_rpc.data[0, :, :], abs=10)
    assert right_grid.data[1, :, :] == pytest.approx(right_grid_rpc.data[1, :, :], abs=10)
    assert img_size_row == pytest.approx(img_size_row_rpc, abs=1e-8)
    assert img_size_col == pytest.approx(img_size_col_rpc, abs=1e-8)
    assert mean_br == pytest.approx(mean_br_rpc, abs=1e-3)



@pytest.mark.unit_tests
def test_compute_stereorectification_epipolar_grids_geomodel_pyrugged_dtm_geoid_correction_sinusoid():
    """
    Test epipolar grids generation : check epipolar grids, epipolar image size, mean_baseline_ratio

    Input Geomodels: Pyrugged_geom
    Earth elevation: DTM + GEOID
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
    geoid_file = "/new_cars/shareloc/tests/data/pyrugged/rectification/egm96.grd"

    epi_step = 300
    elevation_offset = 50
    default_elev = Location(
            file_dimap_left,
            sensor_name="sensor_a",
            physical_data_dir=None,
            dem_path=dtm_file,
            geoid_path=geoid_file,
            alti_over_ellipsoid=0.0,
            light_time = geom_right.light_time,
            aberration_light = geom_right.aberration_light,
            atmospheric_refraction = geom_right.atmospheric_refraction,
            )
    

    abs_date = PHRParser(file_dimap_right,"just_for_time_ref").start_time
    amp = 0.2e-6
    freq = 30
    phase = 0

    elev_2 = Location(
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
    
    left_grid, right_grid, img_size_row, img_size_col, mean_br = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2
    )


    elev_2_ref = Location(
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

    left_grid_ref, right_grid_ref, img_size_row_ref, img_size_col_ref, mean_br_ref = compute_stereorectification_epipolar_grids(
        left_im, geom_left, right_im, geom_right, default_elev, epi_step, elevation_offset,elev_2_ref
    )



    # right_row = abs(right_grid.data[0, :, :]-right_grid_ref.data[0, :, :])
    # right_col = abs(right_grid.data[1, :, :]-right_grid_ref.data[1, :, :])
    # left_row = abs(left_grid.data[0, :, :]-left_grid_ref.data[0, :, :])
    # left_col = abs(left_grid.data[1, :, :]-left_grid_ref.data[1, :, :])

    # # write grid
    # write_diff_grid( "/new_cars/left_col_dtm.tif",left_col)
    # write_diff_grid( "/new_cars/left_row_dtm.tif",left_row)
    # write_diff_grid( "/new_cars/right_diff_col_dtm.tif",right_col)
    # write_diff_grid( "/new_cars/right_diff_row_dtm.tif",right_row)

    print("Rectif DTM with correction")
    print(np.amax(abs(left_grid.data[0, :, :]-left_grid_ref.data[0, :, :])))
    print(np.amax(abs(left_grid.data[1, :, :]-left_grid_ref.data[1, :, :])))
    print(np.amax(abs(right_grid.data[0, :, :]-right_grid_ref.data[0, :, :])))
    print(np.amax(abs(right_grid.data[1, :, :]-right_grid_ref.data[1, :, :])),end="\n\n")

    # Rectif DTM with correction
    # 0.0029982045818996994
    # 0.008901000728201325
    # 5.80162125891502
    # 1.5637583463671945


    assert left_grid.data[0, :, :] == pytest.approx(left_grid_ref.data[0, :, :], abs=1e-2)
    assert left_grid.data[1, :, :] == pytest.approx(left_grid_ref.data[1, :, :], abs=1e-2)
    assert right_grid.data[0, :, :] == pytest.approx(right_grid_ref.data[0, :, :], abs=0.9)
    assert right_grid.data[1, :, :] == pytest.approx(right_grid_ref.data[1, :, :], abs=1e-2)
    assert img_size_row == pytest.approx(img_size_row_ref, abs=1e-8)
    assert img_size_col == pytest.approx(img_size_col_ref, abs=1e-8)
    assert mean_br == pytest.approx(mean_br_ref, abs=1e-3)