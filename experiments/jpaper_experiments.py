#from numpy.lib.arraysetops import isin
#from common.fields import FunctionVectorField
from collections import OrderedDict
from common.runner import Runner
import time
import numpy as np
from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
try:
    import cvxopt as co
except ModuleNotFoundError:
    print('CVXOPT not installed. ')
import scipy

from common.solvers import group_sparse_cvx, weighted_group_sparse_scipy
from common.solvers import group_sparsify as solvers_group_sparsify

import gsim
from gsim.gfigure import GFigure
from common.utilities import dB_to_natural, dbm_to_watt, empty_array, natural_to_dB, watt_to_dbW, watt_to_dbm
from common.grid import RectangularGrid3D
from channels.channel import Channel, FreeSpaceChannel
from channels.tomographic_channel import TomographicChannel
from channels.ray_tracing_channel import RayTracingChannel
from common.environment import BlockUrbanEnvironment1, BlockUrbanEnvironment2, GridBasedBlockUrbanEnvironment, UrbanEnvironment, Building, RayTracingEnvironment


from placement.placers import FlyGrid, SingleUAVPlacer, \
    GroupSparseUAVPlacer, SparseUAVPlacer, KMeansPlacer, \
        SpaceRateKMeans, GridRatePlacer, SpiralPlacer,\
            SparseRecoveryPlacer, GeneticPlacer
from simulators.PlacementSimulator import metrics_vs_min_user_rate, \
    metrics_vs_num_users,     place_and_plot, mean_num_uavs, user_loc_mc,\
        metrics_vs_environments_and_channels, metrics_vs_placers

import grid_utilities
import gc
import sys

import pickle
import os

# from cpython cimport PyObject
# from cpython.ref cimport Py_INCREF, Py_XDECREF

# import tensorflow as tf
# from tensorflow import keras

# tf.keras.backend.set_floatx('float64')
# print("TensorFlow version: ", tf.__version__)
# print("Num GPUs Available: ",
#       len(tf.config.experimental.list_physical_devices('GPU')))

# tf.config.experimental_run_functions_eagerly(False)
# #tf.config.experimental_run_functions_eagerly(True)


class ExperimentSet(gsim.AbstractExperimentSet):

    # num_mc_iter is currently set as 1 in all experiments. Higher num_mc_iter, e.g., hundreds, should be set to obtain the same figures as in the paper.
    
    # Mean number of ABSs vs num_users
    def experiment_1003(l_args):

        # run monte carlo and save results
        exp_num = "1003"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists("output/tempt/"):
            os.makedirs("output/tempt/")
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        l_min_user_rates = np.array([20.]) * 1e6
        l_max_uav_total_rates = np.array([99.]) * 1e6

        l_num_users = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

        num_mc_iter = 1
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    env = GridBasedBlockUrbanEnvironment(
                        area_len=[500, 400, 150],
                        num_pts_slf_grid=[50, 40, 15],
                        num_pts_fly_grid=[9, 9, 5],
                        min_fly_height=50,
                        building_absorption=1)

                    channel = TomographicChannel(slf=env.slf,
                                                 freq_carrier=2.4e9,
                                                 bandwidth=20e6,
                                                 tx_dbpower=watt_to_dbW(.1),
                                                 noise_dbpower=-96)

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=200,
                        num_sols_in_pop=50,
                        mut_rate=0.02,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=90)

                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=90)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=90)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        channel,
                        min_user_rate,
                        l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=run_new,
                        b_load_env=not run_new,
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # plot the results
        change_para = "num_users"

        label_x = 'Number of GTs (M)'
        l_ticks_y = np.arange(5, 50, 5)
        legend_loc = 2
        exp_num = 1003

        f_plot_with_rate_users(change_para, exp_num, l_num_users,
                               l_min_user_rates / 1e6,
                               l_max_uav_total_rates / 1e6, start_mc_iter,
                               num_mc_iter, label_x, l_ticks_y, legend_loc)

    # Mean number of ABSs vs min_user_rate
    def experiment_1004(l_args):

        # Run monte carlo and save results
        exp_num = "1004"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists("output/tempt/"):
            os.makedirs("output/tempt/")
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        l_min_user_rates = np.array([10., 15, 20, 25, 30, 35, 40, 45]) * 1e6
        l_max_uav_total_rates = np.array([100.]) * 1e6

        l_num_users = np.array([70])

        num_mc_iter = 1
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    env = GridBasedBlockUrbanEnvironment(
                        area_len=[500, 400, 150],
                        num_pts_slf_grid=[50, 40, 15],
                        num_pts_fly_grid=[9, 9, 5],
                        min_fly_height=50,
                        building_absorption=1)

                    channel = TomographicChannel(slf=env.slf,
                                                 freq_carrier=2.4e9,
                                                 bandwidth=20e6,
                                                 tx_dbpower=watt_to_dbW(.1),
                                                 noise_dbpower=-96)

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=200,
                        num_sols_in_pop=50,
                        mut_rate=0.02,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=90)

                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=90)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=90)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        channel,
                        min_user_rate,
                        l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=run_new,
                        b_load_env=not run_new,
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # plot the results
        change_para = "min_user_rate"

        label_x = 'Minimum GT rate [Mbps]'
        l_ticks_y = np.arange(5, 75, 5)
        legend_loc = 2
        exp_num = 1004

        f_plot_with_rate_users(change_para, exp_num, l_num_users,
                               l_min_user_rates / 1e6,
                               l_max_uav_total_rates / 1e6, start_mc_iter,
                               num_mc_iter, label_x, l_ticks_y, legend_loc)

    # Mean number of ABSs vs total rate of each ABS
    def experiment_1005(l_args):

        # Run monte carlo and save results
        exp_num = "1005"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists("output/tempt/"):
            os.makedirs("output/tempt/")
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        l_min_user_rates = np.array([20.]) * 1e6
        l_max_uav_total_rates = np.array([39., 49, 59, 69, 79, 89, 99]) * 1e6

        l_num_users = np.array([70])

        num_mc_iter = 1
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    env = GridBasedBlockUrbanEnvironment(
                        area_len=[500, 400, 150],
                        num_pts_slf_grid=[50, 40, 15],
                        num_pts_fly_grid=[9, 9, 5],
                        min_fly_height=50,
                        building_absorption=1)

                    channel = TomographicChannel(slf=env.slf,
                                                 freq_carrier=2.4e9,
                                                 bandwidth=20e6,
                                                 tx_dbpower=watt_to_dbW(.1),
                                                 noise_dbpower=-96)

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=200,
                        num_sols_in_pop=50,
                        mut_rate=0.02,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=90)

                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=90)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=90)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        channel,
                        min_user_rate,
                        l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=run_new,
                        b_load_env=not run_new,
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # plot the results
        change_para = "max_uav_total_rate"

        label_x = 'Total rate of each ABS [Mbps]'
        l_ticks_y = np.arange(5, 75, 5)
        legend_loc = 1
        exp_num = 1005

        f_plot_with_rate_users(change_para, exp_num, l_num_users,
                               l_min_user_rates / 1e6,
                               l_max_uav_total_rates / 1e6, start_mc_iter,
                               num_mc_iter, label_x, l_ticks_y, legend_loc)

    # Mean number of ABSs vs building_absorption
    def experiment_1006(l_args):

        # experiment parameters
        exp_num = "1006"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_bd_absorption = [
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25
        ]
        num_mc_iter = 1
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for bd_absorption in l_bd_absorption:

            exp_type = str(bd_absorption) + "_buildAbsorp"

            env = GridBasedBlockUrbanEnvironment(
                area_len=[500, 400, 150],
                num_pts_slf_grid=[50, 40, 15],
                num_pts_fly_grid=[9, 9, 5],
                min_fly_height=50,
                building_absorption=bd_absorption)

            channel = TomographicChannel(slf=env.slf,
                                         freq_carrier=2.4e9,
                                         bandwidth=20e6,
                                         tx_dbpower=watt_to_dbW(.1),
                                         noise_dbpower=-96)

            # Set to None one of the following
            min_user_rate = 17e6
            max_uav_total_rate = 84e6
            num_users = 70

            b_admm_decrease_err_tol = True

            if not os.path.exists("output/tempt/"):
                os.makedirs("output/tempt/")
            if not os.path.exists(save_exp_at):
                os.makedirs(save_exp_at)

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=100,
                admm_stepsize=1e-7,
                admm_max_num_iter=70,
                admm_initial_error_tol=5,
                eps_abs=1e-4,
                eps_rel=1e-4,
                b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
                                  mut_rate=0.02,
                                  num_sols_in_pop=50,
                                  max_uav_total_rate=max_uav_total_rate,
                                  max_num_uavs=90)

            pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                 max_uav_total_rate=max_uav_total_rate,
                                 max_num_uavs=90)

            pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate,
                                    max_uav_total_rate=max_uav_total_rate,
                                    max_num_uavs=90)

            # Choose:
            results = mean_num_uavs(env,
                                    channel,
                                    min_user_rate,
                                    l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                                    num_users=num_users,
                                    num_mc_iter=num_mc_iter,
                                    max_uav_total_rate=max_uav_total_rate,
                                    start_mc_iter=start_mc_iter,
                                    b_save_env=run_new,
                                    b_load_env=not run_new,
                                    b_save_results=b_save_results,
                                    save_exp_at=save_exp_at,
                                    exp_type=exp_type)

            print(results)

        # Plot the results
        exp_num = "1006"
        plt_label_x = 'Building absorption [dB/m]'
        post_fix = "_buildAbsorp"
        l_varied_parameters = l_bd_absorption
        plt_ticks_y = np.arange(10, 60, 5)

        start_mc = start_mc_iter
        num_users = 70  # Mpbs
        min_user_rate = 17  # Mpbs
        max_uav_total_rate = 84  # Mpbs
        f_plot_with_changed_env(
            exp_num,
            post_fix,
            plt_label_x,
            l_varied_parameters,
            plt_ticks_y,
            num_mc_iter,
            start_mc,
            num_users,
            min_user_rate,
            max_uav_total_rate,
        )

    def experiment_1007(l_args):
        """
        Mean number of ABSs vs num_user for the
        given min_user_rate and max_uav_rate
        """
        # experiment parameters
        exp_num = "1007"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        l_min_user_rates = np.array([20]) * 1e6
        l_max_uav_total_rates = np.array([74]) * 1e6
        l_num_users = np.array([30, 40, 50, 60, 70, 80, 90, 100])

        num_mc_iter = 1
        start_mc_iter = 0

        env = RayTracingEnvironment(dataset='ottawa_06')

        ch_ray_tracing = RayTracingChannel(
            env=env,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            disable_gridpts_by_dominated_verticals=False)

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:
                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True
                    run_new = True
                    b_save_results = True

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=20,
                        mut_rate=0.05,
                        num_sols_in_pop=20,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105)

                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=105)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        ch_ray_tracing,
                        min_user_rate,
                        l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=False,
                        b_load_env=not run_new,
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # plot the results
        change_para = "num_users"
        label_x = 'Number of GTs (M)'
        l_ticks_y = l_num_users
        legend_loc = 2

        f_plot_with_rate_users(change_para, exp_num, l_num_users,
                               l_min_user_rates / 1e6,
                               l_max_uav_total_rates / 1e6, start_mc_iter,
                               num_mc_iter, label_x, l_ticks_y, legend_loc)

    """ This experiment plots the mean number of UAVs vs the minimum GT rate
        for the given max_uav_total_rate.
    """

    def experiment_1008(l_args):
        """ This experiment plots the mean number of UAVs vs the minimum GT rate
            for the given max_uav_total_rate.
        """

        exp_num = "1008"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_min_user_rates = np.array([11, 19, 27, 35, 45]) * 1e6
        l_max_uav_total_rates = np.array([100]) * 1e6

        l_num_users = np.array([50])

        num_mc_iter = 1
        start_mc_iter = 0

        env = RayTracingEnvironment(dataset='ottawa_06')
        ch_ray_tracing = RayTracingChannel(
            env=env,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            disable_gridpts_by_dominated_verticals=False)

        for num_users in l_num_users:
            for min_user_rate in l_min_user_rates:
                for max_uav_total_rate in l_max_uav_total_rates:

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True
                    run_new = True
                    b_save_results = True

                    # simulation code
                    if not os.path.exists(save_exp_at):
                        os.makedirs(save_exp_at)

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=200,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105,
                        mut_rate=0.05,
                        num_sols_in_pop=20,
                    )
                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=105)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        ch_ray_tracing,
                        min_user_rate,
                        l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=False,
                        b_load_env=False,  # not run_new
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # plot the results
        change_para = "min_user_rate"
        label_x = 'Minimum GT rate [Mbps]'
        l_ticks_y = l_min_user_rates / 1e6
        legend_loc = 2

        f_plot_with_rate_users(change_para, exp_num, l_num_users,
                               l_min_user_rates / 1e6,
                               l_max_uav_total_rates / 1e6, start_mc_iter,
                               num_mc_iter, label_x, l_ticks_y, legend_loc)

    """ This experiment plots the mean number of UAVs vs
            the minimum GT rate for the proposed GSPA with different 
            backhaul link capacity .
    """

    def experiment_1009(l_args):
        """ This experiment plots the mean number of UAVs vs
        the minimum GT rate for the proposed GSPA with different
        backhaul link capacity .
        """
        # experiment parameters
        exp_num = "1009"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_min_user_rates = np.array([3, 5, 7, 9, 11]) * 1e6
        l_max_uav_total_rates = np.array([34, 44, 54, 64, 74]) * 1e6
        l_num_users = np.array([50])

        num_mc_iter = 1
        start_mc_iter = 0

        env = RayTracingEnvironment(dataset='ottawa_06')
        ch_ray_tracing = RayTracingChannel(
            env=env,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            disable_gridpts_by_dominated_verticals=False)

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True
                    run_new = True
                    b_save_results = True

                    # simulation code
                    if not os.path.exists(save_exp_at):
                        os.makedirs(save_exp_at)

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=200,
                        mut_rate=0.05,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105,
                        num_sols_in_pop=15,
                    )

                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=105)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        ch_ray_tracing,
                        min_user_rate,
                        l_placers=[pl_gs],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=False,
                        b_load_env=False,  # not run_new
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # Plot the figure
        num_users = l_num_users[0]
        l_min_user_rates = l_min_user_rates / 1e6
        l_max_uav_total_rates = l_max_uav_total_rates / 1e6
        x_ticks_value = l_min_user_rates

        l_lower_bounds = [
            np.ceil(num_users * l_min_user_rates / max_uav_total_rate)
            for max_uav_total_rate in l_max_uav_total_rates
        ]

        l_placer_names = ["GroupSparseUAVPlacer"]

        l_markers = ["o", "^", "*", "v", "d"]
        m_num_uavs = np.zeros(
            (len(l_max_uav_total_rates), len(l_min_user_rates)))

        placer_name = l_placer_names[0]

        for ind_minRate in range(len(l_min_user_rates)):

            for ind_maxRate in range(len(l_max_uav_total_rates)):

                # exp_type = str(min_user_rate) + "minRate" + str(
                #     l_max_uav_total_rate[ind_maxRate]) + "_maxRate"
                exp_type = str(num_users) + "_users_" + str(
                    int(l_min_user_rates[ind_minRate])) + "_minRate_" + str(
                        int(l_max_uav_total_rates[ind_maxRate])) + "_maxRate"

                for ind_env in range(start_mc_iter,
                                     num_mc_iter + start_mc_iter):
                    load_at = f"output/exp_{exp_num}/results/" + placer_name + "/" + exp_type + "_env_" + str(
                        ind_env) + ".pck"

                    with open(load_at, "rb") as f:
                        l_uavs = pickle.load(f)

                    m_num_uavs[ind_maxRate, ind_minRate] += len(l_uavs)

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            f"max_uav_total_rate={max_uav_total_rate} Mbps"
            for max_uav_total_rate in l_max_uav_total_rates
        ]

        for ind_maxRate in range(len(l_max_uav_total_rates)):
            plt.plot(l_min_user_rates,
                     m_num_uavs[ind_maxRate],
                     marker=l_markers[ind_maxRate],
                     label=l_labels[ind_maxRate])

            plt.plot(
                l_min_user_rates,
                l_lower_bounds[ind_maxRate],
                linestyle='dashed',
                marker=l_markers[ind_maxRate],
                label=f"L. bound for {l_max_uav_total_rates[ind_maxRate]} Mbps",
                # color="black"
            )

        plt.xlabel('Minimum GT rate [Mbps]')
        plt.xticks(x_ticks_value)

        plt.ylabel('Mean number of ABSs')
        plt.legend(loc=2, ncol=2)
        plt.grid()
        plt.show()

        print('end of experiment')

    """ This experiment gets the mean number of 
    ABSs vs. backhaul link capacity for the given 
    minimum GT rate.
    """

    def experiment_1010(l_args):
        """ This experiment gets the mean number of
        ABSs vs. backhaul link capacity for the given
        minimum GT rate.
        """
        # experiment parameters
        exp_num = "1010"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_min_user_rates = np.array([7]) * 1e6
        l_max_uav_total_rates = np.array([34, 44, 54, 64, 74, 80, 100]) * 1e6

        l_num_users = np.array([50])

        num_mc_iter = 1
        start_mc_iter = 0

        env = RayTracingEnvironment(dataset='ottawa_06')
        ch_ray_tracing = RayTracingChannel(
            env=env,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            disable_gridpts_by_dominated_verticals=False)

        for num_users in l_num_users:
            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True
                    run_new = True
                    b_save_results = True

                    # simulation code
                    if not os.path.exists(save_exp_at):
                        os.makedirs(save_exp_at)

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    pl_ge = GeneticPlacer(
                        min_user_rate=min_user_rate,
                        max_num_gens=100,
                        mut_rate=0.05,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105,
                        num_sols_in_pop=20,
                    )

                    pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                         max_uav_total_rate=max_uav_total_rate,
                                         max_num_uavs=105)

                    pl_sr = SpaceRateKMeans(
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        max_num_uavs=105)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        ch_ray_tracing,
                        min_user_rate,
                        l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=False,
                        b_load_env=False,  # not run_new
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # plot the results
        change_para = "max_uav_rate"
        label_x = 'Total rate of each ABS [Mbps]'
        l_ticks_y = l_max_uav_total_rates / 1e6
        legend_loc = 2

        f_plot_with_rate_users(change_para, exp_num, l_num_users,
                               l_min_user_rates / 1e6,
                               l_max_uav_total_rates / 1e6, start_mc_iter,
                               num_mc_iter, label_x, l_ticks_y, legend_loc)

    # Mean number of ABSs vs min_user_rate for the proposed placer only
    def experiment_1011(l_args):

        # experiment parameters
        exp_num = "1011"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists("output/tempt/"):
            os.makedirs("output/tempt/")
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        l_min_user_rates = np.array([5, 7, 9, 11, 13, 15, 17]) * 1e6
        l_max_uav_total_rates = np.array([40, 60, 80, 100]) * 1e6

        l_num_users = np.array([70])

        num_mc_iter = 1
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    env = GridBasedBlockUrbanEnvironment(
                        area_len=[500, 400, 150],
                        num_pts_slf_grid=[50, 40, 15],
                        num_pts_fly_grid=[9, 9, 5],
                        min_fly_height=50,
                        building_absorption=1)

                    channel = TomographicChannel(slf=env.slf,
                                                 freq_carrier=2.4e9,
                                                 bandwidth=20e6,
                                                 tx_dbpower=watt_to_dbW(.1),
                                                 noise_dbpower=-96)

                    # Set to None one of the following
                    exp_type = str(num_users) + "_users_" + str(
                        int(min_user_rate / 1e6)) + "_minRate_" + str(
                            int(max_uav_total_rate / 1e6)) + "_maxRate"

                    b_admm_decrease_err_tol = True

                    pl_gs = GroupSparseUAVPlacer(
                        sparsity_tol=1e-2,
                        criterion="min_uav_num",
                        min_user_rate=min_user_rate,
                        max_uav_total_rate=max_uav_total_rate,
                        backend="admm",
                        reweighting_num_iter=100,
                        admm_stepsize=1e-7,
                        admm_max_num_iter=70,
                        admm_initial_error_tol=5,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                        b_save_group_weights=run_new,
                        b_load_group_weights=not run_new)

                    # Choose:
                    results = mean_num_uavs(
                        env,
                        channel,
                        min_user_rate,
                        l_placers=[pl_gs],
                        num_users=num_users,
                        num_mc_iter=num_mc_iter,
                        max_uav_total_rate=max_uav_total_rate,
                        start_mc_iter=start_mc_iter,
                        b_save_env=run_new,
                        b_load_env=not run_new,
                        b_save_results=b_save_results,
                        save_exp_at=save_exp_at,
                        exp_type=exp_type)

                    print(results)

        # Plot the results
        num_users = l_num_users[0]
        l_min_user_rate = l_min_user_rates / 1e6
        l_max_uav_total_rate = l_max_uav_total_rates / 1e6

        l_lower_bounds = [
            np.ceil(num_users * l_min_user_rate / max_uav_total_rate)
            for max_uav_total_rate in l_max_uav_total_rate
        ]

        placer_name = "GroupSparseUAVPlacer"

        l_markers = ["o", "^", "*", "v", "d"]
        m_num_uavs = np.zeros(
            (len(l_max_uav_total_rate), len(l_min_user_rate)))

        for ind_maxRate in range(len(l_max_uav_total_rate)):

            for ind_minRate in range(len(l_min_user_rate)):

                exp_type = str(num_users) + "_users_" + str(
                    int(l_min_user_rate[ind_minRate])) + "_minRate_" + str(
                        int(l_max_uav_total_rate[ind_maxRate])) + "_maxRate"

                for ind_env in range(0, num_mc_iter):

                    load_at = "output/exp_" + exp_num + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                        ind_env) + ".pck"

                    with open(load_at, "rb") as f:
                        l_uavs = pickle.load(f)

                    m_num_uavs[ind_maxRate, ind_minRate] += len(l_uavs)

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            r'$c^{\rm{BH}}$' + f" = {int(max_uav_total_rate)} Mbps"
            for max_uav_total_rate in l_max_uav_total_rate
        ]

        l_labels_lower_bounds = [
            'Lower bound for ' + r'$c^{\rm{BH}}$' +
            f" = {int(max_uav_total_rate)} Mbps"
            for max_uav_total_rate in l_max_uav_total_rate
        ]

        l_color_lower_bounds = [
            'tab:blue', 'tab:green', 'tab:purple', 'tab:red'
        ]

        for ind_maxRate in range(len(l_max_uav_total_rate)):
            plt.plot(l_min_user_rate,
                     m_num_uavs[ind_maxRate],
                     marker=l_markers[ind_maxRate],
                     label=l_labels[ind_maxRate],
                     color=l_color_lower_bounds[ind_maxRate])

            plt.plot(l_min_user_rate,
                     l_lower_bounds[ind_maxRate],
                     label=l_labels_lower_bounds[ind_maxRate],
                     marker=l_markers[ind_maxRate],
                     linestyle='dashed',
                     color=l_color_lower_bounds[ind_maxRate])

        plt.xlabel('Minimum GT rate [Mbps]')
        plt.xticks(l_min_user_rate)
        plt.yticks(np.arange(0, 50, 5))
        plt.ylabel('Mean number of ABSs')
        plt.legend(loc=2, ncol=2)
        plt.grid()
        plt.show()

    # Mean number of ABSs vs minimum flight height
    def experiment_1012(l_args):

        # experiment parameters
        exp_num = "1012"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_min_fly_height = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        num_mc_iter = 1
        start_mc_iter = 0

        min_user_rate = 17e6
        max_uav_total_rate = 84e6
        num_users = 70

        b_admm_decrease_err_tol = True
        run_new = True
        b_save_results = True

        if not os.path.exists("output/tempt/"):
            os.makedirs("output/tempt/")
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        for min_fly_height in l_min_fly_height:
            exp_type = str(int(min_fly_height)) + "_minFlyHeight"

            env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                                 num_pts_slf_grid=[50, 40, 15],
                                                 num_pts_fly_grid=[9, 9, 5],
                                                 min_fly_height=min_fly_height,
                                                 building_absorption=1)

            channel = TomographicChannel(slf=env.slf,
                                         freq_carrier=2.4e9,
                                         bandwidth=20e6,
                                         tx_dbpower=watt_to_dbW(.1),
                                         noise_dbpower=-96)

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=100,
                admm_stepsize=5e-8,
                admm_max_num_iter=100,
                admm_initial_error_tol=5,
                eps_abs=1e-5,
                eps_rel=1e-5,
                b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
                                  num_sols_in_pop=50,
                                  max_uav_total_rate=max_uav_total_rate,
                                  max_num_uavs=90)

            pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                 max_uav_total_rate=max_uav_total_rate,
                                 max_num_uavs=90)

            pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate,
                                    max_uav_total_rate=max_uav_total_rate,
                                    max_num_uavs=90)

            # Choose:
            results = mean_num_uavs(env,
                                    channel,
                                    min_user_rate,
                                    l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                                    num_users=num_users,
                                    num_mc_iter=num_mc_iter,
                                    max_uav_total_rate=max_uav_total_rate,
                                    start_mc_iter=start_mc_iter,
                                    b_save_env=run_new,
                                    b_load_env=not run_new,
                                    b_save_results=b_save_results,
                                    save_exp_at=save_exp_at,
                                    exp_type=exp_type)

            print(results)

        # Plot the results
        exp_num = "1012"
        plt_label_x = 'Minimum flight height [m]'
        post_fix = "_minFlyHeight"
        l_varied_parameters = l_min_fly_height
        plt_ticks_y = np.arange(10, 60, 5)
        start_mc = 0
        f_plot_with_changed_env(
            exp_num,
            post_fix,
            plt_label_x,
            l_varied_parameters,
            plt_ticks_y,
            num_mc_iter,
            start_mc,
            num_users,
            min_user_rate / 1e6,
            max_uav_total_rate / 1e6,
        )

    # Mean number of ABSs vs building_height
    def experiment_1013(l_args):

        # experiment parameters
        exp_num = "1013"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_building_height = [0, 10, 20, 30, 40, 50]

        min_user_rate = 17e6
        max_uav_total_rate = 84e6
        num_users = 70

        num_mc_iter = 1
        start_mc_iter = 0
        run_new = True
        b_save_results = True
        b_admm_decrease_err_tol = True

        if not os.path.exists("output/tempt/"):
            os.makedirs("output/tempt/")
        if not os.path.exists(save_exp_at):
            os.makedirs(save_exp_at)

        for building_height in l_building_height:

            exp_type = str(int(building_height)) + "_buildingHeight"

            env = GridBasedBlockUrbanEnvironment(
                area_len=[500, 400, 150],
                num_pts_slf_grid=[50, 40, 75],
                num_pts_fly_grid=[7, 7, 5],
                min_fly_height=50,
                building_absorption=1,
                building_height=building_height)

            channel = TomographicChannel(slf=env.slf,
                                         freq_carrier=2.4e9,
                                         bandwidth=20e6,
                                         tx_dbpower=watt_to_dbW(.1),
                                         noise_dbpower=-96)

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=100,
                admm_stepsize=5e-7,
                admm_max_num_iter=70,
                admm_initial_error_tol=5,
                eps_abs=1e-4,
                eps_rel=1e-4,
                b_admm_decrease_err_tol=b_admm_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
                                  num_sols_in_pop=50,
                                  mut_rate=0.02,
                                  max_uav_total_rate=max_uav_total_rate,
                                  max_num_uavs=90)

            pl_km = KMeansPlacer(min_user_rate=min_user_rate,
                                 max_uav_total_rate=max_uav_total_rate,
                                 max_num_uavs=90)

            pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate,
                                    max_uav_total_rate=max_uav_total_rate,
                                    max_num_uavs=90)

            # Choose:
            results = mean_num_uavs(env,
                                    channel,
                                    min_user_rate,
                                    l_placers=[pl_gs, pl_ge, pl_km, pl_sr],
                                    num_users=num_users,
                                    num_mc_iter=num_mc_iter,
                                    max_uav_total_rate=max_uav_total_rate,
                                    start_mc_iter=start_mc_iter,
                                    b_save_env=run_new,
                                    b_load_env=not run_new,
                                    b_save_results=b_save_results,
                                    save_exp_at=save_exp_at,
                                    exp_type=exp_type)

            print(results)

        # Plot the results
        post_fix = "_buildingHeight"
        plt_label_x = 'Height of the buildings [m]'
        l_varied_parameters = l_building_height
        plt_ticks_y = np.arange(12, 34, 2)
        start_mc = 0
        f_plot_with_changed_env(
            exp_num,
            post_fix,
            plt_label_x,
            l_varied_parameters,
            plt_ticks_y,
            num_mc_iter,
            start_mc,
            num_users,
            min_user_rate / 1e6,
            max_uav_total_rate / 1e6,
        )


def m(A):
    if isinstance(A, list):
        return [m(Am) for Am in A]
    return co.matrix(A)


def um(M):  #"unmatrix"
    if isinstance(M, list):
        return [um(Mm) for Mm in M]
    return np.array(M)


def sparsify(M, tol=0.01):
    n = np.linalg.norm(np.ravel(M), ord=1)
    M[M < tol * n] = 0
    return M


def group_sparsify(M, tol=0.01):
    n = np.linalg.norm(np.ravel(M), ord=1)
    for ind_col in range(M.shape[1]):
        if np.linalg.norm(M[:, ind_col]) < tol * n:
            M[:, ind_col] = 0
    return M


def f_plot_with_rate_users(change_para, exp_num, l_num_users, l_min_user_rate,
                           l_max_uav_total_rate, start_mc_iter, num_mc_iter,
                           label_x, l_ticks_y, legend_loc):

    l_placer_names = [
        "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
        "GroupSparseUAVPlacer"
    ]

    if change_para == "num_users":
        l_change_paras = l_num_users
    elif change_para == "min_user_rate":
        l_change_paras = l_min_user_rate
    else:
        l_change_paras = l_max_uav_total_rate

    lower_bounds = np.ceil(l_num_users * l_min_user_rate /
                           l_max_uav_total_rate)

    l_markers = ["o", "^", "*", "v"]
    m_num_uavs = np.zeros((len(l_placer_names), len(l_change_paras)))

    for ind_users in range(len(l_num_users)):

        num_users = l_num_users[ind_users]

        for ind_min in range(len(l_min_user_rate)):

            minRate = l_min_user_rate[ind_min]

            for ind_max in range(len(l_max_uav_total_rate)):

                if change_para == "num_users":
                    para = ind_users
                elif change_para == "min_user_rate":
                    para = ind_min
                else:
                    para = ind_max

                maxRate = l_max_uav_total_rate[ind_max]

                exp_type = str(num_users) + "_users_" + str(
                    int(minRate)) + "_minRate_" + str(
                        int(maxRate)) + "_maxRate"

                for ind_env in range(start_mc_iter,
                                     start_mc_iter + num_mc_iter):

                    for ind_name in range(len(l_placer_names)):

                        placer_name = l_placer_names[ind_name]

                        load_at = "output/exp_" + str(
                            exp_num
                        ) + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                            ind_env) + ".pck"

                        with open(load_at, "rb") as f:
                            l_uavs = pickle.load(f)

                        m_num_uavs[ind_name, para] += len(l_uavs)

    m_num_uavs = m_num_uavs / num_mc_iter

    l_labels = [
        "K-means Alg. (Galkin et al.)", "Space rate Alg. (Hammouti et al.)",
        "Genetic Alg. (Shehzad et al.)", "GSPA (proposed)"
    ]

    plt.plot(l_change_paras,
             lower_bounds,
             label="Lower bound",
             linestyle='dashed',
             color="black",
             marker='d')

    for ind_name in range(len(l_placer_names)):
        plt.plot(l_change_paras,
                 m_num_uavs[ind_name],
                 marker=l_markers[ind_name],
                 label=l_labels[ind_name])

    plt.xlabel(label_x)
    plt.xticks(l_change_paras)
    plt.yticks(l_ticks_y)
    plt.ylabel('Mean number of ABSs')
    plt.legend(loc=legend_loc)
    plt.grid()
    plt.show()


def f_plot_with_changed_env(
    exp_num,
    post_fix,
    plt_label_x,
    l_varied_parameters,
    plt_ticks_y,
    num_mc_iter,
    start_mc,
    num_users,
    min_user_rate,
    max_uav_total_rate,
):
    l_placer_names = [
        "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
        "GroupSparseUAVPlacer"
    ]
    l_markers = ["o", "^", "*", "v"]
    l_labels = [
        "K-means Alg. (Galkin et al.)", "Space rate Alg. (Hammouti et al.)",
        "Genetic Alg. (Shehzad et al.)", "GSPA (proposed)"
    ]

    lower_bounds = np.ceil(
        num_users * min_user_rate / max_uav_total_rate) * np.ones(
            len(l_varied_parameters))

    m_num_uavs = np.zeros((len(l_placer_names), len(l_varied_parameters)))

    for ind_name in range(len(l_placer_names)):

        placer_name = l_placer_names[ind_name]

        for ind_parameter in range(len(l_varied_parameters)):

            if l_varied_parameters[ind_parameter] == 0.5 or l_varied_parameters[
                    ind_parameter] == 1.5:
                exp_type = str(l_varied_parameters[ind_parameter]) + post_fix
            else:
                exp_type = str(l_varied_parameters[ind_parameter]) + post_fix

            for ind_env in range(start_mc, start_mc + num_mc_iter):

                load_at = "output/" + "exp_" + exp_num + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                    ind_env) + ".pck"

                with open(load_at, "rb") as f:
                    l_uavs = pickle.load(f)

                m_num_uavs[ind_name, ind_parameter] += len(l_uavs)

    m_num_uavs = m_num_uavs / num_mc_iter

    plt.plot(l_varied_parameters,
             lower_bounds,
             label="Lower bound",
             linestyle='dashed',
             color="black",
             marker='d')

    for ind_name in range(len(l_placer_names)):
        plt.plot(l_varied_parameters,
                 m_num_uavs[ind_name],
                 marker=l_markers[ind_name],
                 label=l_labels[ind_name])

    plt.xlabel(plt_label_x)
    plt.xticks(l_varied_parameters)
    plt.yticks(plt_ticks_y)
    plt.ylabel('Mean number of ABSs')
    plt.legend(loc=2)
    plt.grid()
    plt.show()