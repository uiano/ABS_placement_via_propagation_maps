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
    print('[placement_using_channel_maps.py] CVXOPT not installed. ')
import scipy

from common.solvers import group_sparse_cvx, weighted_group_sparse_scipy
from common.solvers import group_sparsify as solvers_group_sparsify

import gsim
from gsim.gfigure import GFigure
from common.utilities import dB_to_natural, dbm_to_watt, empty_array, natural_to_dB, watt_to_dbW, watt_to_dbm
from common.grid import RectangularGrid3D
from channels.channel import Channel, FreeSpaceChannel
from channels.tomographic_channel import TomographicChannel
from common.environment import BlockUrbanEnvironment1, BlockUrbanEnvironment2, GridBasedBlockUrbanEnvironment, UrbanEnvironment, Building

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
    def experiment_1000(l_args):
        print("Test experiment")

        return

    """###################################################################
    10. Preparatory experiments
    ###################################################################

    EXPERIMENT -------------------------------------------

    Channel map associated with a single source in free space.

    """

    def experiment_1001(l_args):

        # Grid
        area_len = [100, 80, 50]
        grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])

        # Free-space channel
        channel = FreeSpaceChannel(freq_carrier=3e9)
        pt_tx = grid.random_pts(z_val=0)[0]
        print(f"pt_tx = {pt_tx}")
        fl_path_loss = channel.dbgain_from_pt(grid=grid, pt_1=pt_tx)

        # Map at different heights
        F = fl_path_loss.plot_z_slices(zvals=[1, 7.5, 20, 40])

        return F

    """ EXPERIMENT -------------------------------------------

    Plot of two buildings.

    """

    def experiment_1002(l_args):

        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])

        env.plot()
        env.show()

    """ EXPERIMENT -------------------------------------------

    Approximation of a line integral.

    """

    def experiment_1003(l_args):
        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])

        pt_tx = np.array([50, 60, 37])
        pt_rx = np.array([19, 1, 0])
        print("points = ", [pt_tx, pt_rx])

        li = env.slf.line_integral(pt_tx, pt_rx, mode="python")
        print("line integral (Python) = ", li)

        li = env.slf.line_integral(pt_tx, pt_rx, mode="c")
        print("line integral (C) = ", li)

        env.dl_uavs = {'tx-rx': [pt_tx, pt_rx]}
        env.l_lines = [[pt_tx, pt_rx]]
        env.plot()
        env.show()

    """ EXPERIMENT -------------------------------------------

    Absorption and channel gain vs. position of the UAV for a single ground user.

    """

    def experiment_1004(l_args):
        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            #max_link_capacity=10,
        )
        #channel = TomographicChannel(grid=grid, slf=env.slf)

        pt_rx = np.array([80, 40, 0])
        pt_tx_start = np.array([1, 1, 1])
        pt_tx_end = np.array([1, 70, 40])
        v_t = np.linspace(0, 1, 1000)

        # Path loss vs. position --> the transmitter moves
        l_pt_tx = [pt_tx_start + t * (pt_tx_end - pt_tx_start) for t in v_t]
        absorption_loss = [
            channel.dbabsorption(pt_tx, pt_rx) for pt_tx in l_pt_tx
        ]
        free_space_gain = [
            channel.dbgain_free_space(pt_tx, pt_rx) for pt_tx in l_pt_tx
        ]
        total_gain = [channel.dbgain(pt_tx, pt_rx) for pt_tx in l_pt_tx]

        env.dl_uavs = {'rx': [pt_rx]}
        env.l_lines = [[pt_tx_start, pt_tx_end]]

        env.plot()
        env.show()

        F = GFigure(xaxis=v_t,
                    yaxis=absorption_loss,
                    xlabel="t",
                    ylabel="Absorption Loss [dB]")
        F.next_subplot(xaxis=v_t,
                       yaxis=free_space_gain,
                       xlabel="t",
                       ylabel="Free Space Gain [dB]")
        F.next_subplot(xaxis=v_t,
                       yaxis=total_gain,
                       xlabel="t",
                       ylabel="Total Gain [dB]")
        return F

    """ EXPERIMENT -------------------------------------------

    Channel gain map for a single ground user.

    """

    def experiment_1005(l_args):

        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            #max_link_capacity=10,
        )

        pt_rx = np.array([19, 1, 0])
        map = channel.dbgain_from_pt(grid=env.slf.grid, pt_1=pt_rx)
        print("number of grid points = ", map.t_values.size)

        env.l_users = [pt_rx]
        env.plot()
        env.show()

        return map.plot_z_slices(zvals=[0, 20, 30, 40])

    """ EXPERIMENT -------------------------------------------

    Optimal placement of a single UAV for communicating with two users on the
    ground. 

    Good illustration of the objects involved in these simulations.

    """

    def experiment_1006(l_args):

        area_len = [100, 80, 50]
        fly_grid = FlyGrid(area_len=area_len,
                           num_pts=[10, 11, 7],
                           min_height=10)
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[20, 30, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(slf=env.slf)

        env.l_users = np.array([[10, 55, 2], [60, 60, 2]])

        pl = SingleUAVPlacer(criterion="max_min_rate")

        env.dl_uavs = {
            pl.name:
            pl.place(fly_grid=env.fly_grid,
                     channel=channel,
                     user_coords=env.l_users)
        }

        l_F = pl.plot_capacity_maps(fly_grid=fly_grid,
                                    channel=channel,
                                    user_coords=env.l_users)
        #map = channel.dbgain_from_pt(pt_1 = pt_rx_2)
        #print("number of grid points = ", map.t_values.size)

        env.plot()
        env.show()

        return l_F

    """ EXPERIMENT -------------------------------------------

    Tests with specific UrbanEnvironments.

    """

    def experiment_1007(l_args):

        # Base environment
        if False:
            area_len = [100, 80, 50]
            fly_grid = FlyGrid(area_len=area_len,
                               num_pts=[10, 11, 7],
                               min_height=10)
            env = UrbanEnvironment(area_len=area_len,
                                   num_pts_slf_grid=[20, 30, 5],
                                   base_fly_grid=fly_grid,
                                   buildings=[
                                       Building(sw_corner=[30, 50, 0],
                                                ne_corner=[50, 70, 0],
                                                height=70),
                                       Building(sw_corner=[20, 20, 0],
                                                ne_corner=[30, 30, 0],
                                                height=20),
                                   ])
        if True:
            env = BlockUrbanEnvironment1(num_pts_slf_grid=[20, 30, 5],
                                         num_pts_fly_grid=[8, 8, 3],
                                         min_fly_height=10,
                                         building_height=None,
                                         building_absorption=1)

        if False:
            env = BlockUrbanEnvironment2(num_pts_slf_grid=[20, 30, 5],
                                         num_pts_fly_grid=[10, 10, 3],
                                         min_fly_height=10,
                                         building_height=50,
                                         building_absorption=1)

        if False:
            env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                                 num_pts_slf_grid=[20, 30, 5],
                                                 num_pts_fly_grid=[9, 9, 3],
                                                 min_fly_height=50,
                                                 building_absorption=1)

        # Test to determine the dimensions of the area and comm parameters
        if True:
            freq_carrier = 2.4e9
            bandwidth = 20e6
            target_rate = 5e6
            min_snr = natural_to_dB(2**(target_rate / bandwidth) - 1)
            tx_dbpower = watt_to_dbW(.1)
            dbgain = Channel.dist_to_dbgain_free_space(500,
                                                       wavelength=3e8 /
                                                       freq_carrier)
            max_noise_dbpower = tx_dbpower + dbgain - min_snr

            channel = TomographicChannel(
                slf=env.slf,
                freq_carrier=freq_carrier,
                tx_dbpower=tx_dbpower,
                noise_dbpower=max_noise_dbpower,
                bandwidth=bandwidth,
                min_link_capacity=2,
                max_link_capacity=7,
            )

            max_dist = channel.max_distance_for_rate(min_rate=15e6)
            ground_radius = np.sqrt(max_dist**2 -
                                    env.fly_grid.min_enabled_height**2)
            print(f"ground_radius = {ground_radius}")

        env.plot()
        env.show()

        return

    """ EXPERIMENT -------------------------------------------

    Channel gain map for a single ground user. Comparison between the C and
    Python implementations.

    """

    def experiment_1008(l_args):

        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            #max_link_capacity=10,
        )

        pt_rx = np.array([19, 1, 0])

        channel.integral_mode = 'python'
        map_python = channel.dbgain_from_pt(grid=env.slf.grid, pt_1=pt_rx)

        import timeit
        times = 10
        t_python = timeit.timeit(
            lambda: channel.dbgain_from_pt(grid=env.slf.grid, pt_1=pt_rx),
            number=times) / times
        print("time Python = ", t_python)

        channel.integral_mode = 'c'
        map_c = channel.dbgain_from_pt(grid=env.slf.grid, pt_1=pt_rx)

        t_c = timeit.timeit(
            lambda: channel.dbgain_from_pt(grid=env.slf.grid, pt_1=pt_rx),
            number=times) / times
        print("time C = ", t_c)

        print("number of grid points = ", map_c.t_values.size)
        print("|map_c - map_python| = ",
              np.linalg.norm(map_c.t_values - map_python.t_values))

        env.l_users = [pt_rx]
        env.plot()
        env.show()

        return [
            map_python.plot_z_slices(zvals=[0, 20, 30, 40]),
            map_c.plot_z_slices(zvals=[0, 20, 30, 40])
        ]

    """###################################################################
    11. Optimization experiments
    ###################################################################
    """
    """ EXPERIMENT -------------------------------------------

    "Hello world" test with CVX.

    """

    def experiment_1101(l_args):
        """ minimize_x c^T * x
            s.t.       A *x <= b
        """

        num_constr = 6
        num_vars = 4
        A = np.random.random((num_constr, num_vars))
        b = np.random.random((num_constr, ))
        c = np.random.random((num_vars, ))

        p = co.solvers.lp(m(c), m(A), m(b))

        print("solution: x = \n", p['x'])

        return

    """ EXPERIMENT -------------------------------------------

    Check if we can recover sparse solutions with CVX.

    """

    def experiment_1102(l_args):
        """ minimize_x  |x|_1
            s.t.       A *x = b
                       x >= 0

            Whether the problem is feasible depends on the realization.
        """

        num_vars = 8
        num_constr = 2
        G = -np.eye(num_vars)
        h = np.zeros((num_vars, ))
        A = np.random.random((num_constr, num_vars))
        #b = A[:,2] + A[:,4] + A[:,5]
        b = np.random.random((num_constr, ))
        c = np.ones((num_vars, ))

        p = co.solvers.lp(m(c), m(G), m(h), A=m(A), b=m(b))

        if p['x'] is None:
            print("No solution found")
            return

        x_opt = um(p['x'])
        x_opt_sp = sparsify(x_opt)
        print('Sparsified solution:')
        print(x_opt_sp)
        print('Residual                      = ',
              np.linalg.norm(A @ x_opt - b))
        print('Residual after sparsification = ',
              np.linalg.norm(A @ x_opt_sp - b))
        return

    """ EXPERIMENT -------------------------------------------

    Similar, but the problem should always be feasible.

    """

    def experiment_1103(l_args):
        """ minimize_x  |x|_1
            s.t.       A *x >= b
                       x >= 0
            
        """

        num_vars = 8
        num_constr = 70
        G0 = -np.eye(num_vars)
        h0 = np.zeros((num_vars))
        A = np.random.random((num_constr, num_vars))
        #b = A[:,2] + A[:,4] + A[:,5]
        b = np.random.random((num_constr))
        G = np.concatenate((G0, -A), axis=0)
        h = np.concatenate((h0, -b), axis=0)

        c = np.ones((num_vars, ))

        p = co.solvers.lp(m(c), m(G), m(h))

        if p['x'] is None:
            print("No solution found")
            return

        x_opt = um(p['x'])
        x_opt_sp = sparsify(x_opt)
        print('[Sparsified solution, non-sparsified solution]:')
        print(np.concatenate((x_opt_sp, x_opt), axis=1))
        print('Residual                      = ',
              np.linalg.norm(A @ x_opt - b))
        print('Residual after sparsification = ',
              np.linalg.norm(A @ x_opt_sp - b))
        return

    """ EXPERIMENT -------------------------------------------

    Now, let us try to promote group sparsity --> It seems that it works.

    This experiment was used to develop common.solvers.group_sparse

    """

    def experiment_1104(l_args):
        """ minimize_{y}  \sum_m |y_m|_2
            s.t.       E * y >= f
                       
            where y = [y_1; y_2;...;y_M] and each of the y_m has N entries.
        """

        # Problem definition
        num_vars_per_group = 5
        num_groups = 100
        num_constr = 214
        method = "primal"
        enforce_positivity = False
        study_output = False
        E = np.random.random((num_constr, num_vars_per_group * num_groups))
        f = np.random.random((num_constr))

        y_opt, status = group_sparse_cvx(E,
                                         f,
                                         method=method,
                                         num_vars_per_group=num_vars_per_group,
                                         enforce_positivity=enforce_positivity,
                                         study_output=True)

        print("Output status = ", status)
        return

    """ EXPERIMENT -------------------------------------------

    This experiment is to develop the ADMM solver for SparseUAVPlacer.

    """

    def experiment_1110(l_args):

        # 1. Problem data
        num_users = 100
        num_gridpts = 200
        m_capacity = np.random.random((num_users, num_gridpts))
        min_user_rate = 0.3
        v_weights = np.random.random((num_gridpts, ))

        assert all(np.sum(m_capacity, axis=1) >= min_user_rate), "Infeasible"

        # 2. Solution with linprog
        if False:
            m_A = -np.tile(np.eye(num_users), reps=(1, num_gridpts))
            v_b = -np.tile([min_user_rate], (num_users, ))
            R_opt, status = weighted_group_sparse_scipy(v_weights,
                                                        m_A,
                                                        v_b,
                                                        m_capacity,
                                                        thinning=False)

            print("R_opt=", R_opt)
            print("Objective interior point = ",
                  np.max(np.abs(R_opt), axis=0) @ v_weights)

        # 3. Solution with ADMM --> notes-cartography.pdf 2021/08/20
        def X_step(v_weights, m_Z, m_U, admm_step):
            def r_minimizer(weight, v_z, v_u):
                # To do: replaze v_z - v_u with a single variable
                def f(s):
                    return np.sum(np.maximum(v_z - v_u - s,
                                             0)) - weight / admm_step

                a = np.min(v_z - v_u) - weight / num_users / admm_step
                b = np.max(v_z - v_u) - weight / num_users / admm_step
                sg = scipy.optimize.bisect(f, a - margin_ab, b + margin_ab)
                return np.minimum(sg, v_z - v_u)

            l_R = [
                r_minimizer(v_weights[ind_gridpt], m_Z[:, ind_gridpt],
                            m_U[:, ind_gridpt])
                for ind_gridpt in range(num_gridpts)
            ]

            return np.array(l_R).T

        def X_step_cvx(v_weights, m_Z, m_U, admm_step):
            # for debugging purposes

            r = Runner("placement", "sparsity.m")
            data_in = OrderedDict()
            data_in["v_weights"] = v_weights
            data_in["m_Z"] = m_Z
            data_in["m_U"] = m_U
            data_in["admm_step"] = admm_step

            return r.run("x_step", data_in)[0]

        def Z_step(min_user_rate, m_capacity, m_R, m_U):
            def z_minimizer(min_user_rate, v_capacity, v_rate, v_u):
                def f(lam):
                    return np.sum(
                        np.maximum(0,
                                   np.minimum(v_capacity, v_rate_plus_u -
                                              lam))) - min_user_rate

                v_rate_plus_u = v_rate + v_u
                a = np.min(v_rate_plus_u - v_capacity)
                b = np.max(
                    v_rate_plus_u[v_capacity > min_user_rate /
                                  num_gridpts]) - min_user_rate / num_gridpts
                lam = scipy.optimize.bisect(f, a - margin_ab, b + margin_ab)
                return np.maximum(0, np.minimum(v_capacity,
                                                v_rate_plus_u - lam))

            l_Z = [
                z_minimizer(min_user_rate, m_capacity[ind_user], m_R[ind_user],
                            m_U[ind_user]) for ind_user in range(num_users)
            ]
            return np.array(l_Z)

        def Z_step_cvx(min_user_rate, m_capacity, m_R, m_U):
            # for debugging purposes

            r = Runner("placement", "sparsity.m")
            data_in = OrderedDict()
            data_in["min_user_rate"] = min_user_rate
            data_in["m_capacity"] = m_capacity
            data_in["m_R"] = m_R
            data_in["m_U"] = m_U

            return r.run("z_step", data_in)[0]

        margin_ab = 1e-5
        admm_step = 5
        num_iter = 50
        m_U = np.zeros((num_users, num_gridpts))
        m_Z = m_U
        l_obj = []
        l_feas = []
        l_min_rate = []
        l_change = []
        for ind_iter in range(0, num_iter):
            m_R = X_step(v_weights, m_Z, m_U, admm_step)
            # # debugging
            # m_R_cvx = X_step_cvx(v_weights, m_Z, m_U, admm_step)

            m_Z = Z_step(min_user_rate, m_capacity, m_R, m_U)
            # debugging
            #m_Z = Z_step_cvx(min_user_rate, m_capacity, m_R, m_U)
            #print("diff = ", np.linalg.norm(m_Z_non_cvx - m_Z))

            m_U = m_U + m_R - m_Z
            #print("feasibility -> ", np.linalg.norm(m_R - m_Z))
            l_obj.append(np.max(np.abs(m_R), axis=0) @ v_weights)
            l_feas.append(np.linalg.norm(m_R - m_Z))
            l_min_rate.append(np.min(np.sum(m_R, axis=1)))
            if ind_iter > 0:
                l_change.append(np.linalg.norm(m_R - m_R_old))
            m_R_old = m_R

            print("Objective ADMM = ", np.max(np.abs(m_R), axis=0) @ v_weights)

        G = GFigure(yaxis=l_obj, title="Objective")
        G.next_subplot(title="norm (R-Z)", yaxis=l_feas)
        G.next_subplot(title="norm (R^{k} - R^{k-1})", yaxis=l_change)
        G.next_subplot(title="min_rate",
                       yaxis=l_min_rate,
                       xlabel="Iteration",
                       ylim=[0, 1])

        return G

    """###################################################################
    20. Placement of multiple UAVs
    ###################################################################
    """
    """ EXPERIMENT -------------------------------------------

    Playground to run tests.

    """

    def experiment_2001(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        # channel = TomographicChannel(
        #     slf=env.slf,
        #     tx_dbpower=90,
        #     min_link_capacity=2,
        #     max_link_capacity=min_user_rate,
        # )

        pl_gs = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                     criterion="min_uav_num",
                                     min_user_rate=min_user_rate,
                                     max_uav_total_rate=100)
        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_gr = GridRatePlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate)
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        # # Choose:
        place_and_plot(environment=env,
                       channel=channel,
                       min_user_rate=min_user_rate,
                       l_placers=[pl_s, pl_km, pl_sr, pl_sp],
                       num_users=40)
        #d_out = mean_num_uavs(environment=env, channel=channel, min_user_rate=min_user_rate, l_placers=[pl_sp, pl_gr], num_users=135, num_mc_iter=3)
        #
        # d_out = user_loc_mc(env,
        #                     channel,
        #                     l_placers=[pl_sr, pl_km],
        #                     num_users=12,
        #                     min_user_rate=min_user_rate,
        #                     num_mc_iter=3)
        # print("output=", d_out)

    # beautiful illustration of placement
    # ICASSP
    def experiment_2002(l_args):
        #np.random.seed(2021)

        env = BlockUrbanEnvironment1(num_pts_slf_grid=[20, 30, 5],
                                     num_pts_fly_grid=[8, 8, 3],
                                     min_fly_height=10,
                                     building_height=None,
                                     building_absorption=3)

        # Set to None one of the following
        min_user_rate = 15e6
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
            disable_gridpts_by_dominated_verticals=False,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        # channel = TomographicChannel(
        #     slf=env.slf,
        #     tx_dbpower=90,
        #     min_link_capacity=2,
        #     max_link_capacity=min_user_rate,
        # )

        pl_gs = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                     criterion="min_uav_num",
                                     min_user_rate=min_user_rate,
                                     max_uav_total_rate=100)
        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_gr = GridRatePlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate)
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        # # Choose:
        place_and_plot(
            environment=env,
            channel=channel,
            min_user_rate=min_user_rate,
            #l_placers=[pl_s, pl_km, pl_sr, pl_sp],
            l_placers=[pl_s],
            num_users=90,
            disable_flying_gridpts_by_dominated_verticals=False,
            no_axes=True)
        #d_out = mean_num_uavs(environment=env, channel=channel, min_user_rate=min_user_rate, l_placers=[pl_sp, pl_gr], num_users=135, num_mc_iter=3)
        #
        # d_out = user_loc_mc(env,
        #                     channel,
        #                     l_placers=[pl_sr, pl_km],
        #                     num_users=12,
        #                     min_user_rate=min_user_rate,
        #                     num_mc_iter=3)
        # print("output=", d_out)

    def experiment_2031(l_args):
        """ This experiment is to compare
        field_line_integral_c       vs.     field_line_integral_python     
        """

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        # min_user_rate = 5
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        pt1 = np.array([391.25, 251.20, 0.0])
        pt2 = np.array([1, 2, 50])

        for ind in range(200):
            integral_c = channel.slf.grid.field_line_integral_c(
                channel.slf.t_values, pt1, pt2)
            integral_python = channel.slf.grid.field_line_integral_python(
                channel.slf.t_values, pt1, pt2)
            print("integral_c: {}; integral_python: {}".format(
                integral_c, integral_python))

        return

    """ EXPERIMENT -------------------------------------------

    Num UAVs to guarantee a minimum rate vs. num users.

    """

    def experiment_2010(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_user_rate = 5e6  # 15e6

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        v_num_users = [10, 15, 30, 50, 70, 90]
        d_out = metrics_vs_num_users(
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s, pl_srec],
            #l_placers=[pl_sp],
            v_num_users=v_num_users,
            min_user_rate=min_user_rate,
            num_mc_iter=30 * 2)

        G = GFigure(
            xlabel="Number of users",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_num_users,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # scaling by the sqrt of the distance (NeSh scaling) and greater absorption
    # -> ICASSP
    def experiment_2011(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=3)

        min_user_rate = 5e6  # 15e6

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96,
                                     min_link_capacity=1e6,
                                     max_link_capacity=min_user_rate,
                                     nesh_scaling=True)

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        v_num_users = [10, 15, 30, 50, 70, 90]
        d_out = metrics_vs_num_users(
            environment=env,
            channel=channel,
            l_placers=[pl_srec, pl_km, pl_sp, pl_sr, pl_s],
            #l_placers=[pl_sp],
            v_num_users=v_num_users,
            min_user_rate=min_user_rate,
            num_mc_iter=60)  # 15/hour

        G = GFigure(
            xlabel="Number of users",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            legend_loc="upper left",
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_num_users,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    """ EXPERIMENT -------------------------------------------

    Num UAVs to guarantee a minimum rate vs. building height.

    """

    def experiment_2020(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 60, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 5],
                                           num_pts_fly_grid=[9, 9, 3],
                                           min_fly_height=50,
                                           building_absorption=1,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 5e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=600)

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # higher rate
    def experiment_2021(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 60, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 5],
                                           num_pts_fly_grid=[9, 9, 3],
                                           min_fly_height=50,
                                           building_absorption=1,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 20e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=1800)

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # denser slf grid along z
    def experiment_2022(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 60, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 150],
                                           num_pts_fly_grid=[9, 9, 3],
                                           min_fly_height=50,
                                           building_absorption=1,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 20e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=300)

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # scaling by the sqrt of the distance (NeSh scaling) and greater absorption
    # -> ICASSP
    def experiment_2023(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 45, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 150],
                                           num_pts_fly_grid=[9, 9, 5],
                                           min_fly_height=50,
                                           building_absorption=3,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 20e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
                nesh_scaling=True,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sp, pl_sr, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=180)  # 100/hour

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    """ EXPERIMENT -------------------------------------------

    Num UAVs to guarantee a minimum rate vs. the minimum rate.

    """

    def experiment_2030(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )
        #pl_srec = SparseRecoveryPlacer(min_user_rate=v_min_user_rate[0])

        num_users = 40
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More users
    def experiment_2032(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # Denser SLF grid and less noise
    def experiment_2033(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-100,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # Even less noise
    def experiment_2034(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # Less absorption
    def experiment_2035(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=.1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More absorption -> Iterate this for 400 MC (5 h)
    def experiment_2036(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=.5)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More absorption, and NeSh scaling -> Iterate this for 400 MC (5 h)
    def experiment_2037(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=3)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,  #-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
            nesh_scaling=True)

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=80)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More absorption, more gridpts, and NeSh scaling -> Iterate this for 400 MC (5 h)
    # ICASSP
    def experiment_2038(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=3)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,  #-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
            nesh_scaling=True)

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 80
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)  # 20/35 min

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    """###################################################################
    30. Placement of multiple UAVs with limited backhaul
    ###################################################################
    """

    # compate admm and scipy
    def experiment_3001(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[5, 5, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        # min_user_rate = 5
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        pl_gs_scipy = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                           criterion="min_uav_num",
                                           min_user_rate=min_user_rate,
                                           max_uav_total_rate=100e6,
                                           backend="scipy",
                                           reweighting_num_iter=1)

        pl_gs_admm = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                          criterion="min_uav_num",
                                          min_user_rate=min_user_rate,
                                          max_uav_total_rate=100e6,
                                          backend="admm",
                                          reweighting_num_iter=1)

        # Choose:
        place_and_plot(environment=env,
                       channel=channel,
                       min_user_rate=min_user_rate,
                       l_placers=[pl_gs_admm, pl_gs_scipy],
                       num_users=3)

    # comparison admm and scipy in one backend comparison
    def experiment_3002(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        # min_user_rate = 5
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        pl_gs_compare = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                             criterion="min_uav_num",
                                             min_user_rate=min_user_rate,
                                             max_uav_total_rate=100e6,
                                             backend="comparison",
                                             reweighting_num_iter=1)

        # Choose:
        place_and_plot(environment=env,
                       channel=channel,
                       min_user_rate=min_user_rate,
                       l_placers=[pl_gs_compare],
                       num_users=90)

    def experiment_3003(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        # min_user_rate = 5
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        pl_gs_compare = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                             criterion="min_uav_num",
                                             min_user_rate=min_user_rate,
                                             max_uav_total_rate=100e6,
                                             backend="comparison",
                                             reweighting_num_iter=1)

        # Choose:
        results = mean_num_uavs(environment=env,
                                channel=channel,
                                min_user_rate=min_user_rate,
                                l_placers=[pl_gs_compare],
                                num_users=10)

        print("end of experiments")

    # plot from saved results
    def experiment_3004(l_args):

        # exp_3007      min = 15        max = 45
        # exp_3008      min =  5        max = 45
        # exp_3009      min = 25        max = 45
        # exp_3010      min = 20        max = 45
        # exp_3011      min = 10        max = 45
        num_mc_iter = 100
        l_num_users = np.array([10, 30, 50, 70, 90])
        min_user_rate = 5e6
        max_uav_total_rate = 45e6
        exp_num = "3008"

        lower_bounds = np.ceil(l_num_users * min_user_rate /
                               max_uav_total_rate)

        l_placer_names = [
            "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
            "GroupSparseUAVPlacer"
        ]

        l_markers = ["o", "^", "*", "v"]
        m_num_uavs = np.zeros((len(l_num_users), len(l_placer_names)))

        for ind_name in range(len(l_placer_names)):
            for ind_num_user in range(len(l_num_users)):

                exp_type = str(l_num_users[ind_num_user]) + "_users"

                for ind_env in range(0, num_mc_iter):

                    num_users = l_num_users[ind_num_user]
                    placer_name = l_placer_names[ind_name]

                    load_at = "output/received_results/exp_" + exp_num + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                        ind_env) + ".pck"

                    with open(load_at, "rb") as f:
                        l_uavs = pickle.load(f)

                    # print(
                    #     f"{l_placer_names[ind_name]};    Users: {l_num_users[ind_num_user]};    Env: {ind_env};    Num_uavs: {len(l_uavs)}"
                    # )

                    if l_uavs[0] is None:
                        print(
                            f"Placer: {ind_name},    users: {ind_num_user},    env: {ind_env}"
                        )

                    m_num_uavs[ind_num_user, ind_name] += len(l_uavs)
                    # to plot experiment_3007, use the following line
                    # m_num_uavs[ind_num_user, ind_name] += len(l_uavs[0])

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            "KMeansPlacer", "SpaceRateKMeans (Hammouti et al.)",
            "GeneticPlacer (Shehzad et al.)", "GroupSparseUAVPlacer (proposed)"
        ]

        plt.plot(l_num_users,
                 lower_bounds,
                 label="Lower bound",
                 linestyle='dashed',
                 color="black")

        for ind_name in range(len(l_placer_names)):
            plt.plot(l_num_users,
                     m_num_uavs[:, ind_name],
                     marker=l_markers[ind_name],
                     label=l_labels[ind_name])

        plt.xlabel('Number of users')
        plt.xticks(l_num_users)
        plt.yticks(np.linspace(10, 90, 9))
        plt.ylabel('Mean number of ABSs')
        plt.legend()
        plt.grid()
        plt.show()

        print('end of experiment')

    # plot from saved results
    def experiment_3004_1(l_args):

        num_mc_iter = 50
        num_users = 30
        l_min_user_rate = np.array([5, 10, 15, 20, 25])
        l_max_uav_total_rate = np.array([5, 15, 25, 35])
        l_exp_num = ["3012", "3013", "3014", "3015"]

        l_lower_bounds = [
            np.ceil(num_users * l_min_user_rate / max_uav_total_rate)
            for max_uav_total_rate in l_max_uav_total_rate
        ]

        placer_name = "GroupSparseUAVPlacer"

        l_markers = ["o", "^", "*", "v"]
        m_num_uavs = np.zeros(
            (len(l_max_uav_total_rate), len(l_min_user_rate)))

        for ind_max_rate in range(len(l_max_uav_total_rate)):

            for ind_min_rate in range(len(l_min_user_rate)):

                exp_type = str(int(l_min_user_rate[ind_min_rate])) + "_minRate"

                for ind_env in range(0, num_mc_iter):

                    min_rate = l_min_user_rate[ind_min_rate]

                    load_at = "output/received_results/exp_" + l_exp_num[
                        ind_max_rate] + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                            ind_env) + ".pck"

                    with open(load_at, "rb") as f:
                        l_uavs = pickle.load(f)

                    # print(
                    #     f"{l_placer_names[ind_name]};    Users: {l_num_users[ind_num_user]};    Env: {ind_env};    Num_uavs: {len(l_uavs)}"
                    # )

                    if l_uavs[0] is None:
                        print(
                            f"Max_uav_total_rate: {ind_max_rate},    users: {ind_min_rate},  env: {ind_env}"
                        )

                    m_num_uavs[ind_max_rate, ind_min_rate] += len(l_uavs)
                    # to plot experiment_3007, use the following line
                    # m_num_uavs[ind_num_user, ind_name] += len(l_uavs[0])

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            "Max_uav_total_rate = 5e6", "Max_uav_total_rate = 15e6",
            "Max_uav_total_rate = 25e6", "Max_uav_total_rate = 35e6",
            "Max_uav_total_rate = 45e6"
        ]

        plt.plot(l_min_user_rate,
                 l_lower_bounds[0],
                 linestyle='dashed',
                 color="black")

        plt.plot(l_min_user_rate,
                 l_lower_bounds[1],
                 label="Lower bound",
                 linestyle='dashed',
                 color="black")

        for ind_max_rate in range(len(l_max_uav_total_rate)):
            plt.plot(l_min_user_rate,
                     m_num_uavs[ind_max_rate],
                     marker=l_markers[ind_max_rate],
                     label=l_labels[ind_max_rate])

        plt.xlabel('Min user rate [Mbps]')
        plt.xticks(np.linspace(5, 25, 5))
        plt.yticks(np.linspace(10, 150, 8))
        plt.ylabel('Mean number of ABSs')
        plt.legend()
        plt.grid()
        plt.show()

        print('end of experiment')

    # plot from saved results
    def experiment_3004_2(l_args):

        num_mc_iter = 100
        num_users = 30
        l_min_user_rate = np.array([5, 10, 20, 25])
        max_uav_total_rate = 45
        l_exp_num = ["3008", "3011", "3010", "3009"]

        l_lower_bounds = np.ceil(num_users * l_min_user_rate /
                                 max_uav_total_rate)

        l_placer_names = [
            "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
            "GroupSparseUAVPlacer"
        ]

        l_markers = ["o", "^", "*", "v"]
        m_num_uavs = np.zeros((len(l_placer_names), len(l_min_user_rate)))
        exp_type = str(30) + "_users"

        for ind_name in range(len(l_exp_num)):

            for ind_placer in range(len(l_placer_names)):

                placer_name = l_placer_names[ind_placer]

                for ind_env in range(0, num_mc_iter):

                    load_at = "output/received_results/exp_" + l_exp_num[
                        ind_name] + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                            ind_env) + ".pck"

                    with open(load_at, "rb") as f:
                        l_uavs = pickle.load(f)

                    m_num_uavs[ind_placer, ind_name] += len(l_uavs)
                    # to plot experiment_3007, use the following line
                    # m_num_uavs[ind_num_user, ind_name] += len(l_uavs[0])

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            "KMeansPlacer", "SpaceRateKMeans (Hammouti et al.)",
            "GeneticPlacer (Shehzad et al.)", "GroupSparseUAVPlacer (proposed)"
        ]

        plt.plot(l_min_user_rate,
                 l_lower_bounds,
                 label="Lower bound",
                 linestyle='dashed',
                 color="black")

        for ind_name in range(len(l_exp_num)):
            plt.plot(l_min_user_rate,
                     m_num_uavs[ind_name],
                     marker=l_markers[ind_name],
                     label=l_labels[ind_name])

        plt.xlabel('Min user rate [Mbps]')
        plt.xticks(np.linspace(5, 25, 5))
        plt.yticks(np.linspace(10, 40, 4))
        plt.ylabel('Mean number of ABSs')
        plt.legend()
        plt.grid()
        plt.show()

        print('end of experiment')

    # This experiment tests a genetic placer.
    def experiment_3005(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        # min_user_rate = 5
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        max_uav_total_rate = 100e6
        # div operator
        num_links_per_uav = max_uav_total_rate // min_user_rate

        pl_gs_compare = GeneticPlacer(min_user_rate=min_user_rate,
                                      num_iter=20,
                                      max_links_per_uav=num_links_per_uav)

        # Choose:
        results = mean_num_uavs(environment=env,
                                channel=channel,
                                min_user_rate=min_user_rate,
                                l_placers=[pl_gs_compare],
                                num_users=10)

        print(results)

        print("end of experiments")

    # This experiment compares the genetic vs admm placer
    def experiment_3006(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            # max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        max_uav_total_rate = 3 * min_user_rate
        # div operator
        max_uav_users = max_uav_total_rate // min_user_rate

        # checked
        pl_gs = GroupSparseUAVPlacer(sparsity_tol=1e-1,
                                     criterion="min_uav_num",
                                     min_user_rate=min_user_rate,
                                     max_uav_total_rate=max_uav_total_rate,
                                     backend="admm",
                                     reweighting_num_iter=4)
        # checked
        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=20,
                              max_links_per_uav=max_uav_users)

        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate,
                                max_uav_total_rate=max_uav_total_rate)

        # Choose:
        results = mean_num_uavs(environment=env,
                                channel=channel,
                                min_user_rate=min_user_rate,
                                l_placers=[pl_gs],
                                num_users=30,
                                max_uav_total_rate=max_uav_total_rate)

        print(results)

        print("end of experiments")

    def experiment_3007(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 15e6
        max_uav_total_rate = 45e6
        num_users = 30

        run_new = False
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3007"
        exp_type = str(num_users) + "_users"

        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                     criterion="min_uav_num",
                                     min_user_rate=min_user_rate,
                                     max_uav_total_rate=max_uav_total_rate,
                                     backend="admm",
                                     reweighting_num_iter=20,
                                     admm_stepsize=1e-7,
                                     admm_max_num_iter=200,
                                     admm_initial_error_tol=5,
                                     eps_abs=1e-4,
                                     eps_rel=1e-4,
                                     b_save_group_weights=run_new,
                                     b_load_group_weights=not run_new)

        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=200,
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

        print("end of experiments")

    # Change min_user_rate = 5e6, change num_users
    def experiment_3008(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 5e6
        max_uav_total_rate = 45e6
        num_users = 10

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3008"
        exp_type = str(num_users) + "_users"

        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
            admm_stepsize=1e-7,
            admm_max_num_iter=200,
            admm_initial_error_tol=5,
            eps_abs=1e-4,
            eps_rel=1e-4,
            b_admm_decrease_err_tol=b_decrease_err_tol,
            b_save_group_weights=run_new,
            b_load_group_weights=not run_new)

        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=200,
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

        print("end of experiments")

    # Change min_user_rate = 25e6, change num_users
    def experiment_3009(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 25e6
        max_uav_total_rate = 45e6
        num_users = 90

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3009"
        exp_type = str(num_users) + "_users"

        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
            admm_stepsize=1e-7,
            admm_max_num_iter=200,
            admm_initial_error_tol=5,
            eps_abs=1e-4,
            eps_rel=1e-4,
            b_admm_decrease_err_tol=b_decrease_err_tol,
            b_save_group_weights=run_new,
            b_load_group_weights=not run_new)

        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=200,
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

        print("end of experiments")

    # Change min_user_rate = 20e6, change num_users
    def experiment_3010(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 20e6
        max_uav_total_rate = 45e6
        num_users = 90

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3010"
        exp_type = str(num_users) + "_users"

        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
            admm_stepsize=1e-7,
            admm_max_num_iter=200,
            admm_initial_error_tol=5,
            eps_abs=1e-4,
            eps_rel=1e-4,
            b_admm_decrease_err_tol=b_decrease_err_tol,
            b_save_group_weights=run_new,
            b_load_group_weights=not run_new)

        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=200,
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

        print("end of experiments")

    # Change min_user_rate = 10e6, change num_users
    def experiment_3011(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 10e6
        max_uav_total_rate = 45e6
        num_users = 90

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3011"
        exp_type = str(num_users) + "_users"

        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
            admm_stepsize=1e-7,
            admm_max_num_iter=200,
            admm_initial_error_tol=5,
            eps_abs=1e-4,
            eps_rel=1e-4,
            b_admm_decrease_err_tol=b_decrease_err_tol,
            b_save_group_weights=run_new,
            b_load_group_weights=not run_new)

        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=200,
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

        print("end of experiments")

    # This experiment tests the placers with different min_user_rates with max_uav_total_rates = 5e6
    def experiment_3012(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        l_min_user_rates = np.array([5, 10, 15, 20, 25]) * 1e6
        max_uav_total_rate = 5e6
        num_users = 30

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3012"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        for min_user_rate in l_min_user_rates:

            print(
                f"Max_uav_total_rate: {int(max_uav_total_rate/1e6)};     Min_user_rate: {int(min_user_rate/1e6)}"
            )

            exp_type = str(int(min_user_rate / 1e6)) + "_minRate"

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=20,
                admm_stepsize=1e-7,
                admm_max_num_iter=200,
                admm_initial_error_tol=5,
                eps_abs=1e-4,
                eps_rel=1e-4,
                b_admm_decrease_err_tol=b_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
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

        print("end of experiments")

    # This experiment tests the placers with different min_user_rates with max_uav_total_rates = 15e6
    def experiment_3013(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        l_min_user_rates = np.array([5, 10, 15, 20, 25]) * 1e6
        max_uav_total_rate = 15e6
        num_users = 30

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3013"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        for min_user_rate in l_min_user_rates:

            print(
                f"Max_uav_total_rate: {int(max_uav_total_rate/1e6)};     Min_user_rate: {int(min_user_rate/1e6)}"
            )

            exp_type = str(int(min_user_rate / 1e6)) + "_minRate"

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=20,
                admm_stepsize=1e-7,
                admm_max_num_iter=200,
                admm_initial_error_tol=5,
                eps_abs=1e-4,
                eps_rel=1e-4,
                b_admm_decrease_err_tol=b_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
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

        print("end of experiments")

    # This experiment tests the placers with different min_user_rates with max_uav_total_rates = 25e6
    def experiment_3014(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        l_min_user_rates = np.array([5, 10, 15, 20, 25]) * 1e6
        max_uav_total_rate = 25e6
        num_users = 30

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 1
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3014"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        for min_user_rate in l_min_user_rates:

            print(
                f"Max_uav_total_rate: {int(max_uav_total_rate/1e6)};     Min_user_rate: {int(min_user_rate/1e6)}"
            )

            exp_type = str(int(min_user_rate / 1e6)) + "_minRate"

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=20,
                admm_stepsize=1e-7,
                admm_max_num_iter=200,
                admm_initial_error_tol=5,
                eps_abs=1e-4,
                eps_rel=1e-4,
                b_admm_decrease_err_tol=b_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
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

        print("end of experiments")

    # This experiment tests the placers with different min_user_rates with max_uav_total_rates = 35e6
    def experiment_3015(l_args):

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        # l_min_user_rates = np.array([5, 10, 15, 20, 25]) * 1e6
        l_min_user_rates = np.array([5]) * 1e6
        max_uav_total_rate = 35e6
        num_users = 30

        b_decrease_err_tol = True

        run_new = True
        num_mc_iter = 100
        start_mc_iter = 0
        b_save_results = True

        exp_num = "3015"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        for min_user_rate in l_min_user_rates:

            print(
                f"Max_uav_total_rate: {int(max_uav_total_rate/1e6)};     Min_user_rate: {int(min_user_rate/1e6)}"
            )

            exp_type = str(int(min_user_rate / 1e6)) + "_minRate"

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=20,
                admm_stepsize=1e-7,
                admm_max_num_iter=200,
                admm_initial_error_tol=5,
                eps_abs=1e-4,
                eps_rel=1e-4,
                b_admm_decrease_err_tol=b_decrease_err_tol,
                b_save_group_weights=run_new,
                b_load_group_weights=not run_new)

            pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                                  max_num_gens=200,
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

        print("end of experiments")

    # Mean number of ABSs vs building_height
    def experiment_3021(l_args):

        # experiment parameters
        exp_num = "3021"
        save_exp_at = "output/exp_" + exp_num + "/"

        # building_height = 0, 10, 20, 30, 40, 50, 60, 70, 80
        l_building_height = [0, 10, 20, 30, 40, 50, 60, 70,
                             80]  # 20, 30, 40, 50, 60, 70, 80

        min_user_rate = 15e6
        max_uav_total_rate = 44e6
        num_users = 70

        num_mc_iter = 10
        start_mc_iter = 0
        run_new = True
        b_save_results = True

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

            b_admm_decrease_err_tol = True

            # simulation code
            if not os.path.exists(save_exp_at):
                os.mkdir(save_exp_at)

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=50,
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
            results = mean_num_uavs(
                env,
                channel,
                min_user_rate,
                l_placers=[pl_gs, pl_ge, pl_km, pl_sr],  # 
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

        print("end of experiments")

    # plot from saved results
    def experiment_3021_plot(l_args):
        """
            post_fix =  1. "_users",
                        2. "_minRate", 
                        3. "_maxRate", 
                        4. "_buildingHeight", 
                        5. "_minFlyHeight", 
                        6. "_buildAbsorp", 
                        7. "_numFlyHor", 
                        8. "_numFlyVer",                        
                        9. "_numSlfHor", 
                       10. "_numSlfVer",
        """

        exp_num = "3072"  # exp_3021_600mcs_70users_150slf _600mcs_70users_150slf
        post_fix = "_buildingHeight"
        plt_label_x = 'Height of the buildings [m]'
        l_varied_parameters = np.array([0, 10, 20, 30, 40, 50])
        plt_ticks_y = np.arange(12, 34, 2)
        num_mc_iter = 1000
        start_mc = 0
        num_users = 70  # Mpbs
        min_user_rate = 17  # Mpbs
        max_uav_total_rate = 84  # Mpbs

        # exp_num = "3082"
        # plt_label_x = 'Minimum flight height [m]'
        # post_fix = "_minFlyHeight"
        # l_varied_parameters = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        # # l_varied_parameters = np.array([10, 30, 50, 70, 90])
        # plt_ticks_y = np.arange(10, 60, 5)
        # num_mc_iter = 300
        # start_mc = 0
        # num_users = 70  # Mpbs
        # min_user_rate = 17  # Mpbs
        # max_uav_total_rate = 84  # Mpbs

        # exp_num = "3083"
        # plt_label_x = 'Building absorption [dB/m]'
        # post_fix = "_buildAbsorp"
        # l_varied_parameters = np.array(
        #     [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
        # plt_ticks_y = np.arange(10, 60, 5)
        # num_mc_iter = 200
        # start_mc = 0
        # num_users = 70  # Mpbs
        # min_user_rate = 17  # Mpbs
        # max_uav_total_rate = 84  # Mpbs

        # exp_num = "3024"
        # plt_label_x = 'Number of fly grid points in each axis x and y [x, x, 5]; x = 5, 7, 9, 11'
        # post_fix = "_numFlyHor"
        # l_varied_parameters = np.array([5, 7, 9, 11])
        # plt_ticks_y = np.linspace(12.5, 32.5, 9)
        # num_mc_iter = 50
        # start_mc = 0
        # num_users = 60  # Mpbs
        # min_user_rate = 17  # Mpbs
        # max_uav_total_rate = 84.5  # Mpbs

        # exp_num = "3025"
        # plt_label_x = 'Number of fly grid points in axis z [9, 9, x]; x = 2, 3, 4, 5, 6, 7, 8, 9'
        # post_fix = "_numFlyVer"
        # l_varied_parameters = np.array([2, 3, 4, 5, 6, 7, 8, 9])
        # plt_ticks_y = np.linspace(12.5, 42.5, 11)
        # num_mc_iter = 50
        # start_mc = 0
        # num_users = 60  # Mpbs
        # min_user_rate = 17  # Mpbs
        # max_uav_total_rate = 84.5  # Mpbs

        # exp_num = "3026"
        # plt_label_x = 'Number of SLF points in each axis x and y [x, x, 5]; x = 20, 25, 30, 35, 40, 45, 50, 55'
        # post_fix = "_numSlfHor"
        # l_varied_parameters = np.array([20, 25, 30, 35, 40, 45, 50, 55])
        # plt_ticks_y = np.linspace(12.5, 52.5, 13)
        # num_mc_iter = 50
        # start_mc = 0
        # num_users = 60  # Mpbs
        # min_user_rate = 17  # Mpbs
        # max_uav_total_rate = 84.5  # Mpbs

        # exp_num = "3027"
        # plt_label_x = 'Number of SLF points in axis z [48, 40, x]; x = 3, 4, 5, 6, 7, 8, 9, 10'
        # post_fix = "_numSlfVer"
        # l_varied_parameters = np.array([3, 4, 5, 6, 7, 8, 9, 10])
        # plt_ticks_y = np.linspace(12.5, 37.5, 6)
        # num_mc_iter = 50
        # start_mc = 0
        # num_users = 60  # Mpbs
        # min_user_rate = 17  # Mpbs
        # max_uav_total_rate = 84.5  # Mpbs

        # plt_ticks_y = np.linspace(10, 60, 6)

        # plot_vs_varied_env(start_mc=0,
        #                    num_mc_ter=50,
        #                    num_users=70,
        #                    min_user_rate_mbps=17,
        #                    max_uav_total_rate_mbps=101.5,
        #                    plt_ticks_y=None)

        l_placer_names = [
            "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
            "GroupSparseUAVPlacer"
        ]
        l_markers = ["o", "^", "*", "v"]
        l_labels = [
            "K-means Alg. (Galkin et al.)",
            "Space rate Alg. (Hammouti et al.)",
            "Genetic Alg. (Shehzad et al.)", "GSPA (proposed)"
        ]

        lower_bounds = np.ceil(
            num_users * min_user_rate / max_uav_total_rate) * np.ones(
                len(l_varied_parameters))

        m_num_uavs = np.zeros((len(l_placer_names), len(l_varied_parameters)))

        for ind_name in range(len(l_placer_names)):

            placer_name = l_placer_names[ind_name]

            for ind_parameter in range(len(l_varied_parameters)):

                if l_varied_parameters[
                        ind_parameter] == 0.5 or l_varied_parameters[
                            ind_parameter] == 1.5:
                    exp_type = str(
                        l_varied_parameters[ind_parameter]) + post_fix
                else:
                    exp_type = str(
                        l_varied_parameters[ind_parameter]) + post_fix

                for ind_env in range(start_mc, start_mc + num_mc_iter):

                    load_at = "output/received_results/" + "exp_" + exp_num + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                        ind_env) + ".pck"
                    # load_at = "output/" + "exp_" + exp_num + "/results/" + placer_name + "/" + exp_type + "_env_" + str(
                    #     ind_env) + ".pck"

                    with open(load_at, "rb") as f:
                        l_uavs = pickle.load(f)

                    # if placer_name == "SpaceRateKMeans" and l_varied_parameters[
                    #         ind_parameter] == 20 and len(l_uavs) < 25:
                    #     print(
                    #         f"Placer: {ind_name},    para: {ind_parameter},    env: {ind_env},    num_uavs: {len(l_uavs)}"
                    #     )

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

        print('end of experiment')

    # Mean number of ABSs vs min_fly_height
    def experiment_3022(l_args):

        # experiment parameters
        exp_num = "3022"
        save_exp_at = "output/exp_" + exp_num + "/"

        # min_fly_height = 10, 30, 50, 70, 90, 110
        min_fly_height = 110
        num_mc_iter = 50
        start_mc_iter = 0
        exp_type = str(int(min_fly_height)) + "_minFlyHeight"

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=min_fly_height,
                                             building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 17e6
        max_uav_total_rate = 84.5e6
        num_users = 70

        b_admm_decrease_err_tol = True
        run_new = True
        b_save_results = True

        # simulation code
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
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

        print("end of experiments")

    # Mean number of ABSs vs building_absorption
    def experiment_3023(l_args):

        # experiment parameters
        exp_num = "3023"
        save_exp_at = "output/exp_" + exp_num + "/"

        l_bd_absorption = [0.5, 1.5]  # 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5
        num_mc_iter = 50
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for bd_absorption in l_bd_absorption:

            exp_type = str(bd_absorption) + "_buildAbsorp"

            env = GridBasedBlockUrbanEnvironment(
                area_len=[500, 400, 150],
                num_pts_slf_grid=[48, 40, 5],
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
            max_uav_total_rate = 84.5e6
            num_users = 60

            b_admm_decrease_err_tol = True

            # simulation code
            if not os.path.exists(save_exp_at):
                os.mkdir(save_exp_at)

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=50,
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

        print("end of experiments")

    # Mean number of ABSs vs num_pts_fly_grid_xy
    def experiment_3024(l_args):

        # experiment parameters
        exp_num = "3024"
        save_exp_at = "output/exp_" + exp_num + "/"

        # num_pts_fly_grid_xy = 5, 7, 9, 11
        num_pts_fly_grid_xy = 11
        num_mc_iter = 50
        start_mc_iter = 0
        exp_type = str(int(num_pts_fly_grid_xy)) + "_numFlyHor"

        env = GridBasedBlockUrbanEnvironment(
            area_len=[500, 400, 150],
            num_pts_slf_grid=[48, 40, 5],
            num_pts_fly_grid=[num_pts_fly_grid_xy, num_pts_fly_grid_xy, 5],
            min_fly_height=50,
            building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 17e6
        max_uav_total_rate = 84.5e6
        num_users = 60

        b_admm_decrease_err_tol = True
        run_new = True
        b_save_results = True

        # simulation code
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
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

        print("end of experiments")

    # Mean number of ABSs vs num_pts_fly_grid_z
    def experiment_3025(l_args):

        # experiment parameters
        exp_num = "3025"
        save_exp_at = "output/exp_" + exp_num + "/"

        # num_pts_fly_grid_xy = 2, 3, 4, 5
        num_pts_fly_grid_z = 2
        num_mc_iter = 50
        start_mc_iter = 0
        exp_type = str(int(num_pts_fly_grid_z)) + "_numFlyVer"

        env = GridBasedBlockUrbanEnvironment(
            area_len=[500, 400, 150],
            num_pts_slf_grid=[48, 40, 5],
            num_pts_fly_grid=[9, 9, num_pts_fly_grid_z],
            min_fly_height=50,
            building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 17e6
        max_uav_total_rate = 84.5e6
        num_users = 60

        b_admm_decrease_err_tol = True
        run_new = True
        b_save_results = True

        # simulation code
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
            admm_stepsize=1e-7,
            admm_max_num_iter=200,
            admm_initial_error_tol=5,
            eps_abs=1e-4,
            eps_rel=1e-4,
            b_admm_decrease_err_tol=b_admm_decrease_err_tol,
            b_save_group_weights=run_new,
            b_load_group_weights=not run_new)

        pl_ge = GeneticPlacer(min_user_rate=min_user_rate,
                              max_num_gens=200,
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

        print("end of experiments")

    # Mean number of ABS vs num_pts_slf_grid_xy
    def experiment_3026(l_args):

        # experiment parameters
        exp_num = "3026"
        save_exp_at = "output/exp_" + exp_num + "/"

        # num_pts_slf_grid_xy = 10, 20, 30, 40, 50
        num_pts_slf_grid_xy = 50
        num_mc_iter = 24
        start_mc_iter = 26
        exp_type = str(int(num_pts_slf_grid_xy)) + "_numSlfHor"

        env = GridBasedBlockUrbanEnvironment(
            area_len=[500, 400, 150],
            num_pts_slf_grid=[num_pts_slf_grid_xy, num_pts_slf_grid_xy, 5],
            num_pts_fly_grid=[9, 9, 5],
            min_fly_height=50,
            building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96)

        # Set to None one of the following
        min_user_rate = 17e6
        max_uav_total_rate = 84.5e6
        num_users = 60

        b_admm_decrease_err_tol = True
        run_new = True
        b_save_results = True

        # simulation code
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        pl_gs = GroupSparseUAVPlacer(
            sparsity_tol=1e-2,
            criterion="min_uav_num",
            min_user_rate=min_user_rate,
            max_uav_total_rate=max_uav_total_rate,
            backend="admm",
            reweighting_num_iter=20,
            admm_stepsize=1e-7,
            admm_max_num_iter=200,
            admm_initial_error_tol=5,
            eps_abs=1e-4,
            eps_rel=1e-4,
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

        print("end of experiments")

    # Mean number of ABSs vs num_pts_slf_grid_z
    def experiment_3027(l_args):

        # experiment parameters
        exp_num = "3027"
        save_exp_at = "output/exp_" + exp_num + "/"

        # num_pts_slf_grid_z = 3, 4, 5, 6, 7, 8, 9, 10
        l_num_pts_slf_grid_z = [8]
        num_mc_iter = 50
        start_mc_iter = 0
        run_new = True
        b_save_results = True

        for num_pts_slf_grid_z in l_num_pts_slf_grid_z:

            exp_type = str(int(num_pts_slf_grid_z)) + "_numSlfVer"

            env = GridBasedBlockUrbanEnvironment(
                area_len=[500, 400, 150],
                num_pts_slf_grid=[48, 40, num_pts_slf_grid_z],
                num_pts_fly_grid=[9, 9, 5],
                min_fly_height=50,
                building_absorption=1)

            channel = TomographicChannel(slf=env.slf,
                                         freq_carrier=2.4e9,
                                         bandwidth=20e6,
                                         tx_dbpower=watt_to_dbW(.1),
                                         noise_dbpower=-96)

            # Set to None one of the following
            min_user_rate = 17e6
            max_uav_total_rate = 84.5e6
            num_users = 60

            b_admm_decrease_err_tol = True

            # simulation code
            if not os.path.exists(save_exp_at):
                os.mkdir(save_exp_at)

            pl_gs = GroupSparseUAVPlacer(
                sparsity_tol=1e-2,
                criterion="min_uav_num",
                min_user_rate=min_user_rate,
                max_uav_total_rate=max_uav_total_rate,
                backend="admm",
                reweighting_num_iter=50,
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

        print("end of experiments")

    # Mean number of ABSs vs min_user_rate
    def experiment_3028(l_args):

        # experiment parameters
        exp_num = "3028"
        save_exp_at = "output/exp_" + exp_num + "/"
        if not os.path.exists(save_exp_at):
            os.mkdir(save_exp_at)

        l_min_user_rates = np.array([7]) * 1e6  # 5,7,9,11,13,15,17
        l_max_uav_total_rates = np.array(
            [67.5]) * 1e6  # 33.5, 50.5, 67.5, 84.5, 101.5

        l_num_users = np.array([30])  # 10, 30, 50, 70, 90

        num_mc_iter = 1
        start_mc_iter = 32
        run_new = False
        b_save_results = False

        for num_users in l_num_users:

            for min_user_rate in l_min_user_rates:

                for max_uav_total_rate in l_max_uav_total_rates:

                    env = GridBasedBlockUrbanEnvironment(
                        area_len=[500, 400, 150],
                        num_pts_slf_grid=[50, 40, 15],
                        num_pts_fly_grid=[9, 9, 9],
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

        print("end of experiments")

    # plot from saved results, vs num users
    def experiment_3028_plot_01(l_args):
        """
        This experiment plot mean_num_uavs vs num_users
        
        """

        num_mc_iter = 400
        start_mc_iter = 0
        l_num_users = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        l_max_uav_total_rate = np.array(
            [99.])  #50.5, 59.5, 67.5, 74.5, 84.5, 89.5, 101.5
        l_min_user_rate = np.array([20.])

        lower_bounds = np.ceil(l_num_users * l_min_user_rate[0] /
                               l_max_uav_total_rate[0])

        l_placer_names = [
            "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
            "GroupSparseUAVPlacer"
        ]

        v_values = np.zeros(4)

        l_markers = ["o", "^", "*", "v"]
        m_num_uavs = np.zeros((len(l_placer_names), len(l_num_users)))

        for ind_users in range(len(l_num_users)):

            num_users = l_num_users[ind_users]

            for ind_min in range(len(l_min_user_rate)):

                minRate = l_min_user_rate[ind_min]

                for ind_max in range(len(l_max_uav_total_rate)):

                    maxRate = l_max_uav_total_rate[ind_max]

                    exp_type = str(num_users) + "_users_" + str(
                        minRate) + "_minRate_" + str(maxRate) + "_maxRate"

                    for ind_env in range(start_mc_iter,
                                         start_mc_iter + num_mc_iter):

                        for ind_name in range(len(l_placer_names)):

                            placer_name = l_placer_names[ind_name]

                            load_at = "output/received_results/exp_3079/results/" + placer_name + "/" + exp_type + "_env_" + str(
                                ind_env) + ".pck"

                            with open(load_at, "rb") as f:
                                l_uavs = pickle.load(f)

                            v_values[ind_name] = len(l_uavs)

                            m_num_uavs[ind_name, ind_users] += len(l_uavs)

                        # print(
                        #     f"{num_users} users, minRate {minRate}, maxRate {maxRate}, mc_iter {ind_env}: {v_values}"
                        # )

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            "K-means Alg. (Galkin et al.)",
            "Space rate Alg. (Hammouti et al.)",
            "Genetic Alg. (Shehzad et al.)", "GSPA (proposed)"
        ]

        plt.plot(l_num_users,
                 lower_bounds,
                 label="Lower bound",
                 linestyle='dashed',
                 color="black",
                 marker='d')

        for ind_name in range(len(l_placer_names)):
            plt.plot(l_num_users,
                     m_num_uavs[ind_name],
                     marker=l_markers[ind_name],
                     label=l_labels[ind_name])

        plt.xlabel('Number of GTs (M)')
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        plt.yticks(np.arange(5, 50, 5))
        plt.ylabel('Mean number of ABSs')
        plt.legend(loc=2)
        plt.grid()
        plt.show()

        print('end of experiment')

    # plot from saved results
    def experiment_3028_plot_02(l_args):
        """
        This experiment plot mean_num_uavs vs min user rate for all placers
        
        """

        num_mc_iter = 200
        start_mc_iter = 0
        l_num_users = np.array([70])
        l_max_uav_total_rate = np.array([100.])  # 101.5
        l_min_user_rate = np.array([10, 15, 20., 25, 30, 35, 40,
                                    45])  #  5, 7, 9, 11, 13, 15, 17

        lower_bounds = np.ceil(l_num_users * l_min_user_rate /
                               l_max_uav_total_rate)

        l_placer_names = [
            "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
            "GroupSparseUAVPlacer"
        ]

        l_markers = ["o", "^", "*", "v"]
        m_num_uavs = np.zeros((len(l_placer_names), len(l_min_user_rate)))

        for ind_users in range(len(l_num_users)):

            num_users = l_num_users[ind_users]

            for ind_min in range(len(l_min_user_rate)):

                minRate = l_min_user_rate[ind_min]

                for ind_max in range(len(l_max_uav_total_rate)):

                    maxRate = l_max_uav_total_rate[ind_max]

                    exp_type = str(int(num_users)) + "_users_" + str(
                        (minRate)) + "_minRate_" + str((maxRate)) + "_maxRate"

                    for ind_env in range(start_mc_iter,
                                         start_mc_iter + num_mc_iter):

                        for ind_name in range(len(l_placer_names)):

                            placer_name = l_placer_names[ind_name]

                            load_at = "output/received_results/exp_3078/results/" + placer_name + "/" + exp_type + "_env_" + str(
                                ind_env) + ".pck"

                            with open(load_at, "rb") as f:
                                l_uavs = pickle.load(f)

                            m_num_uavs[ind_name, ind_min] += len(l_uavs)

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            "K-means Alg. (Galkin et al.)",
            "Space rate Alg. (Hammouti et al.)",
            "Genetic Alg. (Shehzad et al.)", "GSPA (proposed)"
        ]

        plt.plot(l_min_user_rate,
                 lower_bounds,
                 label="Lower bound",
                 linestyle='dashed',
                 color="black",
                 marker='d')

        for ind_name in range(len(l_placer_names)):
            plt.plot(l_min_user_rate,
                     m_num_uavs[ind_name],
                     marker=l_markers[ind_name],
                     label=l_labels[ind_name])

        plt.xlabel('Minimum GT rate [Mbps]')
        plt.xticks(np.linspace(10, 45, 8))
        # plt.xticks([5, 7, 9, 11, 13, 15, 17])
        # plt.yticks(np.linspace(0, 20, 5))
        plt.ylabel('Mean number of ABSs')
        plt.legend(loc=2)
        plt.grid()
        plt.show()

        print('end of experiment')

    # plot from saved results
    def experiment_3028_plot_03(l_args):
        """
        This experiment plots mean number of uavs vs min_user_rate for different max_uav_total_rate in the case of using GroupSparseUAVPlacer.
        """

        num_mc_iter = 50
        num_users = 70
        l_min_user_rate = np.array([5., 7, 9, 11, 13, 15, 17])
        l_max_uav_total_rate = np.array([40., 60, 80, 100])

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
                    l_min_user_rate[ind_minRate]) + "_minRate_" + str(
                        l_max_uav_total_rate[ind_maxRate]) + "_maxRate"

                for ind_env in range(0, num_mc_iter):

                    load_at = "output/received_results/exp_3081/results/" + placer_name + "/" + exp_type + "_env_" + str(
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

        # tab:orange
        l_color_lower_bounds = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'
        ]

        for ind_maxRate in range(len(l_max_uav_total_rate)):
            plt.plot(l_min_user_rate,
                     l_lower_bounds[ind_maxRate],
                     label=l_labels_lower_bounds[ind_maxRate],
                     marker=l_markers[ind_maxRate],
                     linestyle='dashed',
                     color=l_color_lower_bounds[ind_maxRate])

        for ind_maxRate in range(len(l_max_uav_total_rate)):
            plt.plot(l_min_user_rate,
                     m_num_uavs[ind_maxRate],
                     marker=l_markers[ind_maxRate],
                     label=l_labels[ind_maxRate],
                     color=l_color_lower_bounds[ind_maxRate])

        plt.xlabel('Minimum GT rate [Mbps]')
        plt.xticks(l_min_user_rate)
        plt.yticks(np.arange(0, 50, 5))
        plt.ylabel('Mean number of ABSs')
        plt.legend(loc=2, ncol=2)
        # plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2)
        plt.grid()
        plt.show()

        print('end of experiment')

    # plot from saved results
    def experiment_3028_plot_04(l_args):
        """
        This experiment plot mean_num_uavs vs max_uav_total_rate for all placers
        
        """

        num_mc_iter = 200
        start_mc_iter = 0
        l_num_users = np.array([70])
        l_max_uav_total_rate = np.array([39., 49, 59, 69, 79, 89, 99])
        l_min_user_rate = np.array([20.])
        # l_max_uav_total_rate = np.array([33., 50, 67, 84, 101])
        # l_min_user_rate = np.array([20.])

        lower_bounds = np.ceil(l_num_users * l_min_user_rate /
                               l_max_uav_total_rate)

        l_placer_names = [
            "KMeansPlacer", "SpaceRateKMeans", "GeneticPlacer",
            "GroupSparseUAVPlacer"
        ]

        l_markers = ["o", "^", "*", "v"]
        m_num_uavs = np.zeros((len(l_placer_names), len(l_max_uav_total_rate)))

        for ind_users in range(len(l_num_users)):

            num_users = l_num_users[ind_users]

            for ind_min in range(len(l_min_user_rate)):

                minRate = l_min_user_rate[ind_min]

                for ind_max in range(len(l_max_uav_total_rate)):

                    maxRate = l_max_uav_total_rate[ind_max]

                    exp_type = str(num_users) + "_users_" + str(
                        minRate) + "_minRate_" + str(maxRate) + "_maxRate"

                    for ind_env in range(start_mc_iter,
                                         start_mc_iter + num_mc_iter):

                        for ind_name in range(len(l_placer_names)):

                            placer_name = l_placer_names[ind_name]

                            load_at = "output/received_results/exp_3080/results/" + placer_name + "/" + exp_type + "_env_" + str(
                                ind_env) + ".pck"

                            with open(load_at, "rb") as f:
                                l_uavs = pickle.load(f)

                            m_num_uavs[ind_name, ind_max] += len(l_uavs)

        m_num_uavs = m_num_uavs / num_mc_iter

        l_labels = [
            "K-means Alg. (Galkin et al.)",
            "Space rate Alg. (Hammouti et al.)",
            "Genetic Alg. (Shehzad et al.)", "GSPA (proposed)"
        ]

        plt.plot(l_max_uav_total_rate,
                 lower_bounds,
                 label="Lower bound",
                 linestyle='dashed',
                 color="black",
                 marker='d')

        for ind_name in range(len(l_placer_names)):
            plt.plot(l_max_uav_total_rate,
                     m_num_uavs[ind_name],
                     marker=l_markers[ind_name],
                     label=l_labels[ind_name])

        plt.xlabel('Total rate of each ABS [Mbps]')
        plt.xticks([39, 49, 59, 69, 79, 89, 99])
        # plt.xticks([
        #     33,
        #     50,
        #     67,
        #     84,
        #     101,
        # ])
        plt.yticks(np.linspace(10, 70, 7))
        plt.ylabel('Mean number of ABSs')
        plt.legend(loc=1)
        plt.grid()
        plt.show()

        print('end of experiment')


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