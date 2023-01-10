from calendar import c
import sys
import os
from tabnanny import check
from IPython.core.debugger import set_trace
from collections import OrderedDict

from scipy.optimize import linprog

from placement.placers import CapacityBasedPlacer

f_find_rate_allocation_real_uavs = CapacityBasedPlacer.find_rate_allocation_real_uavs

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time
from datetime import datetime

from gsim import rng
from gsim.utils import xor

import gc


def gen_feasible_pts(environment, channel, num_users, min_user_rate=None):
    """Returns a list of random points on the street such that there exists an
    arrangement of UAVs guaranteed for which all users rx at least the minimum
    rate `min_user_rate`. 
    """
    ind = 0
    while True:
        l_users = environment.random_pts_on_street(num_users)

        if (min_user_rate is None) or channel.feasible_placement_exists(
                grid=environment.fly_grid,
                user_coords=l_users,
                min_user_rate=min_user_rate):
            return l_users


def place_and_plot(environment,
                   channel,
                   min_user_rate,
                   l_placers,
                   num_users,
                   disable_flying_gridpts_by_dominated_verticals=True,
                   no_axes=False):
    # Generate feasible user positions
    l_users = gen_feasible_pts(environment=environment,
                               channel=channel,
                               min_user_rate=min_user_rate,
                               num_users=num_users)

    # Placement
    for placer in l_placers:
        name = placer.__class__.__name__
        print(f"----- {name} -------")
        uav_coords = placer.place(fly_grid=environment.fly_grid,
                                  channel=channel,
                                  user_coords=l_users)
        environment.dl_uavs[name] = uav_coords
        print(
            channel.assess_placement(grid=environment.fly_grid,
                                     uav_coords=uav_coords,
                                     user_coords=l_users))
        #print(f"Num UAVs {name} = ", len(env.dl_uavs[name]))

    # Show user locations and enabled flying grid points
    environment.l_users = l_users
    if disable_flying_gridpts_by_dominated_verticals:
        environment.disable_flying_gridpts_by_dominated_verticals(channel)

    # Display
    fgcolor = (1., 1., 1.) if no_axes else (0., 0., 0.)
    environment.plot(fgcolor=fgcolor)
    environment.show()


def check_valid_placement(channel, environment, min_user_rate,
                          max_uav_total_rate, l_uavs):
    """
    Returns true iff there exists a rate allocation between the UAVs at 
    positions given by l_uavs and the users so that: 
        + The sum of the rate that each user receives from all UAVs is at 
            least min_user_rate.
        + The sum of the rates that each UAV provides to all users is not
            greater than max_uav_total_rate.
        + The rate that each user receives from each uav is not greater than
            the capacity of the channel between them.
    """
    map = channel.capacity_map(
        grid=environment.fly_grid,
        user_coords=environment.l_users,
    )
    m_capacity = map.list_vals().T

    v_ind_uav_pos = map.grid.nearest_inds(l_uavs)

    m_capacity_submat = m_capacity[:, v_ind_uav_pos]
    """ 
    TODO: replace the following code with an invocation to
        CapacityBasedPlacer.find_rate_allocation
    
    Done
    """

    return f_find_rate_allocation_real_uavs(m_capacity_submat, min_user_rate,
                                            max_uav_total_rate)


def f_save_and_load(location,
                    exp_type,
                    ind_mc,
                    mode=None,
                    data_to_save=None,
                    name_placer=None):

    env_at = location + exp_type + "_env_" + str(ind_mc) + ".pck"

    if mode == "load_env":

        with open(env_at, "rb") as f:
            loaded_env = pickle.load(f)

        environment = loaded_env[0]
        channel = loaded_env[1]
        min_user_rate = loaded_env[2]
        max_uav_total_rate = loaded_env[3]

        return environment, channel, min_user_rate, max_uav_total_rate

    if mode == "save_env":

        with open(env_at, "wb") as f:
            pickle.dump(data_to_save, f)

    if mode == "save_results":

        folder_name = location + name_placer + "/"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        results_at = folder_name + exp_type + "_env_" + str(ind_mc) + ".pck"

        with open(results_at, "wb") as f:
            pickle.dump(data_to_save, f)


def mean_num_uavs(environment,
                  channel,
                  min_user_rate,
                  l_placers,
                  num_users=None,
                  num_mc_iter=1,
                  max_uav_total_rate=None,
                  start_mc_iter=0,
                  b_save_env=False,
                  b_load_env=False,
                  b_save_results=False,
                  save_exp_at=None,
                  exp_type=None):
    """Returns a dict whose keys are the names of the placers in
    `l_placers` and the values are the MC estimates of the mean number
    of UAVs to attain `min_user_rate`."""

    if b_save_env or b_load_env:
        envs_at = save_exp_at + "envs/"
        if not os.path.exists(envs_at):
            os.mkdir(envs_at)

    if b_save_results:
        results_at = save_exp_at + "results/"
        if not os.path.exists(results_at):
            os.mkdir(results_at)

    d_num_uavs = {placer.name: [] for placer in l_placers}

    for ind_mc in range(start_mc_iter, start_mc_iter + num_mc_iter):

        print(f"MC iteration: {ind_mc}")

        if not b_load_env:

            environment.l_users = gen_feasible_pts(environment=environment,
                                                   channel=channel,
                                                   min_user_rate=min_user_rate,
                                                   num_users=num_users)
            if b_save_env:

                f_save_and_load(envs_at,
                                exp_type,
                                ind_mc,
                                mode="save_env",
                                data_to_save=[
                                    environment, channel, min_user_rate,
                                    max_uav_total_rate
                                ])
        else:
            environment, channel, min_user_rate, max_uav_total_rate = f_save_and_load(
                envs_at, exp_type, ind_mc, mode="load_env")

        for placer in l_placers:

            l_uavs = placer.place(fly_grid=environment.fly_grid,
                                  channel=channel,
                                  user_coords=environment.l_users)

            if l_uavs is not None:  # currently not penalizing cases that could not be solved

                if b_save_results:

                    f_save_and_load(results_at,
                                    exp_type,
                                    ind_mc,
                                    mode="save_results",
                                    data_to_save=l_uavs,
                                    name_placer=placer.name)

                d_num_uavs[placer.name].append(len(l_uavs))

        gc.collect()

    return {key: np.mean(val) for key, val in d_num_uavs.items()}


# Generic Monte Carlo algorithm
def user_loc_mc(environment,
                channel,
                l_placers,
                num_users=None,
                min_user_rate=None,
                num_mc_iter=1,
                exclude_keys=["user_rates_mult", "user_rates_sing"]):
    """Returns a dict whose keys are the names of the placers in `l_placers` and
    the values are dicts obtained by averaging the fields of the dicts returned
    by channel.assess_placement across realizations of the user locations. 

    If `min_user_rate` is not None, then the user locations are guaranteed to
    allow a feasible placement where all users receive at least that rate.

    """
    def avg_dicts(ld_ass):
        """Takes a list of dicts as input, returns a dict with the averages of
        each key in `ld_ass`."""

        if len(ld_ass) == 0:
            return None
        d_out = dict()
        for key in ld_ass[0].keys():
            if key in exclude_keys:
                continue
            vals = [dic[key] for dic in ld_ass]
            d_out[key] = np.mean([val for val in vals if val is not None])

        return d_out

    dld_ass = {placer.name: [] for placer in l_placers}
    for _ in range(num_mc_iter):
        user_coords = gen_feasible_pts(environment=environment,
                                       channel=channel,
                                       min_user_rate=min_user_rate,
                                       num_users=num_users)
        for placer in l_placers:
            uav_coords = placer.place(fly_grid=environment.fly_grid,
                                      channel=channel,
                                      user_coords=user_coords)
            # debug
            # if uav_coords.ndim > 2:
            #     uav_coords = placer.place(user_coords=user_coords)
            #if uav_coords is not None:  # currently not penalizing cases that could not be solved
            d_assessment = channel.assess_placement(
                grid=environment.fly_grid,
                uav_coords=uav_coords,
                user_coords=user_coords,
                #min_user_rate=min_user_rate
            )
            dld_ass[placer.name].append(d_assessment)

    # Average metrics
    dd_out = dict()
    for placer in l_placers:
        dd_out[placer.name_on_figs] = avg_dicts(dld_ass[placer.name])

    return dd_out


def metrics_vs_num_users(v_num_users=[], **kwargs):
    """Returns a dict of dicts of lists. The keys of the outer dict are the
    names of the placers. The keys of the inner dict the names of the metrics.
    The lists have the same length as `v_num_users`. """

    assert "num_users" not in kwargs.keys()

    l_metrics = [
        user_loc_mc(num_users=num_users, **kwargs) for num_users in v_num_users
    ]

    if len(l_metrics) == 0:
        return None

    return ldd_to_ddl(l_metrics)
    # {
    #     placer_name: {
    #         metric_name:
    #         [mval[placer_name][metric_name] for mval in l_metrics]
    #         for metric_name in l_metrics[0][placer_name]
    #     }
    #     for placer_name in l_metrics[0].keys()
    # }


def metrics_vs_environments_and_channels(environments, channels, **kwargs):
    """ 
    
    Args:
    
    `environments` and `channels` are lists of the same length. 

    Returns:

    dict of dicts of lists. The keys of the outer dict are the
    names of the placers. The keys of the inner dict the names of the metrics.
    The lists have the same length as `environments`. 
    
    
    """

    l_metrics = [
        user_loc_mc(environment=env, channel=channel, **kwargs)
        for env, channel in zip(environments, channels)
    ]

    if len(l_metrics) == 0:
        return None

    return ldd_to_ddl(l_metrics)


def metrics_vs_placers(ll_placers=[], **kwargs):
    """
    Currently unused. 

    Args:

    `ll_placers` is a list of list of Placers. The Placers in each inner list
    have all the same name. 

    Returns:

    dict of dicts of lists. The keys of the outer dict are the names of the
    placers. The keys of the inner dict the names of the metrics. The lists have
    the same length as `ll_placers`. """

    l_metrics = [
        user_loc_mc(l_placers=t_placers, **kwargs)
        for t_placers in zip(*ll_placers)
    ]

    if len(l_metrics) == 0:
        return None

    return ldd_to_ddl(l_metrics)


def metrics_vs_min_user_rate(l_placers=[], l_min_user_rates=[], **kwargs):
    """
    Args:

    `l_min_user_rates` is a list or iterable.

    Returns:

    dict of dicts of lists. The keys of the outer dict are the names of the
    placers. The keys of the inner dict the names of the metrics. The lists have
    the same length as `l_min_user_rates`. """

    l_metrics = []
    for min_user_rate in l_min_user_rates:
        for placer in l_placers:
            placer.min_user_rate = min_user_rate
        l_metrics.append(
            user_loc_mc(l_placers=l_placers,
                        min_user_rate=min_user_rate,
                        **kwargs))

    if len(l_metrics) == 0:
        return None

    return ldd_to_ddl(l_metrics)


def ldd_to_ddl(ldd):
    """Args:
        list of dicts of dicts
        
        Returns:
        
        dict of dicts of lists
        """

    return {
        placer_name: {
            metric_name: [mval[placer_name][metric_name] for mval in ldd]
            for metric_name in ldd[0][placer_name]
        }
        for placer_name in ldd[0].keys()
    }