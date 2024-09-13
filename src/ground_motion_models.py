# -*- coding: utf-8 -*-
"""
@author: Raul Rincon, Rice University
"""

from scipy.stats import norm, lognorm, multivariate_normal
from scipy import integrate
from tqdm import tqdm
from scipy.interpolate import interp1d
import utm
import pandas as pd
import time
import numpy as np
import math
from matplotlib import pyplot as plt
plt.rc('font', family='Times New Roman')

def random_events_over_line(start_point, end_point, num_sim, seed):
    " Generator of random locations over a line fault   "
    X1, Y1 = start_point
    X2, Y2 = end_point

    rng = np.random.default_rng(seed=seed)

    if (X2-X1) == 0:
        Yi = rng.random(num_sim)*(Y2-Y1)
        Yrand = Yi+Y1
        Xrand = np.ones(len(Yi))*X1
    else:
        m = (Y2-Y1)/(X2-X1)
        Xi = rng.random(num_sim)*(X2-X1)

        Yrand = m*Xi+Y1
        Xrand = X1+Xi

    events_lat, events_lon = Xrand, Yrand
    return events_lat, events_lon


def dist_r_point(x_b, y_b, x_e, y_e, z_b=0, z_e=0):
    """ Computes the distance between the earthquake epicenter and the asset location"""
    R = ((x_e - x_b)**2 + (y_e - y_b)**2 + (z_e - z_b)**2)**0.5
    return R


def DistCalc(Asset1, Asset2):  
    "Computes the distance between two assets"
    X1, Y1 = Asset1
    X2, Y2 = Asset2
    R = ((X1-X2)**2+(Y1-Y2)**2)**0.5
    return R

def magnitude_find(Mmin, Mmax, b, Fmrand):
    "Find Mw using Fmrand a truncated Guttemberg-Richter model"
    C = 1-(Fmrand*(1-10.0**(-b*(Mmax-Mmin))))
    Mwrand = -np.log10(C)/b+Mmin
    return Mwrand

def ComputeCovMat(asset_loc, asset_period, case):
    """ 
    Computes the covariance matrix which uses the distances between the points
    The model used is the presented by Jayaram and Baker (2010)
    asset_loc is the vector of Lat, Long locations of the assets
    asset_period is the vector of structural period
    Case is 1 and 2 according to Jayaram and Baker(2010)
    The third dimension is used for the cov matrix for every period of interest
    """

    # First compute distance marix
    H = np.zeros((asset_loc.shape[0], asset_loc.shape[0]))
    for site_i in range(asset_loc.shape[0]):
        for site_j in range(asset_loc.shape[0]):
            H[site_i, site_j] = DistCalc(
                asset_loc[site_i], asset_loc[site_j])

    # Second compute the correlation model range R (Jayaram and Baker)
    Tunique = np.unique(asset_period)
    if case == 1:
        # short periods
        b1 = (8.5+17.2*Tunique)*(Tunique < 1)
    else:
        b1 = (40.7-15*Tunique)*(Tunique < 1)

    b2 = (22+3.7*Tunique)*(Tunique >= 1)

    RangeSemiv = b1+b2
    # Third, compute the cov matrix for each Period T of interest

    CovMatrix = np.zeros(
        (asset_period.shape[0], asset_period.shape[0], Tunique.shape[0]))

    for t in range(Tunique.shape[0]):
        CovMatrix[:, :, t] = 1-(1-np.exp(-3*H/RangeSemiv[t]))

    return CovMatrix

def compute_seismic_im(assets_lat, assets_long, event_lat,
                       event_long, Mw, type_ims='pga',
                       meshY=None, meshX=None,
                       fundamental_period=None, soil_type='bedrock', soil_variation='clusters',
                       coordWGS84=True, units='m',
                       simul_residuals_per_event=1, include_residuals=True,
                       same_residuals_for_magnitudes=False,
                       depth=10000,
                       seed=12345):
    # Ground motion model based on Atkison and Boore 1995.
    # Median and standard deviations are extrapolated. If the values are out of the limits,
    # the value of the limits is used ('saturated')
    # if im is not given, then the outcome is for pga
    # Soil types are either 'bedrock' or 'deep-soil'. Use deep-soil to represent
    # dense or stiff soils with depth >60m usually common in north east america.
    # If coord. passed are in WGS84, then assign TRUE. They will be converted to cartesian coord.

    print('Calculating location in cartesian units')
    num_assets = len(assets_lat)
    if isinstance(Mw, (list, dict)):
        iterate = len(Mw)
        Mw = np.array(Mw)
    elif isinstance(Mw, (float, int)):
        iterate = 1
        Mw = np.array([Mw])
    else:
        iterate = Mw.shape[0]

    if meshX is not None:
        assets_lat = np.concatenate((assets_lat, meshY))
        assets_long = np.concatenate((assets_long, meshX))

    if coordWGS84:
        x_b, y_b, zone_numb, zone_lett = utm.from_latlon(
            assets_lat, assets_long)
        x_e, y_e, _, _ = utm.from_latlon(
            event_lat, event_long, zone_numb, zone_lett)
    else:
        x_b, y_b = assets_lat, assets_long
        x_e, y_e = event_lat, event_long

    if coordWGS84:
        R_hyp = dist_r_point(x_b, y_b, x_e, y_e, z_e=depth) / \
            1000  # Distance in km
        x_b /= 1000
        y_b /= 1000

    elif coordWGS84 == False and units == 'm':
        R_hyp = dist_r_point(x_b, y_b, x_e, y_e, z_e=depth)/1000

    elif coordWGS84 == False and units == 'km':
        R_hyp = dist_r_point(x_b, y_b, x_e, y_e, z_e=depth)
    else:
        print('Check units or Coordinate Reference System')

    if soil_variation == 'clusters':
        soil_case = 2
    elif soil_variation == 'no clusters':
        soil_case = 1
    else:
        print("Select the proper soil variation: 'clusters' or 'no clusters'")

    cont = 0
    cov_matrix = corr_cov_matrices_gmm(
        x_b, y_b, case=soil_case, fundamental_period=fundamental_period)

    im_events_assets = []
    im_events_mesh = []

    print("Computing the simulations of intensity measures per event")

    if same_residuals_for_magnitudes:
        inter_residuals_unscaled, intra_residuals_unscaled = event_log_residuals(
            cov_matrix, simul_residuals_per_event, case=soil_case, fundamental_period=None, seed=seed)

    for mw in tqdm(range(iterate)):
        cont += 1
        log_median_im, sigma_intra, sigma_inter, fact_soil = gmm_AB95(
            Mw[mw], R_hyp, fundamental_period, type_ims, soil_type)
        im_event_i = np.zeros((simul_residuals_per_event, len(x_b)))

        if same_residuals_for_magnitudes == False:
            inter_residuals_unscaled, intra_residuals_unscaled = event_log_residuals(
                cov_matrix, simul_residuals_per_event, case=soil_case, fundamental_period=None, seed=seed+cont)

        for sim in range(simul_residuals_per_event):
            im_med = 10**(log_median_im)
            im_intra = 10**(sigma_intra*intra_residuals_unscaled[sim])
            im_inter = 10**(sigma_inter*inter_residuals_unscaled[sim])
            im_event_i[sim, :] = im_med*im_intra*im_inter*fact_soil/980.7

            if include_residuals is False:
                im_event_i[sim, :] = im_med*fact_soil/980.7

        im_events_assets.append(im_event_i[:, :num_assets])
        im_events_mesh.append(im_event_i[:, num_assets:])
    return im_events_assets, im_events_mesh


def gmm_AB95_base(Mw, R_hyp, fundamental_period=None, type_im=None, soil_type='bedrock'):
    # Ground motion model based on Atkison and Boore 1995.
    # Median and standard deviations are extrapolated. If the values are out of the limits,
    # the value of the limits is used ('saturated')
    # if im is not given, then the outcome is for PSA(T)
    # Soil types are either 'bedrock', 'firm' or 'soft'

    # if len(Mw) > 1:
    #     print('Only Me with size 1 (one event at a time) is accepted')
    #     return

    if fundamental_period is None and type_im is None:
        print(
            'An im (such as "pga" or T(s) should be given. If both are given, the im prevails.')
        return
    # Computation of ground motion in bedrock
    coefficients = np.array([2.27, 0.634, -0.017, 0,
                             2.6, 0.635, -0.0308, 0,
                             2.77, 0.62, -0.0409, 0,
                             2.95, 0.604, -0.0511, 0,
                             3.26, 0.55, -0.064, 0,
                             3.54, 0.475, -0.0717, 0.000106,
                             3.75, 0.418, -0.0644, 0.000457,
                             3.92, 0.375, -0.0562, 0.000898,
                             3.99, 0.36, -0.0527, 0.00121,
                             4.06, 0.346, -0.0492, 0.00153,
                             4.19, 0.328, -0.0477, 0.00226]).reshape(-1, 4)

    freq = np.array([0.5,
                     0.8,
                     1,
                     1.3,
                     2,
                     3.2,
                     5,
                     7.9,
                     10,
                     13,
                     20])

    coeff_pga_pgv = np.array([3.79, 0.298, -0.0536, 0.00135,
                              2.04, 0.422, -0.0373, 0]).reshape(-1, 4)

    if type_im == 'pga':
        coef = coeff_pga_pgv[0].reshape(1, -1)
    elif type_im == 'pgv':
        coef = coeff_pga_pgv[1].reshape(1, -1)
    else:
        f = 1 / fundamental_period
        interpolator = interp1d(freq, coefficients,
                                fill_value=(min(coefficients),
                                            max(coefficients)),
                                bounds_error=False,  axis=0)
        coef = interpolator(f)
        coef = coef.reshape(1, -1)

    log_median_im = coef[:, 0] + coef[:, 1] * (Mw - 6) + coef[:, 2] * (
        Mw - 6)**2 - np.log10(R_hyp) - coef[:, 3] * R_hyp

    # Computation of standard deviation of the residuals
    sigma_intra = 0.20
    freq_sigma_inter = [1, 2, 5, 10]
    sigma_inter_vals = [0.13, 0.14, 0.17, 0.18]
    if type_im == 'pga' or type_im == 'pgv':
        T = 0.01
    f = 1/T
    interpolator = interp1d(freq_sigma_inter, sigma_inter_vals,
                            fill_value=(min(sigma_inter_vals),
                                        max(sigma_inter_vals)),
                            bounds_error=False,  axis=0)
    sigma_inter = interpolator(f)

    # Definition of soil amplification factor
    soil_ampl_factors = np.array([1.9, 1.9, 2, 1.7, 1.4, 0.93])
    soil_ampl_freq = np.array([0.5, 1, 2, 5, 10, 20])
    if soil_type == 'bedrock':
        fact_soil = 1
    elif soil_type == 'deep-soil':
        if type_im == 'pga' or type_im == 'pgv':
            fact_soil = 0.93  # Check the paper... 1.4-2.0 is for structures with Hz 10 to 2, or a factor of 2 is recommended. 0.93 for structures in 20 Hz
        else:
            interpolator = interp1d(soil_ampl_freq, soil_ampl_factors,
                                    fill_value=(min(soil_ampl_factors), max(
                                        soil_ampl_factors)),
                                    bounds_error=False,  axis=0)
            fact_soil = interpolator(f)
    else:
        print('Select either "bedrock" or "deep-soil" for soil_type')
    # im_median_bedrock = 10**((log_median_im))*fact_soil/ 980.7   This is the way to transform log_im back to accelerations (g).
    return log_median_im, sigma_intra, sigma_inter, fact_soil


def gmm_AB95(Mw, R_hyp, fundamental_period=None, type_im='pga', soil_type='bedrock'):
    """
    # Ground motion model based on Atkison and Boore 1995.
    # Median and standard deviations are extrapolated. If the values are out of the limits,
    # the value of the limits is used ('saturated')
    # if im is not given, then the outcome is for pga
    # Soil types are either 'bedrock', 'firm' or 'soft'
    """

    if type_im != 'sa_ave':
        log_median_im, sigma_intra, sigma_inter, fact_soil = gmm_AB95_base(
            Mw, R_hyp, fundamental_period, type_im, soil_type)
    else:
        # Save computed for T =[1/50,3) usint 20 periods, evenly spaced.
        f = np.linspace(1/3, 20, 50)
        Ts = 1/f
        log_median_im = np.zeros(R_hyp.shape[0])
        # Check equations inside this for. Not ready to be used.
        for i in range(R_hyp.shape[0]):
            sa_ave_i, sigma_intra, sigma_inter, fact_soil = gmm_AB95_base(
                Mw, R_hyp[i], Ts, type_im)
            log_median_im[i] = np.exp(np.log(sa_ave_i).mean())

    return log_median_im, sigma_intra, sigma_inter, fact_soil


def corr_cov_matrices_gmm(x_data, y_data, case=2, fundamental_period=None):
    """
    Computes the covariance matrix which uses the distances between the points
    The model used is the presented by Jayaram and Baker (2010)
    fundamental_period T is the vector of structural periods, in seconds
    Case 1: If the Vs30 values do not show or are not expected to show clustering 
    (i.e. the geologic condition of the soil varies widely over the region.
    Case 2: If the Vs30 values show or are expected to show clustering 
    (i.e. there are clusters of sites in which the geologic conditions of the soil are similar).

    Output: CovMatrix with (dim0, dim1) representinv a cov matrix for a certain unique period T (in dim2).
    """

    # First compute distance marix
    H = np.zeros((x_data.shape[0], x_data.shape[0]))
    vector_locations = [(x_data[i], y_data[i]) for i in range(len(x_data))]

    print('Calculating distances between points')
    for site_i in range(len(x_data)):
        for site_j in range(len(x_data)):
            H[site_i, site_j] = DistCalc(
                vector_locations[site_i], vector_locations[site_j])

    # Second compute the correlation model range R (Jayaram and Baker)
    if fundamental_period is not None:
        # covariances only computed for unique values of T.
        Tunique = np.unique(fundamental_period)
        if len(Tunique) > 15:
            print(
                "Caution, specific periods must not be used. Use bins of representative fundamental periods")
            return
    else:
        Tunique = np.array([0])
    if case == 1:
        # short periods
        b1 = (8.5+17.2*Tunique)*(Tunique < 1)
    else:
        b1 = (40.7-15*Tunique)*(Tunique < 1)

    b2 = (22+3.7*Tunique)*(Tunique >= 1)

    RangeSemiv = b1+b2

    print('Calculating the covariance matrix')
    # Third, compute the cov matrix for each Period T of interest
    cov_matrix = np.zeros((x_data.shape[0], x_data.shape[0], len(Tunique)))

    for t in range(len(Tunique)):
        cov_matrix[:, :, t] = 1-(1-np.exp(-3*H/RangeSemiv[t]))

    return cov_matrix


def event_log_residuals(cov_matrix, num_simulations, case=2, fundamental_period=None, seed=False):
    """
    Output: Two lists are returned.
            List 1 corresponds to a total of num_simulations scenarios of the inter event uniform residuals (to be multiplied by desv std).
            List 2 is a list with num_simulations scenarios. Each scenario consists on (dim0, dim1) representing
                the correlated samples of log residuals, for each period on dim2.
    """
    if seed:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
        seed = 1234

    num_points = cov_matrix.shape[0]
    # Random values for residuals of the inter event
    inter_residuals_unscaled = rng.normal(0, 1, num_simulations)

    # Residuals of the inter event
    if fundamental_period is not None:
        # covariances only computed for unique values of T.
        Tunique = np.unique(fundamental_period)
        if len(Tunique) > 15:
            print(
                "Caution, specific periods must not be used. Use bins of representative fundamental periods")
            return
    else:
        Tunique = np.array([0])

    mean_intra = np.zeros(num_points)
    intra_residuals_unscaled_period = np.zeros(
        (num_simulations, num_points, len(Tunique)))

    for i in range(len(Tunique)):
        mult_random = rng.multivariate_normal(mean_intra, cov_matrix[:, :, i],
                                              size=num_simulations)
        intra_residuals_unscaled_period[:, :, i] = mult_random

    # Clean the Correlation Matrix for the Periods of interest, if fundamental_period is not None
    if fundamental_period is not None:
        intra_residuals_unscaled = np.zeros((num_simulations, num_points))

        for i in range(num_points):
            col = np.where(Tunique == fundamental_period[i])[0][0]
            intra_residuals_unscaled[:,
                                     i] = intra_residuals_unscaled_period[:, i, col]
    else:
        intra_residuals_unscaled = intra_residuals_unscaled_period[:, :, 0]

    return inter_residuals_unscaled, intra_residuals_unscaled


def compute_hazardcurve_pointsource_singlelocation(lambda_0, Mmin, Mmax, beta,
                                                   event_lat, event_long,
                                                   assets_lat, assets_long,
                                                   gmm='AB_95', type_ims='pga',
                                                   fundamental_period=None,
                                                   soil_type='bedrock',
                                                   coordWGS84=True, units='m',
                                                   depth=10000, im_vector=None
                                                   ):

    if coordWGS84:
        x_b, y_b, zone_numb, zone_lett = utm.from_latlon(
            assets_lat, assets_long)
        x_e, y_e, _, _ = utm.from_latlon(
            event_lat, event_long, zone_numb, zone_lett)
    else:
        x_b, y_b = assets_lat, assets_long
        x_e, y_e = event_lat, event_long

    if coordWGS84:
        R_hyp = dist_r_point(x_b, y_b, x_e, y_e, z_e=depth) / \
            1000  # Distance in km
        x_b /= 1000
        y_b /= 1000

    elif coordWGS84 == False and units == 'm':
        R_hyp = dist_r_point(x_b, y_b, x_e, y_e, z_e=depth)/1000

    elif coordWGS84 == False and units == 'km':
        R_hyp = dist_r_point(x_b, y_b, x_e, y_e, z_e=depth)
    else:
        print('Check units or Coordinate Reference System')

    # Number of points used to integrate
    m_vector = np.linspace(Mmin, Mmax, 100, endpoint=True).reshape(-1,)

    # Vector of im to construct H(im)
    if im_vector is None:
        im_vector = np.arange(0.001, 5, 0.001)

    # For a distance R and M in m_vector, compute mean lnIM and sigma_lnIM
    ln_medians_im = np.zeros(len(m_vector))
    sigma_lnim = np.zeros(len(m_vector))
    for mw in tqdm(range(m_vector.shape[0])):
        log_median_im, sigma_intra, sigma_inter, fact_soil = gmm_AB95(
            m_vector[mw], R_hyp, fundamental_period, type_ims, soil_type)
        ln_medians_im[mw] = np.log((10**log_median_im*fact_soil)/980.7)
        sigma_lnim[mw] = np.log(10**(sigma_intra**2 + sigma_inter**2)**0.5)

    # Unconditional probabiity of exceedance P(IM>x)
    prob_exceed_im = np.zeros(len(im_vector))
    f_m = ((beta*np.log(10)*10**(-beta*(m_vector - Mmin))) /
           (1 - 10**(-beta*(Mmax - Mmin)))).reshape(-1,)
    for j in range(im_vector.shape[0]):
        prob_exceed_condit_m_r = 1 - \
            norm(ln_medians_im, sigma_lnim).cdf(np.log(im_vector[j]))
        prob_exceed_im[j] = integrate.simpson(
            prob_exceed_condit_m_r*f_m, m_vector)

    exceed_rate_im = prob_exceed_im*lambda_0

    return exceed_rate_im