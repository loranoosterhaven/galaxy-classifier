"""Functions to generate statistics and labels from object maps"""

import numpy as np
import warnings
import photutils
import math
from skimage.color import label2rgb
from mtolib import moment_invariants
from astropy.modeling import models, fitting, custom_model
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import Fittable1DModel, Parameter
from scipy.special import binom
import matplotlib.pyplot as plt

def colour_labels(label_map):
    """Apply a colour to each object in the image non-sequentially and convert to 8-bit int RGB format"""
    return np.uint8(label2rgb(label_map)*255)


def relabel_segments(label_map, shuffle_labels=False):
    """Relabel segments with sequential numbers"""

    original_shape = label_map.shape

    label_map = label_map.ravel()
    output = np.zeros(label_map.shape, dtype=label_map.dtype)

    # Sort the object ID map for faster pixel retrieval
    sorted_ids = label_map.argsort()
    id_set = list(set(label_map))
    id_set.sort()

    id_set.remove(-1)

    # Get the locations in sorted_ids of the matching pixels
    right_indices = np.searchsorted(label_map, id_set, side='right', sorter=sorted_ids)
    left_indices = np.searchsorted(label_map, id_set, side='left', sorter=sorted_ids)

    # Generate a list of labels
    label_list = list(range(0, 1 + len(id_set)))

    # Shuffle order in which labels are allocated
    if shuffle_labels:
        np.random.shuffle(label_list)

    # Relabel pixels
    for n in range(len(id_set)):
        pixel_indices = np.unravel_index(sorted_ids[left_indices[n]:right_indices[n]], label_map.shape)

        output[pixel_indices] = label_list[n]

    return output.reshape(original_shape)


def levelled_segments(img, label_map):
    """Replace object ids with the value at which the object was detected"""
    output = np.zeros(img.shape)

    label_map = label_map.ravel()

    # Sort the object ID map for faster pixel retrieval
    sorted_ids = label_map.argsort()
    id_set = list(set(label_map))

    id_set.remove(-1)

    # Get the locations in sorted_ids of the matching pixels
    right_indices = np.searchsorted(label_map, id_set, side='right', sorter=sorted_ids)
    left_indices = np.searchsorted(label_map, id_set, side='left', sorter=sorted_ids)

    for n in range(len(id_set)):
        pixel_indices = np.unravel_index(sorted_ids[left_indices[n]:right_indices[n]], img.shape)

        pixel_values = img[pixel_indices]

        min_value = np.min(pixel_values)

        output[pixel_indices] = min_value

    return output


def get_image_parameters(img, object_ids, sig_ancs, nodes, node_attribs, img_coords, params,):
    """Calculate the parameters for all objects in an image"""

    # Treat warnings as exceptions
    warnings.filterwarnings('error', category=RuntimeWarning, append=True)

    parameters = []
    headings = ['ID', 'X', 'Y', 'A', 'B', 'theta',  # 'kurtosis',
                           'total_flux', 'mu_max', 'mu_median', 'mu_mean', 'R_fwhm', 'R_e', 'R10', 'R90']

    parameters.append(headings)

    # Sort the object ID map for faster pixel retrieval
    sorted_ids = object_ids.argsort()
    id_set = list(set(object_ids))

    if -1 in id_set:
        id_set.remove(-1)

    # Get the locations in sorted_ids of the matching pixels
    right_indices = np.searchsorted(object_ids, id_set, side='right', sorter=sorted_ids)
    left_indices = np.searchsorted(object_ids, id_set, side='left', sorter=sorted_ids)

    # For each object in the list, get the pixels, calculate parameters, and write to file
    for n in range(len(id_set)):

        pixel_indices = np.unravel_index(sorted_ids[left_indices[n]:right_indices[n]], img.shape)
        parameters.append(get_object_parameters(img, id_set[n], pixel_indices, sig_ancs, nodes, node_attribs, img_coords))

    # Return to printing warnings
    warnings.resetwarnings()

    return parameters

"""Calculate an object's data matrix given the indices of its pixels"""
def node_to_data_matrix(img, pixel_indices):
    minX = np.min(pixel_indices[1])
    minY = np.min(pixel_indices[0])
    maxX = np.max(pixel_indices[1])
    maxY = np.max(pixel_indices[0])

    #print(f'x: [{minX}, {maxX}], y: [{minY}, {maxY}]')
    s = (maxX - minX + 1, maxY - minY + 1)
    z = np.zeros(s)
    for i in range(0, pixel_indices[1].shape[0]):
        ix = pixel_indices[1][i]
        iy = pixel_indices[0][i]
        z[ix-minX][iy-minY] = img.data[iy, ix]
    return np.asmatrix(z)


def get_object_node_invs(img, pixel_indices):
    pixel_values = np.nan_to_num(img.data[pixel_indices])

    # Subtract min values if required
    pixel_values -= max(np.min(pixel_values), 0)

    pixel_value_set = list(set(pixel_values))
    pixel_value_set.sort(reverse=True)

    max_value = pixel_value_set[0]
    min_value = pixel_value_set[-1]

    #print(min_value, max_value)

    data_matrix = node_to_data_matrix(img, pixel_indices)
    
    # calculate the moment invariants on 4 different intensity levels
    with np.errstate(divide='ignore',invalid='ignore'):
        invts = moment_invariants.calculateInvariants(3, data_matrix)
        #print( invts)
        data_matrix[data_matrix < max_value * 0.65] = 0
        invts = moment_invariants.calculateInvariants(3, data_matrix)
        #print( invts)
        data_matrix[data_matrix < max_value * 0.8] = 0
        invts = moment_invariants.calculateInvariants(3, data_matrix)
        #print( invts)
        data_matrix[data_matrix < max_value * 0.95] = 0
        invts = moment_invariants.calculateInvariants(3, data_matrix)


def get_pixel_coordinates(p, img_coords):
    delta = p - img_coords.ref_pixel
    coords = img_coords.ref_coords + np.dot(img_coords.ra_dec_per_pixel_matrix, delta)
    return coords 

def complex_moment(p, q, raw_moments):
    c = 0 + 0j
    for k in range(0, p+1):
        for l in range(0, q+1):
            c += binom(p,k)*binom(q,l)*((-1)**(q-l))*(1j**(p+q-k-l))*raw_moments[k+l, p+q-k-l]
        
    return c

def central_moment(p, q, raw_moments):
    c = 0

    xbar = raw_moments[1,0] / raw_moments[0, 0]
    ybar = raw_moments[0,1] / raw_moments[0, 0]

    for m in range(0, p+1):
        for n in range(0, q+1):
            c += binom(p,m)*binom(q,n)*((-xbar)**(p-m))*(-ybar**(q-n))*raw_moments[m, n]
        
    return c

def log_transform(x):
    return np.sign(x) * np.log10(abs(x) + 1)    

def flusser_invariants(raw_moments, area):
    ''' calculate flusser's moment invariants. see:
    http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/MOMINV/ 
    '''
    a2 = area*area
    a25 = a2*np.sqrt(area)

    # scale invariant moments
    s11 = complex_moment(1, 1, raw_moments) / a2
    s20 = complex_moment(2, 0, raw_moments) / a2
    s21 = complex_moment(2, 1, raw_moments) / a25
    s12 = complex_moment(1, 2, raw_moments) / a25
    s30 = complex_moment(3, 0, raw_moments) / a25


    #combined and rescaled (so the values are in a similar range) to get 6 rotation invariants
    I1 = np.real(s11)

    i2 = s21*s12
    I2 = np.real(1000.0*s21*s12)

    i34 = s20 * i2
    I3 = np.real(i34)
    I4 = np.imag(i34)

    i56 = s30 * s12 * i2
    I5 = np.real(i56)
    I6 = np.imag(i56)

    invs = np.array([I1, I2, I3, I4, I5, I6])

    return invs

def affine_moment_invariants(M, area):
    u00 = central_moment(0, 0, M)
    u11 = central_moment(1, 1, M)
    u12 = central_moment(1, 2, M)
    u02 = central_moment(0, 2, M)
    u20 = central_moment(2, 0, M)
    u21 = central_moment(2, 1, M)

    u03 = central_moment(0, 3, M)
    u30 = central_moment(3, 0, M)

    I1 = (u20*u02 - u11*u11) / np.power(u00, 4.0)
    I2 = (-u30*u30*u03*u03 - 6.0*u30*u21*u12*u03 -
            - 4.0*u30*u12*u12*u12 - 4.0*u21*u21*u03
            + 3*u21*u21*u12*u12) / np.power(u00, 10.0)
    I3 = (u20*u02 - u11*u11) / np.power(u00, 7.0)

    return np.array([I1, I2, I3])

def normalized_AMIs(M, area):
    c11 = complex_moment(1,1,M)
    c21c12 = complex_moment(2,1,M) * complex_moment(1,2,M)
    c30 = complex_moment(3,0,M)
    c03 = complex_moment(0,3,M)
    c12 = complex_moment(1,2,M)

    a = np.array([c11, c21c12, c30*c03, np.real(c30*c12*c12*c12)])
    
    return np.abs(a)


class SersicCurve1D(Fittable1DModel):
    logI0 = Parameter()
    k = Parameter()
    n = Parameter()

    @staticmethod
    def evaluate(x, logI0, k, n):
        return logI0 - k*x**(1.0/n)

    @staticmethod
    def fit_deriv(x, logI0, k, n):
        d_logI0 = np.ones_like(x)
        d_k = -x**(1.0/n)
        d_n = -(k * x**(1.0/n) * np.log(x)) / (n*n)
        return [d_logI0, d_k, d_n]


def fit_sersic_model_to_invariant(x, y):
    guess_logI0 = y[0] + 0.5
    guess_n = 4.0
    guess_k = 0

    #print(y)

    print(len(y))

    # Fit model to data
    m_init = SersicCurve1D(logI0 = guess_logI0, k = guess_k, n = guess_n)
    print(m_init)
    fit = fitting.LevMarLSQFitter()
    m = fit(m_init, x, y, acc=1e12)
    print(f'{m}')
    return m

def get_object_parameters(img, node_id, pixel_indices, sig_ancs, nodes, node_attribs, img_coords):
    """Calculate an object's parameters given the indices of its pixels"""
    p = [node_id]

    parents = nodes['parent']
    areas = nodes['area']

    # Get pixel values for an object
    pixel_values = np.nan_to_num(img.data[pixel_indices])
    pixel_nodes = nodes[pixel_indices]

    # get central peak coordinates
    peak_index = np.argmax(pixel_values)
    a0, a1 = pixel_indices[0][peak_index], pixel_indices[1][peak_index]
    current_node_idx, area = nodes[a0, a1]

    # calculate ra,dec coordinates of central peak
    peak_coords = get_pixel_coordinates(np.array([a0, a1]), img_coords)
    print(f'******************************')
    print(f'object {node_id}: ra = {peak_coords[0]}, dec = {peak_coords[1]}')

    # areas and parents for this object
    pixel_parents = parents[pixel_indices]
    pixel_areas = areas[pixel_indices]
    pixel_radii = np.sqrt(pixel_areas)
    raw_moments = node_attribs[pixel_indices]['moments']

    ind = np.unravel_index(np.argsort(pixel_radii, axis=None), pixel_radii.shape)[0]
    ind = ind[0:1000]

    moment_invs = np.empty((len(ind), 4))

    for i in range(0, len(moment_invs)):
        pixel_idx = ind[i]
        rm = np.reshape(raw_moments[pixel_idx], (4,4))
        moment_invs[i] = normalized_AMIs(rm, float(pixel_areas[pixel_idx]))

    # create plots
    R = pixel_radii[ind]
    plt.subplot(6,1,1)
    plt.plot(R, moment_invs[:, 0], '-r')
    plt.subplot(6,1,2)
    plt.plot(R, moment_invs[:, 1], '-g')
    plt.subplot(6,1,3)
    plt.plot(R, moment_invs[:, 2], '-b')
    plt.subplot(6,1,4)
    plt.plot(R, moment_invs[:, 3], '-c')
    # plt.subplot(6,1,5)
    # plt.plot(R, moment_invs[:, 4], '-m')
    # plt.subplot(6,1,6)
    # plt.plot(R, moment_invs[:, 5], '-y')
    plt.savefig(f'plots/{node_id}')
    plt.clf()

    # log transformed plots
    moment_invs = log_transform(moment_invs)
    R = np.log(R)
    #sm1 = fit_sersic_model_to_invariant(R, moment_invs[:, 0])
    plt.subplot(6,1,1)
    #plt.plot(R, sm1(R))
    plt.plot(R, moment_invs[:, 0], '-r')
    plt.subplot(6,1,2)
    plt.plot(R, moment_invs[:, 1], '-g')
    plt.subplot(6,1,3)
    plt.plot(R, moment_invs[:, 2], '-b')
    plt.subplot(6,1,4)
    plt.plot(R, moment_invs[:, 3], '-c')
    # plt.subplot(6,1,5)
    # plt.plot(R, moment_invs[:, 4], '-m')
    # plt.subplot(6,1,6)
    # plt.plot(R, moment_invs[:, 5], '-y')
    plt.savefig(f'plots/{node_id}_log_transformed')
    plt.clf()


    # indices_ravelled = np.ravel_multi_index(pixel_indices, nodes.shape)

    # step_count = 1
    # inside_object = True
    # spans = 1

    # node_power = node_attribs[a0, a1]['power']
    # node_intensity = pixel_values[peak_index]
    # print(f'in {step_count}. i={pixel_values[peak_index]} area={area}. power = {node_power}')

    # traversed_nodes_pixels = []


    # while current_node_idx >= 0 and inside_object:
    #     prev_power = node_power
    #     prev_intensity = node_intensity
    #     next_i, next_j = np.unravel_index(current_node_idx, nodes.shape)
    #     next_node_idx, area = nodes[next_i, next_j]

    #     if current_node_idx in indices_ravelled:
    #         step_count += 1
    #         pixels_in_node = pixel_parents[pixel_parents == next_node_idx]
    #         areas_in_node = pixel_areas[pixel_parents == next_node_idx]

    #         #all_pixels_in_node = parents[parents == next_node_idx]
    #         #all_areas_in_node = areas[parents == next_node_idx]

    #         detection_level = node_attribs[next_i, next_j]['detection_level']

    #         moments = np.reshape(node_attribs[next_i, next_j]['moments'], (4,4))
    #         invs = flusser_invariants(moments, float(area))
    #         #print(log_transform(invs))

    #         node_intensity = img.data[next_i, next_j]
    #         delta_intensity = (prev_intensity - node_intensity) / area

    #         traversed_nodes_pixels.extend(pixels_in_node.tolist())
    #         s = pixels_in_node.size
    #         print(f'in {step_count}. dl = {detection_level}. area={area}, \
    #         {areas_in_node}')
    #         spans += s
    #     else:
    #         inside_object = False
        
    #     current_node_idx = next_node_idx

    # print(f'traversed {step_count} nodes. spans = {spans}. have {pixel_values.size} pixels')
    # numDelta = pixel_values[pixel_values == np.max(pixel_values)].size
    # print(f'delta = {numDelta}')
    # print(f'range={np.max(pixel_values) - np.min(pixel_values)}')

    #get_object_node_invs(img, pixel_indices)

    # Subtract min values if required
    pixel_values -= max(np.min(pixel_values), 0)

    flux_sum = np.sum(pixel_values)

    # Handle situations where flux_sum is 0 because of minimum subtraction
    if flux_sum == 0:
        almost_zero = np.nextafter(flux_sum,1)
        flux_sum = almost_zero
        pixel_values[pixel_values == 0] = almost_zero

    # Get first-order moments
    f_o_m = get_first_order_moments(pixel_indices, pixel_values, flux_sum)
    p.extend(f_o_m)

    # Get second-order moments
    second_order_moments = [*get_second_order_moments(pixel_indices, pixel_values, flux_sum, *f_o_m)]
    p.extend(second_order_moments)

    p.append(flux_sum)
    p.extend(get_basic_stats(pixel_values))

    radii, half_max = get_light_distribution(pixel_values, flux_sum)
    p.append(half_max)
    p.extend(radii)

    return p


def get_basic_stats(pixel_values):
    """Return basic statistics about a pixel distribution"""
    return np.max(pixel_values), np.median(pixel_values), np.mean(pixel_values)


def get_first_order_moments(indices, values, flux_sum):
    """Find the weighted centre of the object"""

    x = np.dot(indices[1], values)/flux_sum
    y = np.dot(indices[0], values)/flux_sum

    return x, y


def get_second_order_moments(indices, values, flux_sum, x, y):
    """Find the second order moments of the object"""
    x_indices = indices[1]
    y_indices = indices[0]

    x_pows = np.power(x_indices,2)
    y_pows = np.power(y_indices,2)

    # Find the second order moments
    x2 = (np.sum(np.dot(x_pows, values))/flux_sum) - np.power(x, 2)
    y2 = (np.sum(np.dot(y_pows, values))/flux_sum) - np.power(y, 2)
    xy = (np.sum(x_indices * values * y_indices)/flux_sum) - (x * y)

    lhs = (x2 + y2)/2
    rhs = np.sqrt(np.power((x2 - y2)/2, 2) + np.power(xy,2))

    # Find the major/minor axes
    with np.errstate(invalid='raise'):
        try:
            major_axis = np.sqrt(lhs + rhs)
        except:
            major_axis = 0

        try:
            minor_axis = np.sqrt(lhs - rhs)
        except:
            minor_axis = 0

    # Solve for theta - major axis angle
    try:
        t = np.arctan((2 * xy) / (x2 - y2))
    except RuntimeWarning:
        t = 0

    # Shift theta to the correct value
    if xy < 0 < t:
        theta = (t - np.pi)/2
    elif t < 0 < xy:
        theta = (t + np.pi)/2
    else:
        theta = t/2

    # # Calculate kurtosis
    # if major_axis < 10:
    #     x4 = (x_indices - x) ** 4
    #     y4 = (y_indices - y) ** 4
    #
    #     X4, Y4 = np.meshgrid(x4, y4)
    #
    #     m4x = np.sum(values * X4) / flux_sum
    #     m4y = np.sum(values * Y4) / flux_sum
    #
    #     sx = np.sqrt(x2 + np.power(x, 2))
    #     sy = np.sqrt(y2 + np.power(y, 2))
    #
    #     kx = m4x / sx ** 4
    #     ky = m4y / sy ** 4
    #
    #     kurtosis = (kx + ky) / 2
    # else:
    #     kurtosis = -99

    return major_axis, minor_axis, theta


def get_light_distribution(pixel_values, flux_sum):

    # Sort pixels into order
    sorted_pixels = np.sort(pixel_values)

    # Half maximum radius

    # Find half the maximum pixel value
    half_max = np.max(pixel_values) * 0.5

    # Find the number of pixels of at least that value
    area = sorted_pixels.size - np.searchsorted(sorted_pixels, half_max)
    half_max_rad = find_radius(area)

    # Nth percentile area

    # Get the cumulative sums of the reverse sorted pixels
    summed_pixels = np.cumsum(sorted_pixels[::-1])

    # Find the flux contained by n% of the object's pixels
    thresholds = np.array([0.5, 0.1, 0.9]) * flux_sum

    # Find the first instance where the cumulative sum is over each threshold
    areas = np.searchsorted(summed_pixels, thresholds)

    # Convert the areas to approximate radii
    radii = find_radius(areas)
    
    return radii, half_max_rad


def find_radius(area):
    """Calculate the radius of a circle of a given radius"""
    return np.sqrt(area/np.pi)