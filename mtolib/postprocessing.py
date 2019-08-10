"""Functions to generate statistics and labels from object maps"""

import numpy as np
import warnings
from skimage.color import label2rgb
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting, custom_model
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import Fittable1DModel, Parameter
from scipy.special import binom
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
from astropy.modeling.models import Sersic1D


FTS_NUM_POINTS = 10

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

def make_headings(prefix, n):
    ''' make headings of the form prefix0, prefix x / (n-1)... prefix1'''
    return list(map(lambda x : f'{prefix}{round(x/(n-1), 2)}', range(0,n)))


def get_image_parameters(img, gz_cat, object_ids, nodes, node_attribs, img_coords, params,):
    """Calculate the parameters for all objects in an image"""

    # Treat warnings as exceptions
    warnings.filterwarnings('error', category=RuntimeWarning, append=True)

    parameters = []
    headings = ['ID', 'Area', 'gz2_label', 'ra', 'dec']

    headings.extend(['ea', 'fy0', 'fa', 'gy0', 'ga', 'eb', 'fb', 'gb'])
    headings.extend(make_headings('circ_', FTS_NUM_POINTS))
    headings.extend(make_headings('circ_log_', FTS_NUM_POINTS))
    headings.extend(make_headings('conv_', FTS_NUM_POINTS))

    for i in range(1, 5):
        headings.extend(make_headings(f'ami_{i}_', FTS_NUM_POINTS))
        headings.extend(make_headings(f'ami_log_{i}_', FTS_NUM_POINTS))

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

        object_params = get_object_parameters(img, gz_cat, params.plot_dir, id_set[n], pixel_indices, nodes, node_attribs, img_coords)
        if object_params != None:
            parameters.append(object_params)

    # Return to printing warnings
    warnings.resetwarnings()

    return parameters

def get_pixel_coordinates(p, img_coords):
    delta = p - img_coords.ref_pixel
    coords = img_coords.ref_coords + np.dot(img_coords.ra_dec_per_pixel_matrix, delta)
    return coords 

def complex_moment(p, q, raw_moments):
    c = 0+0j
    for k in range(0, p+1):
        for l in range(0, q+1):
            c += binom(p,k)*binom(q,l)*((-1)**(q-l))*(1j**(p+q-k-l))*raw_moments[k+l, p+q-k-l]
    return c

def central_moment(p, q, M):
    c = 0
    #print(M)
    xc = M[1,0] / M[0, 0]
    yc = M[0,1] / M[0, 0]

    for m in range(0, p+1):
        for n in range(0, q+1):
            #c += binom(p,m)*binom(q,n)*((-xc)**(p-m))*(-yc**(q-n))*M[m, n]
            c += binom(p,m)*binom(q,n)*((-1)**(m+n))*(xc**m)*(yc**n)*M[p-m,q-n]
        
    return c


def central_moments(M):
    nm = np.empty(shape=M.shape)
    for p in range(nm.shape[0]):
        for q in range(nm.shape[1]):
            nm[p][q] = central_moment(p,q,M)
    return nm

# U = central moments
def normalized_moment(p, q, U):
    omega = (p + q + 2) / 2
    return U[p, q] / (U[0,0] ** omega)

def normalized_moments(M):
    if (M[0][0] == 0):
        return M

    U = central_moments(M)
    v = np.empty(shape=M.shape)
    for p in range(U.shape[0]):
        for q in range(U.shape[1]):
            v[p][q] = normalized_moment(p,q,U)
    
    return v


def log_transform(x):
    return np.sign(x) * np.log10(abs(x) + 1)

def flusser_invariants(raw_moments, area):
    ''' calculate flusser's moment invariants. see:
    http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/MOMINV/ 
    '''

    # transform raw moments to normalized (Translation and scale invariant) v_pq
    V = normalized_moments(raw_moments)

    c11 = complex_moment(1, 1, V)
    c20 = complex_moment(2, 0, V)
    c21 = complex_moment(2, 1, V)
    c12 = complex_moment(1, 2, V)
    c30 = complex_moment(3, 0, V)

    # rotation invariant moments
    f_1_1 = c11
    f_2_1 = c21*c12
    f_2_0 = c20*c12*c12
    f_3_0 = c30*c12*c12*c12

    invs = np.array([f_1_1.real, f_2_1.real, f_2_0.real, f_2_0.imag, f_3_0.real, f_3_0.imag])
    #print(invs)
    return invs

def normalized_AMIs(M):
    V = normalized_moments(M)
    c11 = complex_moment(1,1,V)
    c21c12 = complex_moment(2,1,V) * complex_moment(1,2,V)
    c30 = complex_moment(3,0,V)
    c03 = complex_moment(0,3,V)
    c12 = complex_moment(1,2,V)

    a = np.array([
        50*c11, 
        10000*c21c12, 
        5000*c30*c03, 
        100000000*c30*c12*c12*c12
    ])
    
    return np.abs(a)

def hu_invariants(M):
    v = normalized_moments(M)
    I1 = v[2,0] + v[0,2]

    i21 = v[2,0] - v[0,2]
    I2 = i21*i21 + 4*v[1,1]

    i31 = v[3,0] - 3*v[1,2]
    i32 = 3*v[2,1]-v[0,3]
    I3 = i31*i31 + i32*i32

    i41 = v[3,0]+v[1,2]
    i42 = v[2,1]+v[0,3]
    I4 = i41*i41 + i42*i42

    return np.array([I1,I2,I3,I4])

def normalized_AMIs2(M):
    ''' http://optics.sgu.ru/~ulianov/Bazarova/LASCA_literature/AffineMomentsInvariants.pdf '''
    u = central_moments(M)
    I1 = (u[2,0]*u[0,2]-u[1,1]*u[1,1])/np.power(u[0,0],4)
    I2 = (u[3,0]*u[3,0]*u[0,3]*u[0,3] - 6*u[3,0]*u[2,1]*u[1,2]*u[0,3] +
        4*u[3,0]*u[1,2]*u[1,2]*u[1,2] + 4*u[0,3]*u[2,1]*u[2,1]*u[2,1] - 
        3*u[2,1]*u[2,1]*u[1,2]*u[1,2]) / np.power(u[0,0],10)
    I3 = (u[2,0]*(u[2,1]*u[0,3] - u[1,2]*u[1,2]) -
        u[1,1]*(u[3,0]*u[0,3] - u[2,1]*u[1,2]) +
        u[0,2]*(u[3,0]*u[1,2]-u[2,1]*u[2,1])) / np.power(u[0,0],7)
    
    return np.array([I1, I2, I3, I1])

def circularity(M):
    """ Calculate the circularity C(S) of a shape given the moments"""
    eps = 10e-7
    with np.errstate(invalid='ignore'):
        U = central_moments(M)
        denom = 2*np.pi*(U[2,0] + U[0,2])
        if abs(denom) < eps:
            return 1.0
        else:
            return (U[0,0]*U[0,0]) / denom
        

def elongation(M):
    U = central_moments(M)
    s = U[2,0] + U[0,2]
    d0 = U[2,0] - U[0,2]
    d = np.sqrt(4*U[1,1]*U[1,1] + d0*d0)

    with np.errstate(invalid='ignore', divide='ignore'):
        return (s + d) / (s - d)

def mymean(x):
    if len(x) == 0:
        return 0
    else:
        return np.mean(x)

def smooth(y, box_pts):
    box_pts = min(box_pts, len(y))
    if box_pts == 0:
        return y
    else:
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

def pixel_corner_points(x, y):
    """ return an array containing the corner points of a pixel x,y """
    return np.array([[x,y],[x+1, y],[x+1,y+1],[x,y+1]])

    """ return center of the pixel x,y """
    #return np.array([[x+0.5,y+0.5]])

def pixel_corner_points(p):
    return pixel_corner_points(p[0],p[1])

def logbase(n, x):
    return np.log(x) / np.log(n)

def lerp(a, b, t):
    return (1 - t)*a + t*b

def fit_exponential(R, Mag, r0, r1, r2):
    y0 = np.interp(r0, R, Mag)
    y1 = np.interp(r1, R, Mag)
    y2 = np.interp(r2, R, Mag)

    b = logbase( (r1-r0)/(r2-r0), (y1-y0)/(y2-y0))
    a = -(y1-y0) / np.power(r1-r0, b)
    return (y0, a, b)

def sorted_array_contains(a, v):
    return a[np.minimum(len(a) - 1, np.searchsorted(a,v))] == v

def get_object_parameters(img, gz_catalogue, plot_dir, node_id, pixel_indices, nodes, node_attribs, img_coords):
    # skip root object
    if node_id == 0:
        return None

    """Calculate an object's parameters given the indices of its pixels"""
    p = [node_id]

    parents = nodes['parent']
    areas = nodes['area']

    # Get pixel values for an object
    #pixel_values = np.nan_to_num(img.data[pixel_indices])
    #pixel_areas = areas[pixel_indices]
    #pixel_radii = np.sqrt(pixel_areas)
    pixel_attribs = node_attribs[pixel_indices]
    pixel_parents = parents[pixel_indices]
    num_pixels = len(pixel_attribs)

    # get central peak coordinates
    # calculate ra,dec coordinates of central peak
    peak_index = np.argmax(pixel_attribs['brightness'])
    peak_brightness = pixel_attribs[peak_index]['brightness']
    peak_x, peak_y = pixel_indices[0][peak_index], pixel_indices[1][peak_index]
    peak_coords = get_pixel_coordinates(np.array([peak_y, peak_x]), img_coords)

    # for stripe82 coadd
    peak_coords[1], peak_coords[0] = peak_coords[0], peak_coords[1]

    ra_str = round(peak_coords[0], 4)
    dec_str = round(peak_coords[1], 4)

    label_txt = ''
    if gz_catalogue != None:
        sky_cat, labels = gz_catalogue
        s = SkyCoord(peak_coords[0] * u.deg, peak_coords[1]*u.deg)
        idx, d2d, _ = s.match_to_catalog_sky(sky_cat)
        lbl = labels[idx].decode('UTF-8')
        label_txt = f'{lbl}'
        if d2d > 0.05 * u.arcminute:
            return None

    print(f'object {node_id} / {label_txt}: x,y={(peak_x, peak_y)},ra,dec = {peak_coords}, area = {num_pixels}')

    indices_flat_sorted = np.sort(np.ravel_multi_index(pixel_indices, nodes.shape))

    R = [] #radius
    Brightness = [] # intensity
    Circ = [] #circularity
    AMI1 = [] # affine moment invariants
    AMI2 = []
    AMI3 = []
    AMI4 = []
    #Elongation = []
    Convexity = []

    inside_object = True
    idx_flat = np.ravel_multi_index(np.array([peak_x, peak_y]), img.shape)
    min_x, min_y = peak_x, peak_y
    max_x, max_y = min_x, min_y

    # initialize convex hull
    # ch_pixels = np.transpose(pixel_indices)
    # ch_points = np.empty((len(ch_pixels)*4,2))
    # for i in range(0, len(ch_pixels)):
    #     ch_points[i*4]    = ch_pixels[i]
    #     ch_points[i*4+1]  =ch_pixels[i]+[1,0]  
    #     ch_points[i*4+2]  =ch_pixels[i]+[1,1]
    #     ch_points[i*4+3]  =ch_pixels[i]+[0,1]

    # ch_points = np.unique(ch_points, axis=0)
    # CH = ConvexHull(points=ch_points)
    # convexity=len(ch_pixels)/CH.volume

    br_sorted_indices = pixel_attribs['brightness'].argsort()[::-1]
    Convexity_Brightness = []
    ch_pixels = np.transpose(pixel_indices)
    ch_points = set()
    ch_area = 0

    # re-calculate CH a max of 250 times 
    ch_use_indices = np.round(np.linspace(0, len(ch_pixels) - 1, min(150, len(ch_pixels-1))).astype(int))
    ch_use_current_index = 0

    for i in range(0, len(ch_pixels)):
        px,py = ch_pixels[i][0], ch_pixels[i][1]
        ch_points.add((px  , py  ))
        ch_points.add((px+1, py  ))
        ch_points.add((px+1, py+1))
        ch_points.add((px  , py+1))

        ch_area += 1

        if ch_use_indices[ch_use_current_index] == i:
            Convexity_Brightness.append(pixel_attribs[br_sorted_indices[i]]['brightness'])
            Convexity.append(ch_area/ConvexHull(points=list(ch_points)).volume)
            ch_use_current_index += 1


    ''' Traverse object from central peak to faintest regions '''
    while idx_flat > 0 and sorted_array_contains(indices_flat_sorted, idx_flat):
        cur_node_idx = np.unravel_index(idx_flat, img.shape)
        min_x = min(min_x, cur_node_idx[0])                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        max_x = max(max_x, cur_node_idx[0])
        min_y = min(min_y, cur_node_idx[1])
        max_y = max(max_y, cur_node_idx[1])

        attribs = node_attribs[cur_node_idx]
        b = attribs['brightness']

        M = np.reshape(attribs['moments'], (4,4))

        #CH.add_points(pixel_corner_points(*cur_node_idx))
        #print(f'good={CH.good} cur_node_idx={cur_node_idx} ch.area={CH.volume}')

        R.append(np.sqrt(areas[cur_node_idx]/np.pi))
        Brightness.append(b)
        Circ.append(circularity(M))
        #Elongation.append(elongation(M))
        

        amis = normalized_AMIs(M)
        amis = log_transform(amis)
        AMI1.append(amis[0])
        AMI2.append(amis[1])
        AMI3.append(amis[2])
        AMI4.append(amis[3])

        ''' move up the tree '''
        idx_flat = parents[cur_node_idx]

    R = np.array(R)
    Brightness = np.array(Brightness)
    Mag = log_transform(np.array(Brightness))
    with np.errstate(divide='ignore'):
        Mag_Normal = Mag / np.max(Mag)
    #Convexity_Brightness = np.log(pixel_attribs[br_sorted_indices]['brightness']+1)[:ch_area]


    ''' finish incremental processing of the convex hull '''
    #NP = int(len(R)/4)
    #s1 = Sersic1D(amplitude=np.sum(Brightness[:NP])*0.5, r_eff=R[-1]*0.5, n=1, fixed={'n': False}, 
    #    bounds={'r_eff': (R[0],R[-1])})
    #fit_sersic = fitting.LevMarLSQFitter()
    #sersic_model = fit_sersic(s1, R[:NP], Brightness[:NP], maxiter=250)

    #smooth data
    #AMI1 = smooth(AMI1, 3)
    #AMI2 = smooth(AMI2, int(len(R)/8))
    #AMI3 = smooth(AMI3, int(len(R)/8))
    #AMI4 = smooth(AMI4, int(len(R)/8))

    #print(Circ)
    # obtain features
    #features = get_features(R, moment_invs)

    # create cutout
    #cutout_size = max_x-min_x+1, max_y-min_y+1
    #cutout = Cutout2D(img.data, (peak_y,peak_x), size=cutout_size, mode='partial', fill_value=0.0)

    # K = np.transpose(pixel_indices)
    # K_min_x = np.min(K[:,0])
    # K_max_x = np.max(K[:,0])
    # K_min_y = np.min(K[:,1])
    # K_max_y = np.max(K[:,1])

    # K -= K_min_x, K_min_y
    # cutout = np.zeros((K_max_x-K_min_x+1, K_max_y-K_min_y+1))
    # min_brightness = 50000
    # max_brightness = -10000

    # for i in range(0, len(K[:,0])):
    #     K_x = K[i,0]
    #     K_y = K[i,1]
    #     node_br = node_attribs[K_x+K_min_x, K_y+K_min_y]['brightness']
    #     node_br = np.log(node_br + 1)
    #     min_brightness = min(min_brightness, node_br)
    #     max_brightness = max(max_brightness, node_br)
    #     cutout[K_x, K_y] = node_br

    # delta = max_brightness - min_brightness
    # if delta > 0:
        # for i in range(0, len(K[:,0])):
            # K_x = K[i,0]
            # K_y = K[i,1]
            # min_val = 0.01
            # node_br = node_attribs[K_x+K_min_x, K_y+K_min_y]['brightness']
            # cutout[K_x, K_y] = min_val + (1-min_val) * node_br / delta
    
    # fit exponential curves
    r0 = R[0]
    r1 = lerp(R[0], R[-1], 0.075)
    r2 = lerp(R[0], R[-1], 0.15)

    r3 = lerp(R[0], R[-1], 0.15)
    r4 = lerp(R[0], R[-1], 0.375)
    r5 = lerp(R[0], R[-1], 0.60)

    r6 = lerp(R[0], R[-1], 0.60)
    r7 = lerp(R[0], R[-1], 0.80)
    r8 = R[-1]

    ey0, ea, eb = ep0 = fit_exponential(R, Mag, r0, r1, r2)
    fy0, fa, fb = ep1 = fit_exponential(R, Mag, r3, r4, r5)
    gy0, ga, gb = ep2 = fit_exponential(R, Mag, r6, r7, r8)

    if plot_dir is not None:
        object_plot_dir_name=f'{plot_dir}/{label_txt}/{ra_str},{dec_str}'
        os.makedirs(object_plot_dir_name, exist_ok=True)

        #plt.subplot(1,1,1)
        #plt.imshow(cutout, origin='lower', interpolation='bicubic')
        #plt.savefig(f'{object_plot_dir_name}/cutout.png')
        #plt.clf()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.grid(True)
        plt.ylabel(r'$\log I$')
        plt.xlabel(r'$r$')
        plt.plot(R, Mag)
        #plt.plot(R, log_transform(sersic_model(R)))
        plt.savefig(f'{object_plot_dir_name}/brightness.pdf')
        plt.clf()

        plt.grid(True)
        plt.ylabel(r'$\log I$')
        plt.xlabel(r'$r$')
        plt.plot(R, Mag, linewidth = 3, alpha = 0.7)
        plt.plot(R[R <= r2], ey0 - ea* np.power(R[R <= r2]-r0, eb), '--k')
        Rbetween = R[np.logical_and(R >= r3, R <= r6)]
        plt.plot(Rbetween, fy0 - fa* np.power(Rbetween-r3, fb), ':k')
        plt.plot(R[R >= r6], gy0 - ga* np.power(R[R >= r6]-r6, gb), '-.k')
        plt.savefig(f'{object_plot_dir_name}/fit_brightness.pdf')
        plt.clf() 

        plt.grid(True)
        plt.ylabel(r'$\textbf{Circularity}$')
        plt.xlabel(r'$r$')
        plt.plot(R, Circ)
        plt.savefig(f'{object_plot_dir_name}/circ.pdf')
        plt.clf()

        # plt.grid(True)
        # plt.ylabel('')
        # plt.xlabel('Radius')
        # plt.plot(R, Mag_Normal, label='Normalized mag')
        # plt.plot(R, Elongation, label='Elongation')
        # plt.legend()
        # plt.savefig(f'{object_plot_dir_name}/elong.png')
        # plt.clf()

        plt.grid(True)
        plt.ylabel(r'\textbf{Convexity}')
        plt.xlabel(r'$\log(I)$')
        plt.plot(Convexity_Brightness, Convexity)
        plt.gca().invert_xaxis()
        plt.savefig(f'{object_plot_dir_name}/convexity.pdf')
        plt.clf()    

        plt.grid(True)
        plt.ylabel(r'\textbf{AMI}')
        plt.xlabel(r'$r$')
        plt.plot(R, AMI1, '-')
        plt.plot(R, AMI2, '--')
        plt.plot(R, AMI3, '-')
        plt.plot(R, AMI4, '--')
        plt.savefig(f'{object_plot_dir_name}/ami.pdf')
        plt.clf()    

    p.append(num_pixels)
    p.append(label_txt)
    p.append(ra_str)
    p.append(dec_str)

    p.extend(get_features(R, Mag, Circ, Convexity_Brightness, Convexity, (AMI1, AMI2, AMI3, AMI4), (ep0, ep1, ep2)))

    return p

def get_features(r, mag, circ, ch_brightness, conv, amis, brightness_params):
    fts = []

    min_mag = np.min(mag)
    rlog = np.log(r)

    ''' brightness features '''
    ey0, ea, eb = brightness_params[0]
    fy0, fa, fb = brightness_params[1]
    gy0, ga, gb = brightness_params[2]

    ey0 -= min_mag
    fy0 -= min_mag
    gy0 -= min_mag

    br_features_1 = np.array([ea, fy0, fa, gy0, ga]) / ey0
    fts.extend(br_features_1)

    br_features_2 = [eb, fb, gb]
    fts.extend(br_features_2)

    ''' circularity features '''
    circ_features = np.interp(np.linspace(r[0], r[-1], FTS_NUM_POINTS), r, circ)
    circ_log_features = np.interp(np.linspace(rlog[0], rlog[-1], FTS_NUM_POINTS), rlog, circ)

    fts.extend(circ_features)
    fts.extend(circ_log_features)

    ''' convexity features. note: ch_brightness is decreasing so need to flip'''
    conv_features = np.interp(np.linspace(ch_brightness[-1], ch_brightness[0], FTS_NUM_POINTS), ch_brightness[::-1], conv[::-1])
    fts.extend(conv_features)

    ''' ami features.  '''
    for i in range(0, 4):
        ami_i_features = np.interp(np.linspace(r[0], r[-1], FTS_NUM_POINTS), r, amis[i])
        fts.extend(ami_i_features)
        
        ami_i_log_features = np.interp(np.linspace(rlog[0], rlog[-1], FTS_NUM_POINTS), rlog, amis[i])
        fts.extend(ami_i_log_features)

    return fts