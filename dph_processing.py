# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:45:05 2016

@author: quentinpeter
"""
import numpy as np
import registrator.image as ir
import registrator.channel as cr
import scipy
import json
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.signal import savgol_filter
import cv2
import os
import background_rm as rmbg
import matplotlib.image as mpimg
from tifffile import imread, imsave
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import re

from scipy.ndimage.filters import gaussian_filter1d as gfilter


def center(prof, pxwidth):
    prof = prof.copy()
    prof[np.isnan(prof)] = 0
    synthetic = np.zeros_like(prof)
    X = np.arange(len(synthetic), dtype=float)
    X -= len(synthetic) / 2
    synthetic += np.exp(-X**2 / (2 * (pxwidth / 2)**2))
    conv = np.convolve(prof, synthetic)
    return np.nanargmax(conv) - len(synthetic) / 2


def get_normalized_side_channel(im, pxwidth, a=None, maxSide=None):
    if a is None:
        # remove eventual angle
        eim = cr.Scharr_edge(im)
        nanmask = np.isnan(eim)
        eim[nanmask] = 0
        eim = ir.rotate_scale(eim, np.pi / 4, 1)
        a = ir.orientation_angle(eim > np.percentile(eim, 95)) + np.pi / 4
        # rotate
        im2 = ir.rotate_scale(im, -a, 1, borderValue=np.nan)

        # remove eventual angle
        if maxSide is None:
            half = int(np.mean(im.shape) // 4)
            maxSide = [im2[:half, :], im2[:, -half:],
                       im2[-half:, :], im2[:, :half]]
            maxSide = np.asarray([np.nansum(i) for i in maxSide])
            maxSide = maxSide.argmax()
        a += np.pi / 2 * maxSide
    im = ir.rotate_scale(im, -a, 1, borderValue=np.nan)

    # find the channel position
    prof = np.diff(np.nanmean(im, 1))
    valid = np.isfinite(prof)
    prof[valid] = scipy.signal.savgol_filter(prof[valid], 21, 3)
    top_idx = np.nanargmin(prof[5:-5]) + 5

    border = int(np.ceil(np.abs(np.tan(ir.clamp_angle(a)) * im.shape[0])))
    xSlice = gfilter(np.nanmean(im[top_idx:, :], 0), 1)[border:-border]
    xSlice = xSlice - np.nanmedian(xSlice)

    cpos = center(xSlice, pxwidth) + border

    left_idx = cpos - pxwidth / 2
    right_idx = cpos + pxwidth / 2

    return a, int(np.round(left_idx)), int(np.round(right_idx)), int(np.round(top_idx))


def getBase(im, left_idx, right_idx, channelMask):
    bases = np.nanmean(im[:, :left_idx], 1)
#    bases += np.nanmean(im[:, right_idx+1:], 1)
#    bases /=2

    return bases


def getmask(im, maskmargin=0):
    im = cv2.GaussianBlur(im, (21, 21), 0)
    mask = im < np.nanmedian(im)
    mask = im < np.nanmedian(im) + 3 * np.nanstd(im[mask])
    mask = binary_dilation(mask, iterations=5)
    mask = binary_erosion(mask, iterations=maskmargin + 5, border_value=1)

    return np.squeeze(mask)


def get_times(Metadata):
    times = np.asarray(Metadata["Times [s]"])
    inittime = Metadata['Contact Time [s]']
    times -= inittime - 1
    times = times / 60

    return times


def get_images(metadata_fn, flatten=True):
    """Get flat images"""
    # Load metadata
    with open(metadata_fn) as f:
        Metadata = json.load(f)

    fns = Metadata["Images File Names"]
    pixel_size_um = Metadata["Pixel Size [m]"] * 1e6
    channel_width_um = Metadata['Dead end width [m]'] * 1e6
    times = get_times(Metadata)
    exposure = Metadata['Exposure Time [s]']

    # Get images
    # If we have only one file, put it in a list
    if not isinstance(fns, list):
        fns = [fns]
    fns = [os.path.join(os.path.dirname(metadata_fn), fn) for fn in fns]
    ims = np.squeeze(np.asarray([imread(fn) for fn in fns], dtype=float))

    # Normalise for exposure
    if exposure is not None:
        if isinstance(exposure, list):
            ims /= np.array(exposure)[:, np.newaxis, np.newaxis]
        elif exposure > 0:
            ims /= exposure

    # Drop frames if needed
    firstframe = Metadata['First Good Frame']
    ims = ims[firstframe:]
    times = times[firstframe:]

    # Flatten the images
    if flatten:
        # Get a mask to ignore intensity from proteins when flattening
        last_im = ims[-1]
        mask_last_im = getmask(last_im)
        prof_main_channel = np.nanmean(last_im, 1)
        mask_main_channel = getmask(prof_main_channel)

        mask_last_im = mask_last_im * mask_main_channel[:, np.newaxis] > 0

        signal_over_background = (np.median(last_im[(1-mask_last_im) > 0])
                                  / np.median(last_im[mask_last_im]))
#        if signal_over_background <= 5:

        if "Background File Name" in Metadata:
            #            try:
            # Load bg image
            bgfn = Metadata["Background File Name"]
            bgfn = os.path.join(os.path.dirname(metadata_fn), bgfn)
            bg = mpimg.imread(bgfn)

            if signal_over_background <= 10:
                # only flatten if not too large
                try:
                    last_im = rmbg.remove_curve_background(
                        last_im, bg, maskim=mask_last_im,
                        reflatten=False, use_bg_curve=True)
                    mask_last_im = getmask(last_im)
                    ims = rmbg.remove_curve_background(
                        ims, bg, maskim=mask_last_im,
                        reflatten=False, use_bg_curve=True)
                    flatten = False
                except:
                    ims = np.asarray(ims, float) - bg
                    mask_last_im = getmask(ims[-1])
            else:
                if bg.shape != ims.shape[1:]:
                    bg = cv2.resize(
                        bg, ims.shape[:0:-1], interpolation=cv2.INTER_AREA)
                bg_curve = rmbg.polyfit2d(bg, 2, mask=mask_last_im)
                if np.any(bg_curve < 0):
                    raise RuntimeError("Problematic")
                ims /= bg_curve * \
                    np.nanmean(ims[..., mask_last_im])/np.nanmean(bg_curve)
                ims -= 1
                flatten = False

#            except BaseException as e:
#                print(f'----\n Failed background removal {metadata_fn}, {signal_over_background}\n ----')

        else:
            print(f"Skipped {metadata_fn}", signal_over_background)

        if flatten:
            last_image = ims[-1]
            ims -= np.nanmedian(ims[-1][mask_last_im])
#                last_image_curve = rmbg.polyfit2d(last_image, 2, mask=mask_last_im)
#                if np.all(last_image_curve > 0):
#                    "Otherwise just stop"
#                    last_image = last_image / last_image_curve
#                    mask_last_im = getmask(last_image)
#                    ims = ims / rmbg.polyfit2d(ims, 2, mask=mask_last_im) - 1

    # Get channel position from last image
    channel_width_px = int(np.round(channel_width_um / pixel_size_um))
    angle, left_idx, right_idx, top_idx = get_normalized_side_channel(
        ims[-1], channel_width_px)

    for im in ims:
        im[:] = ir.rotate_scale(im, -angle, 1, borderValue=np.nan)

    # Normalise by the mean value of the fluorescence in the last 5 frames
    ims /= np.nanmedian(ims[-5:, :top_idx - 5])

    channel_position_px = [left_idx, right_idx, top_idx]
    return ims, channel_position_px, times


def get_profs(ims, channel_position_px, Metadata, maskmargin=20):
    """Extract profiles from flat images"""
    pixel_size_um = Metadata["Pixel Size [m]"] * 1e6
    channel_width_um = Metadata['Dead end width [m]'] * 1e6
    channel_width_px = int(np.round(channel_width_um / pixel_size_um))
    [left_idx, right_idx, top_idx] = channel_position_px

    # Get X_pos
    X_pos = np.arange(ims[0].shape[0]) * pixel_size_um
    X_pos -= top_idx * pixel_size_um

    # Get the profiles and the backgrounds
    profiles = np.ones(np.shape(ims)[:2])
    background_profiles = np.ones(np.shape(ims)[:2])

    data_mask = np.ones((ims[0].shape[1],))
    data_mask[left_idx - maskmargin:right_idx + maskmargin + 1] = 0
    data_mask = data_mask > 0

    def filter_prof(profile):
        profile = np.array(profile)
        valid = np.isfinite(profile)
        profile[valid] = savgol_filter(profile[valid], 21, 1)
        profile[valid] = savgol_filter(profile[valid], 21, 1)
        return profile

    for i, im in enumerate(ims):
        backprof_raw = getBase(im, left_idx - maskmargin,
                               right_idx + maskmargin, data_mask)
        background_profiles[i] = filter_prof(backprof_raw)

        channel_im = im[:, left_idx:right_idx + 1]  # +-1*channel_width_px//4
        channel_prof_raw = np.nanmean(
            channel_im[:, channel_width_px // 4:-channel_width_px // 4], 1)

        profiles[i] = filter_prof(channel_prof_raw)
    return X_pos, profiles, background_profiles


def get_Conc_str(Cm):
    if Cm == 0:
        return '0M'
    exp = int(np.floor(np.log10(np.abs(Cm)) / 3) * 3)
    unit_analyte = "M"
    if exp < -2:
        unit_analyte = 'mM'
        Cm *= 1e3
#        if exp < -5:
#            unit_analyte = 'uM'
#            Cm *= 1e3
    return '{:.3g}{}'.format(Cm, unit_analyte)

#    if plotdebug:
#        mask = np.asarray(maskold, float)
#        mask[mask==1.]=np.nan
#        figure()
#        imshow(im)
#        imshow(ir.rotate_scale(mask,-angle,1, borderValue=np.nan),
#               alpha=.5, cmap='Reds')
#        plt.title(fns[0])
#        plot([0, left_idx, left_idx, right_idx, right_idx, np.shape(im)[1]],
#             [top_idx, top_idx, top_idx+500/pixel_size_um, top_idx+500/pixel_size_um, top_idx, top_idx], 'r', alpha = .5)


def plot_and_save_diffusiophoresis(ims, channel_position_px, times,
                                   X_pos, profiles, background_profiles,
                                   metadata_fn, maskmargin, outfolder=None):

    cmap = matplotlib.cm.get_cmap('plasma')
    norm = LogNorm(vmin=.1, vmax=10)
    with open(metadata_fn) as f:
        Metadata = json.load(f)
    [left_idx, right_idx, top_idx] = channel_position_px
    analyte = Metadata["Analyte Type"]
    Cin = Metadata["Analyte Concentration In [M]"]
    Cout = Metadata["Analyte Concentration Out [M]"]
    pixel_size_um = Metadata["Pixel Size [m]"] * 1e6

    plt.figure()
    # Create dummy colormap for times
    colors = plt.imshow([[.1, 10], [.1, 10]], cmap=cmap, norm=norm)
    plt.clf()

    fig, ax = plt.subplots()
    # Plot curves
    for i, Y in enumerate(profiles):
        ax.plot(X_pos, Y, c=cmap(norm(times[i])), label="%.1fs" % times[i])
    # Plot background profiles
    ax.plot(X_pos,  background_profiles[-1], 'r')
    ax.set_xlim((-50, 600))
    ax.set_xlabel(r'Distance [$\mu$m]')  # , fontsize=18)
    ax.set_ylabel(r'Fluorescence [a.u.]')  # , fontsize=18)
    # plt.legend(loc=1)

    myTitle = "{}: Cin = {}, Cout = {}".format(
        analyte, get_Conc_str(Cin), get_Conc_str(Cout))
    ax.set_title(myTitle)  # , fontsize=18)
    fig.colorbar(colors, ax=ax).set_label(label='Time [min]')  # ,size=18)

    ax.plot([0, 0], [np.nanmin(Y), np.nanmax(Y)], 'black')

#    ax.set_yticks(ax.get_yticks()[:-1])
    #ylim = ax.get_ylim()

    add_inset(ims, channel_position_px,
              profiles, metadata_fn, maskmargin, ax)

    path = os.path.normpath(metadata_fn)
    path = path.split(os.sep)
    arg = next((i for i, x in enumerate(path) if x == Metadata["Date"]))

    assert(Metadata['Date'] == path[arg])
    if Metadata['Device Type'] == '500umX50um':
        channel_type = 'small_channel'
    elif Metadata['Device Type'] == '20mmx50um_curved':
        channel_type = 'long_channel'
    else:
        raise RuntimeError('Not implemented')

    if 'Proteins Concentration In [M]' in Metadata:
        Cpin = Metadata['Proteins Concentration In [M]']
        Cpout = Metadata['Proteins Concentration Out [M]']
        Cpin_str = get_Conc_str(Cpin)
        Cpout_str = get_Conc_str(Cpout)

    else:
        Cpin = Metadata['Proteins Concentration In [g/l]']
        Cpout = Metadata['Proteins Concentration Out [g/l]']
        Cpin_str = f'{Cpin:.3g}' + 'gpl'
        Cpout_str = f'{Cpout:.3g}' + 'gpl'

    content = 'o'
    if Cout > 0:
        content += '{}{}'.format(get_Conc_str(Cout), Metadata['Analyte Type'])
    if Cpout > 0:
        content += '{}{}'.format(Cpout_str, Metadata['Proteins Type'])

    content += '_i'
    if Cin > 0:
        content += '{}{}'.format(get_Conc_str(Cin), Metadata['Analyte Type'])
    if Cpin > 0:
        content += '{}{}'.format(Cpin_str, Metadata['Proteins Type'])
    if Metadata['Analyte Type'] == 'H2O':
        content += '0MH2O'

    add = ''
    number = re.findall('(\d+)_metadata.json', path[-1])
    if len(number) > 0:
        add = f'_{number[0]}'

    folders = [Metadata['Date'],
               channel_type,
               content + add]

    if outfolder is None:
        return
#    path[arg:])[:-14]#Remove _metadata.json
    outfn = os.path.join(outfolder, *folders)
    try:
        os.makedirs(outfn)
    except:
        pass

    plt.savefig(os.path.join(outfn, content) + '.pdf', bbox_inches='tight')

    imsout = ims - np.nanmin(ims)
    imsout = imsout * (2**16 / np.nanmax(imsout))
    imsout = np.asarray(imsout, "uint16")
    imsave(os.path.join(outfn, content) + '.tif', imsout)
    np.savez(os.path.join(outfn, content) + '.npz',
             profiles=profiles,
             X_pos=X_pos,
             times=times)


def add_inset(ims, channel_position_px,
              profiles, metadata_fn, maskmargin, axis):
    with open(metadata_fn) as f:
        Metadata = json.load(f)
    pixel_size_um = Metadata["Pixel Size [m]"]*1e6
    [left_idx, right_idx, top_idx] = channel_position_px
    # Inset
    displayed_im_idx = np.nanargmax(np.nanmax(profiles, -1))
    if np.nanmax(profiles[displayed_im_idx]) < 1.2 * np.nanmax(profiles[-1]):
        displayed_im_idx = -1

    displayed_im = ims[displayed_im_idx, top_idx - 20:top_idx + int(500 / pixel_size_um) + 20,
                       left_idx - maskmargin:right_idx + 1 + maskmargin]
    #    toppos = np.shape(displayed_im)[0]-20
    rightpos = np.shape(displayed_im)[1] - maskmargin

    displayed_im = cv2.GaussianBlur(displayed_im, (5, 5), 0)

    vmin = np.nanmedian(ims[-1, top_idx:, :left_idx])
    vmax = np.nanpercentile(displayed_im[20:, maskmargin:rightpos], 99)

    pos = axis.get_position()
    print(pos.x0)
    xp = pos.x0 + 6/7 * pos.width
    yp = pos.y0 + 2/10 * pos.height
    wp = pos.width/10
    hp = pos.height*8/10
    ax2 = plt.axes([xp, yp, wp, hp])
    ax2.imshow(displayed_im,
               extent=(0, pixel_size_um / 1000 * np.shape(displayed_im)[1],
                       pixel_size_um / 1000 *
                       np.shape(displayed_im)[0] - 20 * pixel_size_um / 1000,
                       -20 * pixel_size_um / 1000),
               vmin=vmin,
               vmax=vmax)
    ax2.plot(pixel_size_um / 1000 * np.array(
        [0,
         maskmargin,
         maskmargin,
         rightpos,
         rightpos,
         np.shape(displayed_im)[1]]),
        np.array([
                 0,
                 0,
                 .500,
                 .500,
                 0,
                 0
                 ]), 'r', alpha=.5)

    ax2.set_xticks([])
    ax2.set_yticks([])
