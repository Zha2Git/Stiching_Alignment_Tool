import numpy as np
from skimage.transform import rescale
from pystackreg import StackReg
import glob
import gc
import os
import sys
import tifffile
from sys import getsizeof
from datetime import datetime

import pandas as pd

def get_filename():
  #Get the filename from the CSV file.
  data_strct = pd.read_csv("channel_name.csv")
  file_name = []
  name_header = data_strct.columns[0]
  for i in range(data_strct.shape[0]):
    file_name.append(data_strct[name_header][i])

  return file_name

def get_tiffiles(source):
  #Fetches the tif filenames in the source folder
    data_strct = pd.read_csv("channel_name.csv")
    name_header = data_strct.columns[0]
    file_name = []
    for i in range(data_strct.shape[0]):
      base = data_strct[name_header][i]
      if base.lower().endswith('.ome.tif'):
        base = base[:-8]
      elif base.lower().endswith('.tif'):
        base = base[:-4]
      hits = (glob.glob(os.path.join(source, base) + '.ome.tif') or
              glob.glob(os.path.join(source, base) + '.tif'))
      if not hits:
        available = glob.glob(os.path.join(source, '*.tif'))
        raise IndexError(
          f"Filename in CSV '{data_strct[name_header][i]}' not found in '{source}'.\n"
          f"Available .tif files: {[os.path.basename(f) for f in available]}"
        )
      file_name.append(os.path.split(hits[0])[-1])
    return file_name

def get_aligned_tiffiles(source):
  #Fetches the tif filenames in the source folder
    data_strct = pd.read_csv("channel_name.csv")
    name_header = data_strct.columns[0]
    file_name = []
    for i in range(data_strct.shape[0]):
      base = data_strct[name_header][i]
      if base.lower().endswith('.ome.tif'):
        base = base[:-8]
      elif base.lower().endswith('.tif'):
        base = base[:-4]
      hits = (glob.glob(os.path.join(source, base) + '_al.ome.tif') or
              glob.glob(os.path.join(source, base) + '_al.tif'))
      if not hits:
        available = glob.glob(os.path.join(source, '*.tif'))
        raise IndexError(
          f"Filename in CSV '{data_strct[name_header][i]}' (aligned) not found in '{source}'.\n"
          f"Available .tif files: {[os.path.basename(f) for f in available]}"
        )
      file_name.append(os.path.split(hits[0])[-1])
    return file_name

def get_ref_map(data_strct, default_ref):
  """CR2: Build {filename: reference_channel_name} from channel_name.csv.
  Uses 'Reference_Channel' column when present; falls back to default_ref."""
  name_header = data_strct.columns[0]
  ref_map = {}
  has_ref_col = 'Reference_Channel' in data_strct.columns
  for i in range(data_strct.shape[0]):
    fname = data_strct[name_header][i]
    if has_ref_col:
      ref_val = str(data_strct['Reference_Channel'][i]).strip()
      ref_map[fname] = default_ref if (ref_val in ('nan', '')) else ref_val
    else:
      ref_map[fname] = default_ref
  return ref_map

def get_aligned_marker_names(ref_map):
  #creates a .txt folder that has the marker names in the correct order for the aligned images
  data = pd.read_csv("channel_name.csv")
  split_char = '|'
  #create a list of the channel names. in the order that they were saved as in CZI
  name_header = data.columns[0]
  # CR2: Exclude Reference_Channel metadata column from physical channel list
  channel_cols = [c for c in data.columns[1:] if c != 'Reference_Channel']
  chan_name = []
  #i is image
  for i in range(data.shape[0]):
    #j is channel
    for col in channel_cols:
      if(str(data[col][i]) != 'nan'):
        chan_name.append(data[col][i] +split_char+ col +split_char+ data[name_header][i])

  #Modify the list to put the reference channel as the first channel of each image
  img = chan_name[0].split(split_char)[2]
  img_idx = 0
  for i in range(len(chan_name)):
    current_img = chan_name[i].split(split_char)[2]
    if current_img == img:
      # CR2: Use per-image reference channel from ref_map
      img_ref = ref_map.get(img, next(iter(ref_map.values())))
      if (chan_name[i].split(split_char)[1] == img_ref):
        chan_name.insert(i-img_idx, chan_name.pop(i))
      img_idx += 1
    else:
      img_idx = 1
      img = current_img

  file_name = open("./aligned/marker_names_al.txt","w")
  for i in range(len(chan_name)):
    file_name.write(chan_name[i]+'\n')
  file_name.close()

def get_final_marker_names(args, ref):
  #creates a .txt folder that has the marker names in the correct order for the final image
  file_al = open("./aligned/marker_names_al.txt","r")
  split_char = '|'
  data = pd.read_csv("channel_name.csv")
  marker_al = file_al.readlines()
  marker_names_final = []
  for i in marker_al:
      marker_names_final.append(i)
  file_al.close()

  # CR1: Filter out channels marked as empty ("empty", "-", or blank/whitespace)
  def _is_excluded_marker(entry):
    v = entry.split(split_char)[0].strip()
    return v == '' or v == '-' or v.lower() == 'empty'
  marker_names_final = [m for m in marker_names_final if not _is_excluded_marker(m)]
  print('CR1: After filtering excluded channels:', len(marker_names_final), 'channels remain')

  #Remove the reference channel from the final image-->
  if False:  #Yang Zhang:  don't remove the reference channel
    for i in range(len(marker_names_final)-data.shape[0]+1):
      if (marker_names_final[i].split(split_char)[1] == ref):
        if i != 0:
          print('Removed reference channel', i,'called', marker_names_final[i])
          marker_names_final.pop(i) 
  
  print('Final image size will have:',len(marker_names_final), 'channels')

  if not args.fullname:
    print('-------------- Marker names only in the final image metadata')
    for i in range(len(marker_names_final)):
      marker_names_final[i] = marker_names_final[i].split(split_char)[0]+'\n'

  else:
    print('-------------- Full marker names in the final image metadata')

  # CR4: Write marker_names_final.txt to destination path
  output_txt_path = os.path.join(args.destination, 'marker_names_final.txt')
  file_name = open(output_txt_path, "w")
  for i in range(len(marker_names_final)):
    file_name.write(marker_names_final[i])
  file_name.close()

def get_max_shape(source):
  #this function gets the maximum dimensions from all the images that are going to be registered
  filepath = glob.glob(os.path.join(source,'image_shape.txt'), recursive=True)
  print ('Getting max dimensions with: ',filepath)
  file = open(filepath[0],'r')
  images_shape = file.read()
  split = images_shape.split(';')
  i_max = 0
  j_max = 0
  print(split)
  #-1 as last element is always an empty string
  for i in range(len(split)-1):
     frag = split[i].split(',')
     i_max = max(i_max,int(frag[1]))
     j_max = max(j_max,int(frag[2]))
  print('(i_max, j_max) --------------- ',i_max,j_max)

  return i_max,j_max

def pad_image(i_max, j_max, image):
  #pads the images if the dimensions are not the same for all of them
  i_diff = i_max - np.shape(image)[0]
  j_diff = j_max - np.shape(image)[1]

  #pads (up, down),(left,right)
  padded_image = np.pad(image,((int(np.floor((i_diff)/2)), int(np.ceil((i_diff)/2)))
                              ,(int(np.floor((j_diff)/2)), int(np.ceil((j_diff)/2)))),'constant')
  return padded_image

def get_metadata(filename, img_shape, mrk_nm, resolution):
  #name of the file you are running, shape of the image, the name of the markers, the scale of the image (pxl = XX um)
  #creates the metadata for the image to have it directly integrated with the ome.tif
  mdata = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?><!-- Warning: this comment is an OME-XML metadata block, which contains crucial dimensional parameters and other important metadata. Please edit cautiously (if at all), and back up the original data before doing so. For more information, see the OME-TIFF web site: https://docs.openmicroscopy.org/latest/ome-model/ome-tiff/. --><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" Creator="OME Bio-Formats 6.5.1" UUID="urn:uuid:e152586d-4214-4027-a620-edb3f6cfc6af" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">'''
  mdata = mdata + '''<Image ID="Image:0" Name="'''+filename+'''">'''
  mdata = mdata + '''<Pixels ID="Pixels:0" DimensionOrder="XYCZT" PhysicalSizeX="'''+str(resolution)+'''" PhysicalSizeY="'''+str(resolution)+'''" Type="float" SizeX="'''+str(img_shape[2])+'''" SizeY="'''+str(img_shape[1])+'''" SizeC="'''+str(img_shape[0])+'''" SizeZ="1" SizeT="1">'''
  for i in range(len(mrk_nm)):
    mdata = mdata + '''<Channel ID="Channel:0:'''+str(i)+'''" Name="'''+mrk_nm[i]+'''" SamplesPerPixel="1"><LightPath/></Channel>'''
  mdata = mdata + '''</Pixels></Image></OME>'''

  return mdata

def get_aligned_images(args, source):
  all_start_time = datetime.now()
  #function used to do registration on all images with respect to the first image of the file list.
  files = get_tiffiles(source)
  filename = get_filename()
  resolution = args.resolution

  #Read the csv file with the data structure
  data_strct = pd.read_csv("channel_name.csv")
  name_header = data_strct.columns[0]

  # CR2: Build per-image reference channel map; exclude Reference_Channel metadata column
  ref_map = get_ref_map(data_strct, args.reference)
  get_aligned_marker_names(ref_map)
  channel_cols = [c for c in data_strct.columns[1:] if c != 'Reference_Channel']

  # The following bit of code is to know which idx is the reference channel.
  # I create a new dataframe idx_values that return the idx of the reference channel with respect to the image
  # that is being treated
  # this takes into account the empty cells if a channel is not used for a given image.
  # (reference channel should always be used)
  idx_values = data_strct.copy()
  for i in range(data_strct.shape[0]):
  #i is image
    idx = 0
    for col in channel_cols:
    #j is channel
      if(str(data_strct[col][i]) != 'nan'):
        idx_values[col][i] = idx
        idx += 1

  # CR2: Use per-image reference channel for the first (fixed) image
  ref_0 = ref_map[filename[0]]
  print('Reference channel for first image:', ref_0)
  print ('Reference image is from:', files[0])
  tif_ref = tifffile.imread(os.path.join(source,files[0]))

  print('Loaded tif_ref', getsizeof(tif_ref)/10**6, 'MB')
  chan_ref = np.array(tif_ref[idx_values[ref_0][idx_values[name_header] == filename[0]].values[0]])
  print('Extracted chan_ref', getsizeof(chan_ref)/10**6, 'MB')

  #Do not need tif_ref only chan_ref from it (free memory)
  del tif_ref
  gc.collect()

  i_max, j_max = get_max_shape(source)

  # To see if padding is required
  if(np.shape(chan_ref)[0] == i_max and np.shape(chan_ref)[1] == j_max):
    pad_chan_ref = chan_ref
  else:
    print('---------------- Images will be padded -----------------')
    pad_chan_ref = pad_image(i_max, j_max, chan_ref)

  #If you asked for downscaling the image
  if args.downscale:
    anti_alias = True
    rescale_fct = args.factor
    print('----- Images will be downscaled',rescale_fct*100,'%  resolution------')
    resolution = round(resolution/rescale_fct,3)
    print('-------------- 1 pixel represents',resolution,'um --------------')
    pad_chan_ref = rescale(pad_chan_ref, rescale_fct, anti_aliasing=anti_alias, preserve_range = True)
    pad_chan_ref = pad_chan_ref.astype(np.float32)
    print('chan_ref rescaled', getsizeof(np.array(pad_chan_ref))/10**6, 'MB')
  else:
    print('--------------- Keeping full resolution ----------------')
    print('chan_ref ', getsizeof(np.array(pad_chan_ref))/10**6, 'MB')

  del chan_ref
  gc.collect()

  # If background subtraction is asked:
  if args.background != 'False':
    # CR2: Use reference channel of the image with the most channels for background alignment
    chan_counts = data_strct[channel_cols].notna().sum(axis=1)
    max_chan_img_name = data_strct[name_header][chan_counts.idxmax()]
    bg_ref = ref_map[max_chan_img_name]
    bg_ref_ch_idx = int(idx_values[bg_ref][data_strct[name_header] == max_chan_img_name].values[0])
    print('--- Aligning tif:', args.background)
    tif_mov = tifffile.imread(os.path.join(source,args.background+'.ome.tif'))
    print('Shape of image is: ', np.shape(tif_mov), 'size', getsizeof(tif_mov)/10**6, 'MB')
    print('------- Background reference channel:', bg_ref, '(idx:', bg_ref_ch_idx + 1, ')')
    # The reference marker channel position of background is the same as the round that has the
    # the most channels in of all the rounds
    chan_mov = np.array(tif_mov[bg_ref_ch_idx])
    del tif_mov
    gc.collect()
    if(np.shape(chan_mov)[0] == i_max and np.shape(chan_mov)[1] == j_max):
      pad_chan_mov = chan_mov
    else:
      print('Padding image size to', i_max, j_max)
      pad_chan_mov = pad_image(i_max, j_max, chan_mov)
    del chan_mov
    gc.collect()
    if args.downscale:
      pad_chan_mov = rescale(pad_chan_mov, rescale_fct, anti_aliasing=anti_alias, preserve_range = True)
      pad_chan_mov = pad_chan_mov.astype(np.float32)
      print('Down scaled the image to', np.shape(pad_chan_mov), getsizeof(np.array(pad_chan_mov))/10**6, 'MB')

    #Doing the registration between both channels
    print('Getting Transform matrix')
    #ATTENTION: If you want to do other types of registration from pystackreg, here is where you do it!!!!!
    # you have the following options: TRANSLATION, RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR
    # if you want more information, please check the following link
    # https://pystackreg.readthedocs.io/en/latest/
    ##### BE SURE TO CHANGE ALSO FOR THE OTHERS JUST BELOW, if you are doing another transformation #####
    sr = StackReg(StackReg.RIGID_BODY)
    sr.register(pad_chan_ref, pad_chan_mov)
    print('Registration matrix acquired and now transforming the channels')

    del pad_chan_mov
    gc.collect()

    #Reload tif_mov to align all the images with the transformation that we got from registration
    tif_mov = tifffile.imread(os.path.join(source,args.background+'.ome.tif'))

    aligned_images = []
    channels = np.shape(tif_mov)[0]
    for channel in range(channels):
      #To put reference channel as first channel in the image and to see if we need padding
      if channel == 0:
        if(np.shape(tif_mov)[1] == i_max and np.shape(tif_mov)[2] == j_max):
          pad_tif_mov = tif_mov[bg_ref_ch_idx,:,:]
        else:
          pad_tif_mov = pad_image(i_max, j_max, tif_mov[bg_ref_ch_idx,:,:])
      else:
        if(np.shape(tif_mov)[1] == i_max and np.shape(tif_mov)[2] == j_max):
          pad_tif_mov = tif_mov[0,:,:]
        else:
          pad_tif_mov = pad_image(i_max, j_max, tif_mov[0,:,:])

      #To free memory as we do not need these channels
      #if it was the first iteration, we want reference channel as our first registered channel
      #thus we have reference channel removed, after it is the normal order (remove first channel)
      if channel == 0:
        if np.shape(tif_mov)[0] > 1:
          tif_mov = np.delete(tif_mov, bg_ref_ch_idx, axis = 0)
      else:
        if np.shape(tif_mov)[0] > 1:
          tif_mov = np.delete(tif_mov, 0, axis = 0)

      #To downscale the channel if asked
      if args.downscale:
        pad_tif_mov = rescale(pad_tif_mov, rescale_fct, anti_aliasing=anti_alias, preserve_range = True)
        pad_tif_mov = pad_tif_mov.astype(np.float32)

      #Doing the image registration
      aligned_tif = sr.transform(pad_tif_mov)
      del pad_tif_mov
      gc.collect()

      #Due to the registration, some values become negative at the edges (with the transformation matrix)
      #Negative values are set to 0 as they are at the edge and can be considered as background
      aligned_tif[aligned_tif <= 0] = 0
      #Store as float32 to preserve decimal precision from registration transformations
      aligned_tif = aligned_tif.astype(np.float32)

      aligned_images.append(aligned_tif)
      print('info -- channel', channel,'aligned')
      del aligned_tif
      gc.collect()

    print('Transformed channels done, image is of size', np.shape(aligned_images),
                                  getsizeof(np.array(aligned_images))/10**6, 'MB')
    del tif_mov
    gc.collect()

    #To put the marker names in the metadata and also the scale/resolution
    mrk_nm = list(channel_cols)
    #Need to move reference name to first position
    mrk_nm.remove(bg_ref)
    mrk_nm.insert(0, bg_ref)
    print('------- The markers of the image are: ',mrk_nm)
    mdata = get_metadata(args.background,np.shape(aligned_images), mrk_nm, resolution)

    print('Saving aligned image \n')
    with tifffile.TiffWriter('./aligned/'+args.background+'_al.ome.tif',
                                 bigtiff = True) as tif:
      tif.write(np.array(aligned_images), description  = mdata) #

    del aligned_images
    gc.collect()


  # We have our reference channel, now we go get our channel to align
  for idx in range(len(filename)):
    start_time = datetime.now()
    # CR2: Use per-image reference channel
    ref_i = ref_map[filename[idx]]
    ref_ch_idx_i = int(idx_values[ref_i][idx_values[name_header] == filename[idx]].values[0])
    print('--- Aligning tif:', files[idx], '(ref channel:', ref_i, ')')
    tif_mov = tifffile.imread(os.path.join(source,files[idx]))
    print('Shape of image is: ', np.shape(tif_mov), 'size', getsizeof(tif_mov)/10**6, 'MB')
    print('------- Reference Channel idx in the image:', ref_ch_idx_i + 1)
    chan_mov = np.array(tif_mov[ref_ch_idx_i])

    #delete tif_mov as it is not needed for registration (only need chan_ref) this is done to free memory.
    del tif_mov
    gc.collect()

    #padding image if it is needed
    if(np.shape(chan_mov)[0] == i_max and np.shape(chan_mov)[1] == j_max):
      pad_chan_mov = chan_mov
    else:
      print('Padding image size to', i_max, j_max)
      pad_chan_mov = pad_image(i_max, j_max, chan_mov)
    del chan_mov
    gc.collect()

    #Downscaling image if asked
    if args.downscale:
      pad_chan_mov = rescale(pad_chan_mov, rescale_fct, anti_aliasing=anti_alias, preserve_range = True)
      pad_chan_mov = pad_chan_mov.astype(np.float32)
      print('Down scaled the image to', np.shape(pad_chan_mov), getsizeof(np.array(pad_chan_mov))/10**6, 'MB')

    #Doing the registration between both channels
    print('Getting Transform matrix')
    #ATTENTION: If you want to do other types of registration from pystackreg, here is where you do it!!!!!
    # you have the following options: TRANSLATION, RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR
    # if you want more information, please check the following link
    # https://pystackreg.readthedocs.io/en/latest/
    sr = StackReg(StackReg.RIGID_BODY)
    sr.register(pad_chan_ref, pad_chan_mov)
    print('Registration matrix acquired and now transforming the channels')
    # if you want to see the transformation matrix : sr.get_matrix()

    del pad_chan_mov
    gc.collect()

    #Reload tif_mov to align all the images with the transformation that we got from registration
    tif_mov = tifffile.imread(os.path.join(source,files[idx]))

    aligned_images = []
    channels = np.shape(tif_mov)[0]
    for channel in range(channels):

      #To put reference channel as first channel in the image and to see if we need padding
      if channel == 0:
        if(np.shape(tif_mov)[1] == i_max and np.shape(tif_mov)[2] == j_max):
          pad_tif_mov = tif_mov[ref_ch_idx_i,:,:]
        else:
          pad_tif_mov = pad_image(i_max, j_max, tif_mov[ref_ch_idx_i,:,:])
      else:
        if(np.shape(tif_mov)[1] == i_max and np.shape(tif_mov)[2] == j_max):
          pad_tif_mov = tif_mov[0,:,:]
        else:
          pad_tif_mov = pad_image(i_max, j_max, tif_mov[0,:,:])

      #To free memory as we do not need these channels
      #if it was the first iteration, we want reference channel as our first registered channel
      #thus we have reference channel removed, after it is the normal order (remove first channel)
      if channel == 0:
        if np.shape(tif_mov)[0] > 1:
          tif_mov = np.delete(tif_mov, ref_ch_idx_i, axis = 0)
      else:
        if np.shape(tif_mov)[0] > 1:
          tif_mov = np.delete(tif_mov, 0, axis = 0)

      #To downscale the channel if asked
      if args.downscale:
        pad_tif_mov = rescale(pad_tif_mov, rescale_fct, anti_aliasing=anti_alias, preserve_range = True)
        pad_tif_mov = pad_tif_mov.astype(np.float32)

      #Doing the image registration
      aligned_tif = sr.transform(pad_tif_mov)
      del pad_tif_mov
      gc.collect()

      #Due to the registration, some values become negative at the edges (with the transformation matrix)
      #Negative values are set to 0 as they are at the edge and can be considered as background
      aligned_tif[aligned_tif <= 0] = 0
      #Store as float32 to preserve decimal precision from registration transformations
      aligned_tif = aligned_tif.astype(np.float32)

      aligned_images.append(aligned_tif)
      print('info -- channel', channel,'aligned')
      del aligned_tif
      gc.collect()

    print('Transformed channels done, image is of size', np.shape(aligned_images),
                                  getsizeof(np.array(aligned_images))/10**6, 'MB')
    del tif_mov
    gc.collect()

    #To put the marker names in the metadata and also the scale/resolution
    marker_names_al = open('./aligned/marker_names_al.txt',"r")
    marker_al = marker_names_al.readlines()
    mrk_nm = []
    for mrk in range(len(marker_al)):
      if marker_al[mrk].split('|')[2].split('\n')[0] == filename[idx]:
        mrk_nm.append(marker_al[mrk].split('\n')[0])
    print('------- The markers of the image are: ',mrk_nm)
    mdata = get_metadata(filename[idx],np.shape(aligned_images), mrk_nm, resolution)

    print('Saving aligned image')
    with tifffile.TiffWriter('./aligned/'+filename[idx]+'_al.ome.tif',
                                 bigtiff = True) as tif:
      tif.write(np.array(aligned_images), description  = mdata)

    del aligned_images
    gc.collect()
    end_time = datetime.now()
    print('------------- Registration Duration of',filename[idx],': {}'.format(end_time - start_time),'\n')

  print('DONE! All images are registered')
  all_end_time = datetime.now()
  print('--- Total Registration Duration: {}'.format(all_end_time - all_start_time), '\n')


def remove_background(args, source, filename):
  data_strct = pd.read_csv("channel_name.csv")
  name_header = data_strct.columns[0]
  # CR2: Exclude Reference_Channel metadata column from physical channel list
  channel_cols = [c for c in data_strct.columns[1:] if c != 'Reference_Channel']
  # CR2: Resolve per-image reference channel
  has_ref_col = 'Reference_Channel' in data_strct.columns
  img_name = filename.replace('_al.ome.tif', '')
  ref = args.reference
  if has_ref_col:
    img_rows = data_strct[data_strct[name_header] == img_name]
    if not img_rows.empty:
      ref_val = str(img_rows['Reference_Channel'].values[0]).strip()
      if ref_val not in ('nan', ''):
        ref = ref_val

  bckgrd_tif = tifffile.imread(os.path.join(source, args.background+'_al.ome.tif'))
  tif = tifffile.imread(os.path.join(source,filename))

  #To get the channel structure of bckgrd we start at 1 as element 0 is Filename
  channels = list(channel_cols)
  channels.remove(ref)
  channels.insert(0, ref)

  marker_names_al = open(os.path.join(source, 'marker_names_al.txt'),"r")
  marker_al = marker_names_al.readlines()

  #We never do the first channel (which is the reference channel) so we start at 1
  tif_pos = 1
  #Going through the list of markers
  for mrk in range(len(marker_al)):
    #Find where our markers start at with their respective filename
    if marker_al[mrk].split('|')[2].split('\n')[0]+'_al.ome.tif' == filename:
      #Go through all the channels and find the right idx between bckgrd_tif and tif
      for chan in range(len(channels)):
        # if the channel names are the same and they are not DAPI then we do the subtraction
        if channels[chan] == marker_al[mrk].split('|')[1] and channels[chan] != ref:
          temp_tif = tif[tif_pos,:,:].astype(np.float64) - bckgrd_tif[chan,:,:].astype(np.float64)*args.backgroundMult
          #If some background value is larger than in the tif, negative values are set to 0
          temp_tif[temp_tif <= 0] = 0
          temp_tif = temp_tif.astype(np.float32)
          tif[tif_pos,:,:] = temp_tif
          del temp_tif
          gc.collect()

          tif_pos += 1
          break
  return tif


def final_image(args,source):
  # loads one registered image after another and saves them all into one image.
  # CR1: channels with marker "empty" in channel_name.csv are excluded from the merged output.
  # CR3: pyramidal image is generated here when args.pyramidal=True.
  # CR4: final image is saved to args.destination.
  print('-------------- Final image --------------')
  ref = args.reference
  resolution = args.resolution
  get_final_marker_names(args, ref)
  files = get_aligned_tiffiles(source)

  if args.downscale:
    resolution = round(resolution/args.factor,3)

  # CR1: Build per-file channel keep masks (exclude channels marked "empty")
  marker_al_path = os.path.join(source, 'marker_names_al.txt')
  with open(marker_al_path, 'r') as f:
    marker_al = f.readlines()

  def get_keep_indices(fname):
    # CR1: Returns channel indices where marker is not "empty", "-", or blank
    file_channels = [line for line in marker_al
                     if line.split('|')[2].strip('\n').strip() == fname]
    return [i for i, ch in enumerate(file_channels)
            if ch.split('|')[0].strip() not in ('', '-') and ch.split('|')[0].strip().lower() != 'empty']

  if args.background != 'False':
    print('------ Doing also Background Subtraction, using multiplier of',args.backgroundMult,' ------ ')
    tif = remove_background(args, source, files[0])
  else:
    tif = tifffile.imread(os.path.join(source,files[0]))

  # CR1: Apply channel filter for first file
  fname0 = files[0].replace('_al.ome.tif', '')
  keep0 = get_keep_indices(fname0)
  if len(keep0) < np.shape(tif)[0]:
    print('CR1: Skipping', np.shape(tif)[0] - len(keep0), '"empty" channel(s) from', files[0])
  tif = tif[keep0, :, :]

  final_img = tif
  print('Adding first image:', files[0], ' ------ ' ,np.shape(final_img))

  for idx in range(len(files)-1):
    print('--- Adding:', files[idx+1])
    if args.background != 'False':
      tif = remove_background(args, source, files[idx+1])
    else:
      tif = tifffile.imread(os.path.join(source,files[idx+1]))

    # CR1: Apply channel filter for each subsequent file
    fname_i = files[idx+1].replace('_al.ome.tif', '')
    keep_i = get_keep_indices(fname_i)
    if len(keep_i) < np.shape(tif)[0]:
      print('CR1: Skipping', np.shape(tif)[0] - len(keep_i), '"empty" channel(s) from', files[idx+1])
    tif = tif[keep_i, :, :]

    final_img = np.append(final_img, tif, axis = 0)
    print('Image size: ', getsizeof(np.array(final_img))/10**6, 'MB')

  print('Final image size: ',np.shape(final_img), getsizeof(np.array(final_img))/10**6, 'MB')

  # CR4: Read marker names from destination path
  with open(os.path.join(args.destination, 'marker_names_final.txt'), "r") as f:
    marker_final = f.readlines()
  mdata = get_metadata('final_image', np.shape(final_img), marker_final, resolution)

  final_image_np = np.array(final_img)

  # CR4:  write flat uncompressed final_image.ome.tif (or .czi) if args.nofinalflat is false (default).  If args.pyramidal, also write pyr_final_image.ome.tif (full-res + SubIFD levels in one pass).
  if not args.nofinalflat:
    out_path = os.path.join(args.destination, 'final_image_flat.ome.tif')
    if args.output == 'tif':
      with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        tif.write(final_image_np, description=mdata)
      print('Final image saved to:', out_path)
    elif args.output == 'czi':
      from pylibCZIrw import czi as pyczi
      out_path = os.path.join(args.destination, 'final_image_flat.czi')
      with pyczi.create_czi(out_path, exist_ok=True) as czi:
        channels, height, width = final_image_np.shape
        for ch in range(channels):
          array2d = final_image_np[ch, ...][..., np.newaxis]  # height x width x 1
          czi.write(data=array2d, plane={"C": ch})
        channel_names = {i: name.strip() for i, name in enumerate(marker_final)}
        czi.write_metadata(document_name='final_image',
                          channel_names=channel_names,
                          scale_x=resolution * 10**-6,
                          scale_y=resolution * 10**-6,
                          scale_z=resolution * 10**-6)
      print('Final image saved to:', out_path)
  else:
    print('Skipping writing flat final image (--nofinalflat)')
    
  # CR3: When --pyramidal, also write pyr_final_image.ome.tif (full-res + SubIFD levels in one pass)
  if args.pyramidal:
    print('CR3: Building pyramid levels for one-pass pyramidal write ...')
    data1 = np.stack([rescale(final_image_np[i], 0.5,   anti_aliasing=True, preserve_range=True).astype(np.float32)
                      for i in range(final_image_np.shape[0])])
    data2 = np.stack([rescale(final_image_np[i], 0.25,  anti_aliasing=True, preserve_range=True).astype(np.float32)
                      for i in range(final_image_np.shape[0])])
    data3 = np.stack([rescale(final_image_np[i], 0.125, anti_aliasing=True, preserve_range=True).astype(np.float32)
                      for i in range(final_image_np.shape[0])])
    pyr_path = os.path.join(args.destination, 'pyr_final_image_unpressed.ome.tif')
    tileSize = 512
    with tifffile.TiffWriter(pyr_path, bigtiff=True) as tif:
      options = dict(tile=(tileSize, tileSize), description=mdata)
      tif.write(final_image_np, subifds=3, **options)
      tif.write(data1, subfiletype=0, **options)
      tif.write(data2, subfiletype=0, **options)
      tif.write(data3, subfiletype=0, **options)
    print('CR3: Pyramidal image (4 levels, 512-tiled, uncompressed) saved to:', pyr_path)


def pyramidal_final_image(args):
  # loads one registed image after another and saves them all into one image. The first image keeps the reference channel
  # whereas we remove reference channels for all the others
  print('-------------- Pyramidal Uncompressed Final image --------------')
  ref = args.reference
  resolution = args.resolution
  #get_final_marker_names(args, ref)

  if args.downscale:
    resolution = round(resolution/args.factor,3)


  # CR4: Read final image and marker names from destination path
  data = tifffile.imread(os.path.join(args.destination, 'final_image.ome.tif'))
  print ('Initial image dimension',data.shape)

  marker_names_final = open(os.path.join(args.destination, 'marker_names_final.txt'), "r")
  marker_final = marker_names_final.readlines()
  mdata = get_metadata('final_image', np.shape(data), marker_final, resolution)

  scale = 0.5
  dataShape = rescale(data[0,:,:], scale, anti_aliasing=True, preserve_range = True)
  data1 = np.ones((data.shape[0],dataShape.shape[0],dataShape.shape[1]))
  del dataShape
  data1 = data1.astype(np.float32)
  for i in range(data.shape[0]):
    data1[i,:,:] = rescale(data[i,:,:], scale, anti_aliasing=True, preserve_range = True)
  print('Level 1 image dimension', data1.shape)

  scale = 0.25
  dataShape = rescale(data[0,:,:], scale, anti_aliasing=True, preserve_range = True)
  data2 = np.ones((data.shape[0],dataShape.shape[0],dataShape.shape[1]))
  del dataShape
  data2 = data2.astype(np.float32)
  for i in range(data.shape[0]):
    data2[i,:,:] = rescale(data[i,:,:], scale, anti_aliasing=True, preserve_range = True)
  print('Level 2 image dimension', data2.shape)

  scale = 0.125
  dataShape = rescale(data[0,:,:], scale, anti_aliasing=True, preserve_range = True)
  data3 = np.ones((data.shape[0],dataShape.shape[0],dataShape.shape[1]))
  del dataShape
  data3 = data3.astype(np.float32)
  for i in range(data.shape[0]):
    data3[i,:,:] = rescale(data[i,:,:], scale, anti_aliasing=True, preserve_range = True)
  print('Level 3 image dimension', data3.shape)

  #if you want to make it work for the small test set of images,
  #you need to change tileSize to 32 as the image is just above 500 pixels
  tileSize = 512
  comp = 'zlib' #zstd, lzma does not work
  #comp = 'uncompressed' # no compression in order to be able to read the image with QuPath (v0.4.3)

  #Write a tiled, multi-resolution, pyramidal, OME-TIFF file using zlib compression.
  #Sub-resolution images are written to SubIFDs:
  # CR4: Write pyramidal image to destination path
  pyr_out_path = os.path.join(args.destination, 'pyr_final_image.ome.tif')
  with tifffile.TiffWriter(pyr_out_path, bigtiff=True) as tif:
    options = dict(tile=(tileSize, tileSize), description  = mdata, compression = comp)
    tif.write(data, subifds=3, **options)

    tif.write(data1, subfiletype=0, **options)
    tif.write(data2, subfiletype=0, **options)
    tif.write(data3, subfiletype=0, **options)
  print('DONE, compressed pyramidal image saved to:', pyr_out_path)


