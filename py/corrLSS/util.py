import numpy as np
import matplotlib.pyplot as plt
import treecorr as tc
import astropy.table
import healpy as hp 

def uniform_sphere(RAlim=None, DEClim=None, size=1):
    """Draw a uniform sample on a sphere

    Args: 
         RAlim : tuple (RAmin,RAmax)
         DEClim: tuple (Decmin,DECmax)
         size : int (optional)
                the size of the random arrays to return (default = 1)

    Return: 

        RA, DEC : ndarray
                  the random sample on the sphere within the given limits.
                  arrays have shape equal to size.
    """
    if DEClim is None:
        DEClim = (-90.,90.)
    if RAlim is None:
        RAlim = (0.,360.)

    cth_lim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    cth = cth_lim[0] + (cth_lim[1] - cth_lim[0]) * np.random.random(size)
    dec = (180. / np.pi) * np.arcsin(cth)
    ra = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)

    return ra, dec  #- in degrees

def ra_dec_to_xyz(ra, dec):
    """Convert ra & dec to Cartesian points

    Args:
           ra, dec : ndarrays

    Returns:
           x, y, z : ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return (cos_ra * sin_dec,
            sin_ra * sin_dec,
            cos_dec)


def radec2thetaphi(ra,dec): 
     """
     Args: 
          ra, dec in degrees
     Return:
          theta,phi in radians
     """
     theta=np.pi/2-dec*np.pi/180
     phi=np.pi/180. *ra
     return theta,phi 

def make_hpmap(nside=32,catfile=None,ra=None,dec=None):
    #- either catfile or ra and dec
    if catfile is not None:
        cat=Table.read(catfile)
        ra=cat['ra']
        dec=cat['dec']
    else:
        ra=ra
        dec=dec
    pix_nums = hp.ang2pix(nside,np.pi/2-dec*np.pi/180,ra*np.pi/180)
    bin_count = np.bincount(pix_nums,weights=np.ones(len(pix_nums),dtype=np.double))
    map_gal = np.append(bin_count,np.zeros(12*nside**2-len(bin_count)))
    return map_gal

def build_mask_map(nside=32,catfile=None,ra=None,dec=None,thresh=0.1):
    #- either catfile or ra and dec
    if catfile is not None:
        cat=Table.read(catfile)
        ra=cat['ra']
        dec=cat['dec']
    else:
        ra=ra
        dec=dec
    theta,phi=radec2thetaphi(ra,dec)
    mask,throwmask=make_mask(theta,phi,thresh=thresh,nside=nside)
    return mask,throwmask
    
def apply_mask(theta,phi,mask,nside=32):
    masked_pix_indices = np.where(mask!=0)[0]   
    pix_nums = hp.ang2pix(nside,theta,phi)
    masked_pix = np.in1d(pix_nums,masked_pix_indices)
    return masked_pix

def find_tilesradec(tilelistfile=None,tiledir=None, saveout=None):
    """
    Must give either tile list file: 
    eg: desimodel/data/footprint/desi-tiles.fits
    or tiledir where there are only tile files
    """
    import glob,os

    if tilelistfile is None or tiledir is None:
        raise IOError("Must give either tilelistfile or tiledir")
    
    #- From tilelisfile
    if tilelistfile is not None:
        tiles = Table.read(tilelistfile)
        indesi=np.where(tiles['IN_DESI']==1)[0]
        tilesra=tiles['RA'][indesi]
        tilesdec=tiles['DEC'][indesi]

    #- or get from the list of tile files
    elif tiledir is not None:
        tilesra=[]
        tilesdec=[]
        for file in os.listdir(tiledir):
            thistile=fits.open(file)
            ra=thistile[1].header['TILERA']
            dec=thistile[1].header['TILEDEC']
            tilesra.append(tilera)
            tilesdec.append(tiledec)

    if saveout is not None:
        np.savetxt(saveout,np.transpose([tilesra,tilesdec]),fmt="%.3f")
    return tilesra,tilesdec

def angular_distance(ra1,dec1,ra2,dec2): #- radec in degrees
   ra1=ra1*np.pi/180.0; dec1=dec1*np.pi/180.0
   ra2=ra2*np.pi/180.0; dec2=dec2*np.pi/180.0

   #- Using wiki:https://en.wikipedia.org/wiki/Angular_distance
   gamma=np.sin(dec1)*np.sin(dec2)+(np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))
   gamma=np.arccos(gamma)

   #- convert back to degree
   gamma*=180.0/np.pi

   return gamma


def random_sample_catalog(catfile,size=None,racut=None,deccut=None,zcut=None,seed=1234,sample=True,outfile='sample_catalog.fits'):
    import astropy
    #- cuts should be tuple
    print("Reading data catalog")
    #datatab=astropy.table.Table.read(catfile)
    cat=astropy.io.fits.open(catfile)
    datacat=cat[1].data
    try:
        z=datacat['Z_COSMO']
        print("Using Z_COSMO for z")
    except:
        try:
            z=datacat['TRUEZ']
            print("Using TRUEZ for z")
        except:
            try:
                z=datacat['Z']
                print("Using Z for z")
            except:
                raise ValueError("None of the specified z-types match. Check fits header")

    ra=datacat['ra']
    dec=datacat['dec']

    print("Catalog contains {} objects".format(len(ra)))
    
    if racut is not None:
        cut=np.logical_and(ra>racut[0],ra<racut[1])#,dec>0,dec<30,z>0.62,z<0.68)
        ra=ra[cut]
        dec=dec[cut]
        z=z[cut]
    if deccut is not None:
        cut=np.logical_and(dec>deccut[0], dec<deccut[1])
        ra=ra[cut]
        dec=dec[cut]
        z=z[cut]
    if zcut is not None:
        cut=np.logical_and(z>zcut[0],z<zcut[1])
        ra=ra[cut]
        dec=dec[cut]
        z=z[cut]
    print("Catalog contains {} objects after cuts".format(len(ra)))
    
    if sample:
        print("Sampling from the data catalog")
        rst=np.random.RandomState(seed)
        idx=rst.choice(z.shape[0],size,replace=False)
    
        ra=ra[idx]
        dec=dec[idx]
        z=z[idx]
    else:
        print("writing data after cuts")

    randdata=astropy.table.Table([ra,dec,z],names=('RA','DEC','Z'))
    randdata.write(outfile,format='fits',overwrite=True)

    
