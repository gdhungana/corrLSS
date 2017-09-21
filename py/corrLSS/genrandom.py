import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.io import fits,ascii
from mask import make_mask
from corrLSS.util import radec2thetaphi, uniform_sphere, ra_dec_to_xyz
from sklearn.neighbors import KDTree


def generate_rnd(z,mask,factor=8,nside=32):
    """
    Creating random that follows N(z)
    mask must have same hp resolution.
    """ 
    z_rnd = np.random.choice(z,size=factor*len(z)) #- ran z
    ra_rnd = 360.*np.random.random(size=factor*len(z)) #- deg
    cth_rnd = -1+2.*np.random.random(size=factor*len(z)) 
    dec_rnd = np.arcsin(cth_rnd)*180/np.pi #- deg (-90 to 90)
    theta,phi=radec2thetaphi(ra_rnd,dec_rnd)
    masked_pix = apply_mask(theta,phi,mask,nside=nside)
    ra_rnd = ra_rnd[masked_pix]
    dec_rnd = dec_rnd[masked_pix]
    z_rnd = z_rnd[masked_pix]

    return ra_rnd,dec_rnd,z_rnd

def make_random(catfile,maskfile=None,savemaskfile=None,savethrowfile=None, outfile=None,factor=8,thresh=0.1):
    print "Reading Input catalog: ", catfile
    #datacat=Table.read(catfile)
    cat=fits.open(catfile)
    datacat=cat[1].data
    ra=datacat['RA']
    dec=datacat['DEC']
    z=datacat['Z_COSMO']

    #- mask first
    if maskfile is None:
        print "Creating mask"
        theta,phi=radec2thetaphi(ra,dec)
        mask,throwmask=make_mask(theta,phi,thresh=thresh,nside=32,outfile=savemaskfile,throwfile=savethrowfile)
    else:
        print "Reading maskfile: ", maskfile
        mask=hp.read_map(maskfile)
    ra_rnd,dec_rnd,z_rnd=generate_rnd(z,mask,factor=factor)
    wt_rnd=np.ones_like(ra_rnd)

    print "Data size: ", len(ra)
    print "Random size: ", len(ra_rnd) 
    if outfile is not None:
        randata=Table([ra_rnd,dec_rnd,z_rnd,wt_rnd],names=('ra','dec','z','wt'))
        randata.write(outfile,overwrite=True)
        print "Written random catalog file",outfile


def generate_rnd_from_tiles(z,tilesra,tilesdec,factor=8,throw=False,thresh=0.05):
    #- read cat file for density
    radius = 1.605 #- tile radius
    size=factor*len(z)
    z_rnd = np.random.choice(z,size=size) #- ran z

    #-limits
    #ramin,ramax=(np.min(tilesra)-1.605,np.max(tilesra)+1.605)
    #decmin,decmax=(np.min(tilesdec),np.max(tilesdec))
    
    #- first create uniform sphere
    ra_rnd,dec_rnd = uniform_sphere(size=size)
    
    #- work in cartesian

    x1,y1,z1=ra_dec_to_xyz(ra_rnd,dec_rnd)
    x2,y2,z2=ra_dec_to_xyz(tilesra,tilesdec)
    data_rn=np.transpose([x1,y1,z1])
    data_tiles=np.transpose([x2,y2,z2])
    
    kdt_data = KDTree(data_rn)
    radius*=np.pi/180.
    ind = kdt_data.query_radius(data_tiles,r=radius) #- within the tile radius

    allind=np.concatenate(ind)
    print allind.shape
    #- throw tiles with very low density
    if throw:
        ntiles=len(tilesra)
        count=np.zeros(ntiles)
        for ii in range(ntiles):
            k=np.where(ii==allind)[0]
            if len(k)>0:
                count[ii]=len(k)

        count=count/np.max(count).astype(float)
        #- find mean density
        k=np.where(count!=0)[0]
        meancount=np.mean(count[k])
        print "Meancount:", meancount
        throw=np.where((count>0) & (count < thresh*meancount))[0]
        print "No. of tiles thrown:", len(throw)
        #- remove those tiles
        for jj in throw:
            allind=filter(lambda a:a!=jj,allind) 
    
    masked_index=np.in1d(np.arange(len(z_rnd)),allind)

    ra_rnd = ra_rnd[masked_index]
    dec_rnd = dec_rnd[masked_index]
    z_rnd = z_rnd[masked_index]
    
    return ra_rnd,dec_rnd,z_rnd
