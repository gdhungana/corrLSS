import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.io import fits,ascii
from corrLSS.mask import make_mask
from corrLSS.util import radec2thetaphi, uniform_sphere, ra_dec_to_xyz, apply_mask
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

def make_random(catfile,maskfile=None,savemaskfile=None,savethrowfile=None, outfile=None,factor=8,thresh=0.1,objtype=None,truthfile=None):
    print("Reading Input catalog: {}".format(catfile))
    #datacat=Table.read(catfile)
    cat=fits.open(catfile)
    datacat=cat[1].data

    ztypes=['Z_COSMO','TRUEZ','Z']
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
    if truthfile is not None: #- required to match targetid for ra,dec
        tru=fits.open(truthfile)
        trucat=tru[1].data
        truid=trucat['TARGETID']
        dataid=datacat['TARGETID']
        #- map targetid sorted as in dataid
        tt=np.argsort(truid)
        ss=np.searchsorted(truid[tt],dataid)
        srt_idx=tt[ss]
        np.testing.assert_array_equal(truid[srt_idx],dataid)
        print("100% targets matched for data catalog")
        ra=trucat['RA'][srt_idx]
        dec=trucat['DEC'][srt_idx]
    else:        
        ra=datacat['ra']
        dec=datacat['dec']
    
    #-select the specified object
    if objtype is not None:
        print("Selecting obj type {} for randoms".format(objtype))
        try:
            kk=np.where(datacat['SOURCETYPE']==objtype)[0]
            print("Using sourcetype {}".format(objtype))
        except:
            try:
                kk=np.where(datacat['SPECTYPE']==objtype)[0]
                print("Using spectype {}".format(objtype))
            except:
                print("Objtype doesn't match header key. Check fits header")
        print("Total {} in the data: {}".format(objtype,len(kk)))
        ra=ra[kk]
        dec=dec[kk]
        z=z[kk]
    else:
        print("Working on full catalog")
        print("Total objects: {}".format(len(z)))
        
    #- mask first
    if maskfile is None:
        print("Creating mask")
        theta,phi=radec2thetaphi(ra,dec)
        mask,throwmask=make_mask(theta,phi,thresh=thresh,nside=32,outfile=savemaskfile,throwfile=savethrowfile)
    else:
        print("Reading maskfile: ".format(maskfile))
        mask=hp.read_map(maskfile)
    print("Generating random: factor {}".format(factor))
    ra_rnd,dec_rnd,z_rnd=generate_rnd(z,mask,factor=factor)
    wt_rnd=np.ones_like(ra_rnd)

    print("Data size: {}".format(len(ra)))
    print("Random size: {}".format(len(ra_rnd))) 
    if outfile is not None:
        randata=Table([ra_rnd,dec_rnd,z_rnd,wt_rnd],names=('ra','dec','z','wt'))
        randata.write(outfile,overwrite=True)
        print("Written random catalog file {}".format(outfile))


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
    print(allind.shape)
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
        print("Meancount: {}".format(meancount))
        throw=np.where((count>0) & (count < thresh*meancount))[0]
        print("No. of tiles thrown: {}".format(len(throw)))
        #- remove those tiles
        for jj in throw:
            allind=filter(lambda a:a!=jj,allind) 
    
    masked_index=np.in1d(np.arange(len(z_rnd)),allind)

    ra_rnd = ra_rnd[masked_index]
    dec_rnd = dec_rnd[masked_index]
    z_rnd = z_rnd[masked_index]
    
    return ra_rnd,dec_rnd,z_rnd
