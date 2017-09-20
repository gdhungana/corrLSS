import numpy as np
import healpy as hp

def countperpix(theta,phi,nside=32):
    """
    args: 
         theta, phi in radian
         nside: healpix param nside
    return:
         normalized count map 
    """
    hppix=hp.ang2pix(nside,theta,phi)
    count=np.zeros(12*nside**2) #- refer to healpy document
    for ii in range(len(count)): #- np.bincount will be faster than this loop
        k=np.where(ii==hppix)[0]
        if len(k) > 0: 
            count[ii]=len(k)
    count=count/np.max(count).astype(float)
    return count


def make_mask(theta,phi,thresh=0.2,nside=32,outfile=None,throwfile=None):
    """
    args: 
        theta,phi: angular positions in radians
    return:
        mask map
        throwmask map : that doesn't pass the threshold cut
    """
    
    count=countperpix(theta,phi,nside=nside)
    k=np.where(count!=0)[0]
    meancount=np.mean(count[k]) #- mean density
    print "Mean density", meancount    
    mask=np.zeros_like(count)
    throwmask=np.zeros_like(count)
    valid=np.where(count>=thresh*meancount)[0]
    invalid=np.logical_and(count>0,count<thresh*meancount)
    print "Invalid pixels", invalid[invalid].shape[0]
    mask[valid]=1
    throwmask[invalid]=count[invalid]
   
    if outfile is not None:
        hp.write_map(outfile,mask) #- healpix map
        print "Written mask map file", outfile
    if throwfile is not None:
        hp.write_map(throwfile,throwmask) #- throwmask
        print "Written throw mask map file", throwfile
    return mask,throwmask


    
