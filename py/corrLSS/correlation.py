import numpy as np
import matplotlib.pyplot as plt
import treecorr as tc
import astropy.table
import healpy as hp
from corrLSS.util import apply_mask,radec2thetaphi

# Cosmology
def set_cosmology():
    Omega_matter = 0.140247/0.6800232**2
    Omega_baryon = 0.022337/0.6800232**2
    Omega_curvature = 0
    H0 = 68.002320
    sigma_8 = 0.811322
    n_s = 0.963180
    from astropy.cosmology import FlatLambdaCDM
    cosmo=FlatLambdaCDM(H0=H0,Om0=Omega_matter)
    return cosmo

def arrange_catalog(catfile,rndfile=None,zmin=None,zmax=None,objtype=None,truthfile=None):
    """
    Use treecorr to evaluate two point correlation given a data catalog and a random catalog
    """
    
    print("Reading data catalog")
    #datatab=astropy.table.Table.read(catfile)
    cat=astropy.io.fits.open(catfile)
    datacat=cat[1].data
    try:
        z_data=datacat['Z_COSMO']
        print("Using Z_COSMO for z")
    except:
        try:
            z_data=datacat['TRUEZ']
            print("Using TRUEZ for z")
        except:
            try:
                z_data=datacat['Z']
                print("Using Z for z")
            except:
                raise ValueError("None of the specified z-types match. Check fits header")
    if truthfile is not None: #- required to match targetid for ra,dec
        tru=astropy.io.fits.open(truthfile)
        trucat=tru[1].data
        truid=trucat['TARGETID']
        dataid=datacat['TARGETID']
        #- map targetid sorted as in dataid
        tt=np.argsort(truid)
        ss=np.searchsorted(truid[tt],dataid)
        srt_idx=tt[ss]
        np.testing.assert_array_equal(truid[srt_idx],dataid)
        print("100% targets matched for data catalog")
        ra_data=trucat['RA'][srt_idx]
        dec_data=trucat['DEC'][srt_idx]
    else:        
        ra_data=datacat['ra']
        dec_data=datacat['dec']
    if objtype is not None:
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
        print("Total {} in the data: {}".format(objtype,len(kk)))
        ra_data=ra_data[kk]
        dec_data=dec_data[kk]
        z_data=z_data[kk]


    cosmo=set_cosmology()
    
    if zmin is None: zmin=np.min(z_data)
    if zmax is None: zmax=np.max(z_data)
    print("zmin:{} to zmax: {}".format(zmin,zmax))
    #TODO make this loop for differnt redshift bins to avoid reading catalogs each time

    wh=np.logical_and(z_data>zmin,z_data<zmax)
    ngal=np.count_nonzero(wh)
    print("Bin contains: {} galaxies".format(np.count_nonzero(wh)))
    print(cosmo.H0)

    cmvr_data=cosmo.comoving_distance(z_data[wh])*cosmo.H0.value/100.
    dmin,dmax=cosmo.comoving_distance([zmin,zmax])*cosmo.H0.value/100.
    print("Dmin to Dmax: {} to {}".format(dmin,dmax))
    print("Organizing data catalog to use")
    datacat=make_catalog(ra_data[wh],dec_data[wh],cmvr_data)

    if rndfile is not None:
        print("Reading random catalog")
        #rndtab=astropy.table.Table.read(rndfile)
        rnd=astropy.io.fits.open(rndfile)
        rndtab=rnd[1].data
        z_rnd=rndtab['z']
        ra_rnd=rndtab['ra']
        dec_rnd=rndtab['dec']
    
        whr=np.logical_and(z_rnd>zmin,z_rnd<zmax)
        nran=np.count_nonzero(whr)
        print("Bin Contains: {} random objects".format( np.count_nonzero(whr)))
        cmvr_rnd=cosmo.comoving_distance(z_rnd[whr])*cosmo.H0.value/100.
        print("Organizing random catalog to use")
        rndcat=make_catalog(ra_rnd[whr],dec_rnd[whr],cmvr_rnd)

        return datacat, rndcat
    else:
        return datacat

def correlate_tc(datacat,rndcat,outfile,cutoff=None):

    """
    datacat and randcat are tc.catalog object
    """
    
    print("Auto correlating data")
    dd=tc.NNCorrelation(min_sep=0.1,bin_size=0.025,max_sep=180.)
    dd.process(datacat)
    print("Auto correlating random")
    rr=tc.NNCorrelation(min_sep=0.1,bin_size=0.025,max_sep=180.)
    rr.process(rndcat)
    print("Cross Correlating")
    dr=tc.NNCorrelation(min_sep=0.1,bin_size=0.025,max_sep=180.)
    dr.process(datacat,rndcat)
    print("Calculating 2-pt. correlation")
    xi,xivar=dd.calculateXi(rr,dr)
    tab=astropy.table.Table([np.exp(dd.logr),xi,xivar],names=('r','xi','xivar'))
    tab.write(outfile,overwrite=True)

def random_data_xyz(datacat,bandwidth=0.2,format='xyz'):
    """
    data cat is treecorr catalog object and should have x, y, and z
    random is created here in xyz
    """
    from scipy.stats import gaussian_kde
    if format=='xyz':
        values=np.vstack([datacat.x,datacat.y,datacat.z])
        kde=gaussian_kde(values,bw_method=bandwidth/values.std(ddof=1))
        nx,ny,nz=kde.resample(2*len(datacat.z))
        randcat=tc.Catalog(x=nx,y=ny,z=nz)
    elif format=='radecr':
        values=np.vstack([datacat.ra/datacat.ra_units,datacat.dec/datacat.dec_units,datacat.r])
        kde=gaussian_kde(values,bw_method=bandwidth/values.std(ddof=1))
        nra,ndec,nr=kde.resample(2*len(datacat.r))
        randcat=tc.Catalog(ra=nra,dec=ndec,ra_units='deg',dec_units='deg',r=nr)
    return randcat
    
    
def make_catalog(ra,dec,cmvr=None): #- ra, dec in degrees
    cat=tc.Catalog(ra=ra,dec=dec,r=cmvr,ra_units='deg',dec_units='deg')
    return cat


def two_point(data,data_R,bins,method='landy-szalay',seed=1234,saverandom=False):
    """
    Uses nearest neighbors KDtree to evaluate two point correlation
    
    args:
        data: n samples x m features data array, eg. x,y,z positions
        bins: 1d bins array
    return:
        two - pt correlation correlation give the method.
        Errors are not returned. A bootstrap sampling can be run N times to 
        evaluate errors.         
    """
    from sklearn.neighbors import KDTree
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = np.random.RandomState(seed)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    n_samples, n_features = data.shape
    Nbins = len(bins) - 1

    # shuffle all but one axis to get background distribution
    if data_R is None:
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])
    else:
        data_R = np.asarray(data_R)
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')

    factor = len(data_R) * 1. / len(data)
    
    KDT_D=KDTree(data)
    KDT_R=KDTree(data_R)
    print("Correlating Data, data size: {}".format(len(data)))
    counts_DD=KDT_D.two_point_correlation(data,bins)
    print('Correlating Random, random size: {}'.format(len(data_R)))
    counts_RR=KDT_R.two_point_correlation(data_R,bins)
    
    DD=np.diff(counts_DD)
    RR=np.diff(counts_RR)

    #- Check for zero in RR
    RR_zero = (RR == 0)
    RR[RR_zero]=1

    if method == 'standard':
        corr = factor**2*DD/RR - 1
    elif method == 'landy-szalay':
        print("Cross Correlating")
        counts_DR=KDT_R.two_point_correlation(data,bins)
        DR=np.diff(counts_DR)
        print("Evaluating correlation using {}".format(method))
        corr = (factor**2 * DD - 2 * factor * DR + RR)/RR 
    corr[RR_zero] = np.nan
    return corr

def extract_catalog(catalog,zmin=None,zmax=None):
    print("Reading catalog.")
    tab = astropy.table.Table.read(catalog)
    ra = tab['RA']
    dec = tab['DEC']
    z = tab['Z']
    print("Objects in catalog: {}".format(len(z)))
    if zmin is None: zmin=np.min(z)
    if zmax is None: zmax=np.max(z)
    sel=np.where((z >= zmin) & (z < zmax))

    ra = ra[sel]
    dec = dec[sel]
    z = z[sel]
    print("Objects in this redshift bin".format(z.shape[0]))
    #- set cosmology
    print("Setting Fiducial Cosmology")
    cosmo = set_cosmology()
    cmv_r = cosmo.comoving_distance(z)*cosmo.H0.value/100.

    #- Coordinates:
    carx,cary,carz = ra_dec_to_xyz(ra,dec) * cmv_r

    #- set data:
    data=np.transpose([carx,cary,carz])
    return data

def make_data_R_catalog(datacat,outfile='random_from_datacat.fits',seed=1234):
    """
    Make random background from shuffling data
    """
    print("Reading data catalog")
    datatab=astropy.table.Table.read(datacat)
    ra = datatab['RA']
    dec = datatab['DEC']
    z = datatab['Z']
    data=np.transpose([ra,dec,z])
    
    #- create random by shuffling all but 1 axis
    print("Making random catalog from data")
    data_R = data.copy()
    n_samples, n_features = data.shape    
    rng = np.random.RandomState(seed)

    for i in range(n_features - 1):
        rng.shuffle(data_R[:, i])
    randdata=astropy.table.Table([data_R[:,0],data_R[:,1],data_R[:,2]],names=('RA','DEC','Z'))
    
    randdata.write(outfile,format='fits')
    print("Written Random file from data shuffling: {}".format(outfile))
    
def est_correlation(data,bins,data_R=None,method='landy-szalay'):
    
    #- correlation
    print("Evaluating 2-pt Correlation.")
    corr=two_point(data,bins,method=method,data_R=data_R)
    return bins,corr
