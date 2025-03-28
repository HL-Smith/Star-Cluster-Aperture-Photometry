# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:25:25 2025

@author: harve
"""

from astropy.wcs import WCS                                                    
from astropy.io import fits                                                    
import matplotlib.pyplot as plt
import matplotlib as pl
import photutils                                                               
import astroquery                                                               
from photutils.detection import DAOStarFinder                                  
from photutils.aperture import CircularAperture, aperture_photometry            
from astropy.stats import sigma_clipped_stats                                  
from astropy.stats import SigmaClip                                            
from photutils.background import SExtractorBackground                         
from astropy.visualization import SqrtStretch                                     
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture                                          
from astropy import coordinates                                                   
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from photutils.background import Background2D, MedianBackground, MMMBackground
import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from scipy.optimize import curve_fit
import iminuit
from iminuit import Minuit, cost
from iminuit.cost import LeastSquares
import numpy.ma as ma
import read_mist_models
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal
import seaborn as sns
from matplotlib.patches import Ellipse


'''Imports two FITS images for analysis, set image 1 to V filter if applicable'''
Image_1 = fits.open('M35_0001_v_60s.fits')    #M35_0001_v_60s.fits    #m34-0001_60s_v.fits  M37_0002_60s_v.fits    ngc744-0002_60s_v.fits
Image_2 = fits.open('M35_0001_b_60s.fits') #M35_0001_b_60s.fits       #m34-0001_60s_b.fits  M37_0002_60s_b.fits  NGC7686-0001_B_30.fits ngc744-0002_60s_b.fits

image_1 = Image_1[0].data
header_1 = Image_1[0].header
wcs_1 = WCS(header_1)


image_2 = Image_2[0].data
header_2 = Image_2[0].header
wcs_2 = WCS(header_2)

'''Aligns both images using the phase cross correlation function'''      
Shift,_,_ = phase_cross_correlation(image_1,image_2)   
print('Image 2 offset by: ',Shift)
shift_image_2 = shift(image_2, shift=(Shift[0],Shift[1]), mode='constant')     
    


'''Inputs'''
FWHM = float(input('FWHM value for identifying stars: ='))  
#Threshold = float(input('Threshold value for identifying stars: ='))
box_size=151
G=2.39 #CCD chip gain = 2.39
t_exp=60 #exposure time of images


'''Background Calculation for both images'''
def Background(image):
    """Calculate background using Background2D."""
    #sigma_clip = SigmaClip(sigma=3.0)
    #bkg = Background2D(image, box_size, bkg_estimator=SExtractorBackground(sigma_clip=sigma_clip))
    bkg = Background2D(image, box_size)
    return bkg.background
bkg_1=Background(image_1)
bkg_2=Background(image_2)

def star_finders(image, background, wcs, Threshold):
    """Run DAOStarFinder on the image."""
    #mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    daofind = DAOStarFinder(fwhm=FWHM, threshold=Threshold, exclude_border=True, brightest=200)
    dao_sources = daofind(image-background)
    for col in dao_sources.colnames:
        dao_sources[col].info.format = '%.8g'
        
     
    positions = np.array(list(zip(dao_sources['xcentroid'], dao_sources['ycentroid'])))
    #print("DAO Star Positions (pixels):", positions)   
    # Convert pixel coordinates to world coordinates (RA, Dec)
    #world_coords = np.column_stack(wcs_1.all_pix2world(dao_sources['xcentroid'], dao_sources['ycentroid'], 0))
    world_coords = np.column_stack(wcs.all_pix2world(dao_sources['xcentroid'], dao_sources['ycentroid'], 0))

   
    apertures = CircularAperture(positions, r=3)
    
    # Normalize and plot the image
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)
    plt.grid()
    #overlay = ax.get_coords_overlay('icrs')
    #overlay.grid(color='black', ls='dotted')
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.title("DAO Star Identifier: Sigma = "+str(FWHM)+"" , loc='left') 
    vminVal = 0.95*np.median(image)      # may need to adjust
    vmaxVal = 1.1*np.median(image)
    plt.title('')
    plt.imshow(image, vmin=vminVal, vmax=vmaxVal, cmap='Greys', origin='lower', norm=norm)
    ax.invert_yaxis()
    
    # Plot the apertures on the image
    apertures.plot(color='blue', lw=1.5, alpha=1)
    
      # Add labels for each detected star
    for i, (x, y) in enumerate(positions):
        # Add a label at the (x, y) coordinates
        ax.text(x + 5, y + 5, str(i+1), color='black', fontsize=8)

    return dao_sources

dao_1=star_finders(image_1, bkg_1, wcs_1, 80)
dao_2=star_finders(image_2, bkg_2, wcs_2, 70)

def plot_image(image, wcs):
    """Plot the input image."""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(stretch=SqrtStretch())
    vmin_val = 0.95 * np.median(image)
    vmax_val = 1.1 * np.median(image)
    plt.imshow(image, vmin=vmin_val, vmax=vmax_val, cmap='inferno', origin='lower', norm=norm)
    ax.invert_yaxis()
    #plt.title(title)
    plt.grid()
    plt.xlabel(r'RA', fontsize=16)
    plt.ylabel(r'Dec', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #plt.colorbar()
    plt.show()

plot_image(image_1, wcs_1)
plot_image(image_2, wcs_2)

def plot_Background(background, wcs, title="Background"):
    """Plot the calculated background."""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)
    plt.imshow(background, origin='lower', cmap='inferno')
    ax.invert_yaxis()
    #plt.title(title)
    #plt.colorbar()
    plt.grid()
    plt.xlabel(r'RA', fontsize=16)
    plt.ylabel(r'Dec', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
plot_Background(bkg_1, wcs_1)
plot_Background(bkg_2, wcs_2)

#def Analyse(image, wcs, image_name):
    # Compute global statistics
    #mean, median, std = sigma_clipped_stats(image, sigma=3)
    #background = Background(image)
    #dao_sources = star_finders(image, background, wcs)
    #return dao_sources


#dao_1 = Analyse(image_1, wcs_1, "Image 1")
#dao_2 = Analyse(shift_image_2, wcs_2, "Shifted Image 2")


def simbad_query_and_plot(image, background, wcs, dao_sources, RA, DEC, file_name,  separation=50):
    """Perform SIMBAD query and plot matches."""
    
    Simbad.add_votable_fields('pmra', 'pmdec', 'flux(V)', 'flux(B)', 'otype')    
    c = coordinates.SkyCoord(ra=RA, dec=DEC, frame='icrs', unit=(u.hourangle, u.deg))
    R = 18  # radius for the SIMBAD query, in arcminutes
    r = R * u.arcminute
    result_table = Simbad.query_region(c, radius=r)
    
    DATA_Simbad_RA = result_table['RA']  # Right Ascension
    DATA_Simbad_DEC = result_table['DEC']  # Declination
    DATA_Simbad_ID = result_table['MAIN_ID']  # SIMBAD ID
    DATA_Simbad_V = result_table['FLUX_V']  # Flux in V filter
    DATA_Simbad_B = result_table['FLUX_B']  # Flux in B filter
    DATA_Simbad_Obj = result_table['OTYPE']  # Object type
    DATA_SIMBAD_pmra = result_table['PMRA']  # Proper motion in RA
    DATA_SIMBAD_pmdec = result_table['PMDEC']  # Proper motion in Dec


    
    
    positions = np.array(list(zip(dao_sources['xcentroid'], dao_sources['ycentroid']))) 
    
    def aperturephotometry(image, positions, background):
        """Perform aperture photometry on the image."""
        apertures = CircularAperture(positions, r=FWHM)
        image_bkg = image - background  # Subtract the background
        #pixel_error = np.sqrt((image_bkg-883)/G+5.84**2)
        
        phot_table = aperture_photometry(image_bkg, apertures, error=None)
        Aperture_Sum = phot_table['aperture_sum']
        #Aperture_Sum_error = phot_table['aperture_sum_err']
        #Flux = G*Aperture_Sum/t_exp
        return Aperture_Sum


    Aperture_Sum = aperturephotometry(image, positions, background)
    mag_values = -2.5 * np.log10(Aperture_Sum/t_exp)
    N_flux=G*Aperture_Sum
    sigma_Flux = np.sqrt(N_flux)
    mag_err = (2.5/np.log(10))*(sigma_Flux/N_flux)
    positions_flux = np.array(list(zip(dao_sources['xcentroid'], dao_sources['ycentroid'], mag_values, mag_err))) 
    
   
    
    #print(positions_flux)
    
    
    # Convert SIMBAD coordinates to pixel coordinates
    simbad_coords = coordinates.SkyCoord(DATA_Simbad_RA, DATA_Simbad_DEC, frame='icrs', unit=(u.hourangle, u.deg))
    simbad_pixel_coords = np.column_stack(wcs.all_world2pix(simbad_coords.ra.deg, simbad_coords.dec.deg, 0))
    
    # Prepare plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.title(f"Double Matched Candidates with threshold {separation}: Sigma = {FWHM}, R = {R} arcmin", loc='left')
    vminVal = 0.95 * np.median(image)
    vmaxVal = 1.1 * np.median(image)
    plt.imshow(image, vmin=vminVal, vmax=vmaxVal, cmap='Greys', origin='lower', norm=norm)
    ax.invert_yaxis()

    # Store matched star data
    triple_matched_ra = []
    triple_matched_dec = []
    triple_matched_pmra = []
    triple_matched_pmdec = []
    triple_matched_ID = []
    triple_matched_V = []
    triple_matched_B = []
    triple_matched_Obj = []
    triple_matched_mag = []  # List to store DAO mag values
    triple_matched_mag_err = []
    triple_matched_indices = []
    triple_matched_wcs_ra = []  # List to store WCS RA
    triple_matched_wcs_dec = []  # List to store WCS Dec

    def Separation(x, y, separation):
        """Calculate distance between two points and check if below threshold"""
        return np.linalg.norm(np.array(x) - np.array(y)) < separation
    
    

    # Loop to find double matches and plot them
    # Loop to find double matches and plot them
    for i, simbad_coord in enumerate(simbad_pixel_coords):
        match_found = False
        for dao_coord in positions_flux:
            dao_coord_x, dao_coord_y, dao_coord_mag, dao_coord_mag_err = dao_coord  # Unpack (x, y, mag)
    
            if Separation(simbad_coord, (dao_coord_x, dao_coord_y), separation):  # Compare with (x, y)
                match_found = True
                break
    
        if match_found:
            # Plot the matched SIMBAD star
            SIMBAD_aperture = CircularAperture([simbad_coord], r=2.5)
            SIMBAD_aperture.plot(color='red', lw=1.5, alpha=1)
    
            # Use SIMBAD ID for labeling
            simbad_id_label = DATA_Simbad_ID[i]
            ax.text(simbad_coord[0] + 5, simbad_coord[1] + 5, f'{simbad_id_label}', color='red', fontsize=8)
    
            # Store matched star data including flux
            triple_matched_pmra.append(DATA_SIMBAD_pmra[i])
            triple_matched_pmdec.append(DATA_SIMBAD_pmdec[i])
            triple_matched_ra.append(DATA_Simbad_RA[i])
            triple_matched_dec.append(DATA_Simbad_DEC[i])
            triple_matched_ID.append(DATA_Simbad_ID[i])
            triple_matched_V.append(DATA_Simbad_V[i])
            triple_matched_B.append(DATA_Simbad_B[i])
            triple_matched_Obj.append(DATA_Simbad_Obj[i])
    
            # Store the DAO magnitude value
            triple_matched_mag.append(dao_coord_mag)
            triple_matched_mag_err.append(dao_coord_mag_err)
                        
            wcs_ra, wcs_dec = wcs.all_pix2world(dao_coord_x, dao_coord_y, 0)
            world_coord = SkyCoord(ra=wcs_ra, dec=wcs_dec, unit='deg', frame='icrs')
            wcs_ra_str = world_coord.ra.to_string(unit=u.hour, sep=':', precision=2)
            wcs_dec_str = world_coord.dec.to_string(unit=u.deg, sep=':', precision=2)
            triple_matched_wcs_ra.append(wcs_ra_str)
            triple_matched_wcs_dec.append(wcs_dec_str)
                      
            # Convert DAOStarFinder pixel coordinates to WCS (RA/Dec)
            #wcs_ra, wcs_dec = wcs.all_pix2world(dao_coord_x, dao_coord_y, 0)
            #wcs_ra_coord = SkyCoord(ra=wcs_ra * u.deg, dec=0 * u.deg, frame='icrs')
            #wcs_dec_coord = SkyCoord(ra=0 * u.deg, dec=wcs_dec * u.deg, frame='icrs')
            #wcs_ra_str = wcs_ra_coord.ra.to_string(unit=u.hour, sep=':', precision=2)
            #wcs_dec_str = wcs_dec_coord.dec.to_string(unit=u.deg, sep=':', precision=2)
            #triple_matched_wcs_ra.append(wcs_ra_str)
            #triple_matched_wcs_dec.append(wcs_dec_str)
            
    # Save data to Excel
    TripleMatchedArray = list(zip(
        triple_matched_ID,
        triple_matched_Obj,
        triple_matched_ra,
        triple_matched_dec,
        triple_matched_wcs_ra,  
        triple_matched_wcs_dec,
        triple_matched_V,
        triple_matched_B,
        triple_matched_pmra,
        triple_matched_pmdec,
        triple_matched_mag,
        triple_matched_mag_err))

    df = pd.DataFrame(TripleMatchedArray, columns=[
        'ID', 'Object Type', 'SIMBAD RA', 'SIMBAD Dec', 'DAO RA', 'DAO Dec', 'Flux (V)', 'Flux (B)', 'Proper Motion RA', 'Proper Motion Dec', 
        'AP Mag', 'AP Mag err'])
    df.to_excel(f'{file_name}.xlsx', index=False)

    print("Saved!")
    
    return {
        "V_cal": triple_matched_V,
        "B_cal": triple_matched_B,
        "ra": triple_matched_ra,
        "dec": triple_matched_dec,
        "ID": triple_matched_ID,
        "Inst": triple_matched_mag,
        'Err': triple_matched_mag_err,
        'pmra':  triple_matched_pmra,
        'pmdec': triple_matched_pmdec,
        'ra': triple_matched_ra,
        'dec': triple_matched_dec}
    
    print(TripleMatchedArray[:7])


    
x=simbad_query_and_plot(image_1,bkg_1,wcs_1, dao_1, '6h9m5.3s', '+24d20m10s','TripleMatchedStars_1')
y=simbad_query_and_plot(image_2,bkg_2,wcs_2, dao_2, '6h9m5.3s', '+24d20m10s','TripleMatchedStars_2')

#Decimal_Dec =   0.0

def Calibration():
   
    #V candidates
    V_cal = np.array(x['V_cal'], dtype=np.float64)
    V_inst = np.array(x['Inst'], dtype=np.float64)
    star_ids_V = np.array(x['ID'])  # SIMBAD IDs for V
    V_err = np.array(x['Err'], dtype=np.float64)
    
    pmra_V = np.array(x['pmra'], dtype=np.float64)
    pmdec_V = np.array(x['pmdec'], dtype=np.float64)
    #ra_V = np.array(x['ra'], dtype=np.float64)
    #dec_V = np.array(x['dec'], dtype=np.float64)
 
    B_cal = np.array(y['B_cal'], dtype=np.float64)
    B_inst = np.array(y['Inst'], dtype=np.float64)
    star_ids_B = np.array(y['ID'])  # SIMBAD IDs for B
    B_err = np.array(y['Err'], dtype=np.float64)
    
    pmra_B = np.array(y['pmra'], dtype=np.float64)
    pmdec_B = np.array(y['pmdec'], dtype=np.float64)
    
    simbad_coords = coordinates.SkyCoord(x['ra'], x['dec'], frame='icrs', unit=(u.hourangle, u.deg))
    ra_V=np.array(simbad_coords.ra.deg, dtype=np.float64)
    dec_V=np.array(simbad_coords.dec.deg, dtype=np.float64)
    
    

    # **Find Common IDs Between Both Images**
    common_ids = np.intersect1d(star_ids_V, star_ids_B)
    
    
    
    
    #print(np.shape(common_ids))

    # **Filter Data to Include Only Common IDs**
    mask_V = np.isin(star_ids_V, common_ids)
    mask_B = np.isin(star_ids_B, common_ids)
    
    #print(star_ids_V[1])
    #print(star_ids_B[1])
    
    #print(mask_V)                                              
    #print(mask_B)

    V_cal = V_cal[mask_V]
    V_inst = V_inst[mask_V]
    V_err = V_err[mask_V]
    star_ids_V = star_ids_V[mask_V]  # Only common IDs
    
    pmra_V = pmra_V[mask_V]
    pmdec_V = pmdec_V[mask_V]
    
    ra_V=ra_V[mask_V]
    dec_V=dec_V[mask_V]
    
    

    B_cal = B_cal[mask_B]
    B_inst = B_inst[mask_B]
    B_err = B_err[mask_B]
    star_ids_B = star_ids_B[mask_B]  # Only common IDs
    
    pmra_B = pmra_B[mask_B]
    pmdec_B = pmdec_B[mask_B]
    
    
    #n=np.random_integers(0,len(star_ids_V))
    #print(star_ids_V[0])
    #print(star_ids_B[0])

    # **Remove Non-Finite Values**
    finite_mask = np.isfinite(V_cal) & np.isfinite(V_inst) & np.isfinite(V_err) & np.isfinite(pmra_V) & np.isfinite(pmdec_V) & np.isfinite(ra_V) & np.isfinite(dec_V) & \
                  np.isfinite(B_cal) & np.isfinite(B_inst) & np.isfinite(B_err) & np.isfinite(pmra_B) & np.isfinite(pmdec_B)
                  
 

    V_cal = V_cal[finite_mask]
    V_inst = V_inst[finite_mask]
    V_err = V_err[finite_mask]
    B_cal = B_cal[finite_mask]
    B_inst = B_inst[finite_mask]
    B_err = B_err[finite_mask]
    star_ids_V = star_ids_V[finite_mask]  # Filter IDs too
    star_ids_B = star_ids_B[finite_mask] 
    
    pmra_V = pmra_V[finite_mask]
    pmdec_V = pmdec_V[finite_mask]
    
    ra_V=ra_V[finite_mask]
    dec_V=dec_V[finite_mask]
 
    
    V_stars = np.array(list(zip(star_ids_V,V_cal,V_inst,V_err, pmra_V, pmdec_V, ra_V, dec_V)))   
    B_stars = np.array(list(zip(star_ids_B,B_cal,B_inst,B_err)))
    
    V_stars = V_stars[np.argsort(V_stars[:, 0])]
    B_stars = B_stars[np.argsort(B_stars[:, 0])]
    
    #print('V stars', V_stars[14])
    #print('B stars', B_stars[14])
   
    '''------------------------------------------------------------------------------------------'''
    
    V_cal = V_stars[:,1].astype(float)
    B_cal = B_stars[:,1].astype(float)
    
    V_inst = V_stars[:,2].astype(float)
    B_inst = B_stars[:,2].astype(float)
    
    V_error = V_stars[:,3].astype(float)
    B_error = B_stars[:,3].astype(float)
    
    pmra_final = V_stars[:, 4].astype(float)
    pmdec_final = V_stars[:, 5].astype(float)
    
    ra_final = V_stars[:,6].astype(float)
    dec_final = V_stars[:,7].astype(float)
    
    #print(ra_final)
    #print(dec_final)
    
    

    # Debugging
    #print(f"Common Stars Used: {len(common_ids)}")
    #print(f"Final Stars After Filtering Non-Finite Values: {len(V_cal)}")

    # Ensure lengths match
    assert len(V_cal) == len(B_cal), "Length mismatch after filtering!"

    # Proceed with calibration using only matched stars
    #print(f"Filtered to {len(V_cal)} stars with valid detections in both images.")
        
    def BV_matrix(B_cal, V_cal, alpha, beta, b0, v0):  
        a = np.array([[1-alpha, alpha], [beta, 1-beta]])
        y = np.vstack([B_cal-b0, V_cal-v0])
        a_inv = np.linalg.inv(a) #1/det * inverted matrix elements
        x = np.matmul(a_inv,y)
        B = x[0]
        V = x[1]        
        #print('B term', B)
        #print('V term', V)
        return B, V


    def chi_squared(parin):  
        alpha = parin[0]
        beta = parin[1]
        b0 = parin[2]
        v0 = parin[3]
        b_terms = []
        v_terms = []    
        N = len(B_inst)
        
        for j in range(N):
            B, V = BV_matrix(B_cal[j], V_cal[j], alpha, beta, b0, v0) #solve matrix
            b_terms.append(((B_inst[j]-B)**2)/((B_error[j])**2))
            v_terms.append(((V_inst[j]-V)**2)/((V_error[j])**2))
        
        B_term = np.sum(b_terms) #sum all ith terms
        V_term = np.sum(v_terms)
        chi2 = B_term + V_term #final chi^2
        
        print(chi2)
        print(2*N-4)
        return chi2
    
    
    alpha = -0.6346 
    beta = 0.1338 
    b0 = 18.3520 
    v0 = 19.9522 
            
    parin = np.array([alpha, beta, b0, v0])
    parname = ['alpha','beta','b0','v0']
    parstep = np.array([0.1, 0.1, 1., 1.]) #GUESSED BY ORDER OF MAG.
    parfix  = [True, True, True, True]  
    parlim = [(-10, 10), (-10, 10), (None, None), (None, None)] 
    m = Minuit(chi_squared, parin, name=parname)
        
    m.errors = parstep
    m.limits = parlim
    m.fixed = parfix
    m.migrad()                                        #minimised chi2
    MLE = m.values                                    #max-likelihood estimates
    sigmaMLE = m.errors                               #standard deviations
    cov = m.covariance                                #covariance matrix
    rho = m.covariance.correlation()
    
    print(f"Maximum Likelihood Estimates: {MLE} \n")
    print(f"Parameter Errors: {sigmaMLE} \n")
    print(f"Covariance Matrix: {cov}")
    print(f"Correlation Matrix: {rho}")
        
    # Use parameters to calculate calibrated magnitudes.
    Alpha = MLE[0]
    Beta = MLE[1]
    b_zero = MLE[2]
    v_zero = MLE[3]
    zero_point = np.vstack([b_zero, v_zero])
    #zero_point = np.array([[b_zero], [v_zero]])
    
    B_calibrated = []
    V_calibrated = []
    B_min_V_inst = []
    B_min_V_cal = []
    B_min_V_SIM = []
    B_min_V_errors = []
    V_cal_errors = []

    
    #V_inst_list = np.array(x['Inst'])
    #B_inst_list = np.array(y['Inst'])
    '''
    for b_i,v_i in enumerate(V_inst):
        c = np.array([[1-Alpha, Alpha], [Beta, 1-Beta]])
        d = np.vstack([B_inst[b_i], v_i])
        z = np.matmul(c,d)+zero_point
        #solves for b, v as in notes eq.1
        b_cal = z[0] 
        v_cal = z[1] 
        B_calibrated.append(b_cal) 
        V_calibrated.append(v_cal)         
        #create calibrated B-V array for CMD
        min_item = B_inst[b_i]-v_i
        B_min_V_inst.append(min_item)
        min_cal_item = ((1-Alpha-Beta)*(min_item))+(b_zero-v_zero)
        B_min_V_cal.append(min_cal_item)        
    print(B_min_V_cal)
    
    '''
    
    print(Alpha)
    for b_i,v_i in enumerate(V_inst): #find calibrated BVs based on the inst. mags and 
                                  #the parameters we've estimated
                                  
                            
        c = np.array([[1-Alpha, Alpha], [Beta, 1-Beta]])
        d = np.vstack([B_inst[b_i], v_i])
        z = np.matmul(c,d)+zero_point
        #solves for b, v as in notes eq.1
        b_cal = z[0] 
        v_cal = z[1]
        B_calibrated.append(b_cal)
        V_calibrated.append(v_cal)
        
        #find SIMBAD b-v values
        sim_b_min_v = B_cal[b_i]-V_cal[b_i]
        B_min_V_SIM.append(sim_b_min_v)
        
        #create calibrated b-v array for CMD
        min_item = B_inst[b_i]-v_i
        B_min_V_inst.append(min_item)
        min_cal_item = ((1-Alpha-Beta)*(min_item))+(b_zero-v_zero)
        B_min_V_cal.append(min_cal_item)
        
        #find errors on each v_cal point
        sigma_v_cal = ((Beta**2)*B_error[b_i]**2)+(((1-Beta)**2)*V_error[b_i]**2)
        V_cal_errors.append(np.sqrt(sigma_v_cal))
        
        #find errors on each B-V point
        #b_i is just the index so can be used for both here
        B_min_V_error = np.sqrt(B_error[b_i]**2 + V_error[b_i]**2)    
        sigma_b_min_v = (1-Alpha-Beta)*B_min_V_error
        B_min_V_errors.append(sigma_b_min_v)

    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # --- First Plot: V_inst vs V_calibrated ---
    ax1.scatter(V_inst, V_calibrated, s=12, color='black', label="Stars")  # Black points
    A1, B1 = np.polyfit(V_inst, V_calibrated, 1)  # Linear fit
    best_fit_line = (A1 * np.array(V_inst)) + B1
    ax1.plot(V_inst, best_fit_line, color="red", linewidth=2, label="Best-Fit Line", alpha=0.5, )  # Red line
    
    # Labels and Formatting
    ax1.set_xlabel(r'$V_{inst}$', fontsize=16)
    ax1.set_ylabel(r'$v$', fontsize=16)
    #ax1.grid(True)
    ax1.legend(fontsize=14)
    
    # --- Second Plot: B_inst vs B_calibrated ---
    ax2.scatter(B_inst, B_calibrated, s=12, color='black')  # Black points
    C1, D1 = np.polyfit(B_inst, B_calibrated, 1)  # Linear fit
    best_fit_line = (C1 * np.array(B_inst)) + D1
    ax2.plot(B_inst, best_fit_line, color="red", linewidth=2, alpha=0.5)  # Red line
    
    # Labels and Formatting
    ax2.set_xlabel(r'$B_{inst}$', fontsize=16)
    ax2.set_ylabel(r'$b$', fontsize=16)
    #ax2.grid(True)
    ax2.legend()
    
    # --- Third Plot: B-V Color Index ---
    ax3.scatter(B_min_V_SIM, B_min_V_inst, s=12, color='black')  # Black points
    xmin, xmax = min(B_min_V_SIM), max(B_min_V_SIM)
    x_points = np.linspace(xmin, xmax, 100)
    y_plot = (x_points - (b_zero - v_zero)) / (1 - Alpha - Beta)
    ax3.plot(x_points, y_plot, color="red", linewidth=2, alpha=0.5)  # Red line
    
    # Labels and Formatting
    ax3.set_xlabel(r'$b-v$', fontsize=16)
    ax3.set_ylabel(r'$(B-V)_{inst}$', fontsize=16)
    #ax3.set_title('B-V Colour Index', fontsize=14)
    #ax3.grid(True)
    ax3.legend()
    
    # Adjust spacing between plots
    plt.tight_layout()
    
    # Show all three plots
    plt.show()

   
    
    
    '''
    # Third Plot: (B-V)_inst vs (B-V)_cal
    figB, ax3 = plt.subplots(1, 1, figsize=(5, 6))
    ax3.scatter(B_min_V_inst, B_min_V_cal, s=12)
    ax3.set_xlabel(r'$(B-V)_{inst}$')
    ax3.set_ylabel(r'$(B-V)_{cal}$')
    ax3.set_title('Calibrated B-V Colour Index')
    ax3.grid(True)  # Add grid lines
    # Label points for B-V plot (assuming star_ids_V are used here too)
    #for i, star_id in enumerate(star_ids_V):
        #ax3.annotate(star_id, (B_min_V_inst[i], B_min_V_cal[i]), textcoords="offset points", xytext=(5, 5), ha='right', fontsize=8)
    
    plt.show()'''
      
    
    figB, ax4 = plt.subplots(1, 1, figsize=(5, 6))
    ax4.scatter(B_min_V_cal,V_calibrated, s=12, color='black', label='Instrumental')
    ax4.scatter(B_cal-V_cal, V_cal, s=12, color='red', label='SIMBAD')
    ax4.errorbar(B_min_V_cal, V_calibrated, yerr=V_cal_errors, xerr=B_min_V_errors, fmt='o', color='b', ecolor='black', mec='black')
    ax4.set_xlabel(r'$(B-V)$')
    ax4.set_ylabel(r'$V$')
    #ax4.set_title('B-V vs V')
    ax4.grid(True)  # Add grid lines
    ax4.invert_yaxis()
    #ax4.legend(fontsize=16)
    #for i, star_id in enumerate(V_stars[:,0]):
    #    ax4.annotate(star_id, ( B_min_V_cal[i], V_calibrated[i]), textcoords="offset points", xytext=(5, 5), ha='right', fontsize=8)
    plt.show()
    
    
    
    
    
    '''Membership Statistic'''
 
    # Compute median (to mitigate effects of outliers) and covariance of cluster's proper motion
    median_pm = np.array([np.median(pmra_final), np.median(pmdec_final)])  
    mean_pm = np.array([np.mean(pmra_final), np.mean(pmdec_final)])  
    cov_pm = np.cov(np.vstack([pmra_final, pmdec_final]))
    
    mean_pos = np.array([np.mean(ra_final), np.mean(dec_final)])  
    cov_pos = np.cov(np.vstack([ra_final, dec_final]))      
    

    mvn_pm = multivariate_normal(mean=mean_pm, cov=cov_pm) #Gaussian distribution for proper motion
    mvn_pos = multivariate_normal(mean=mean_pos, cov=cov_pos) #Gaussian distribution for positions
    
    Uniform_pm = 1/((np.max(pmra_final)-np.min(pmra_final))*(np.max(pmdec_final)-np.min(pmdec_final)))
    Uniform_pos = 1/((np.max(ra_final)-np.min(ra_final))*(np.max(dec_final)-np.min(dec_final))) 
    
    #print('Uniform_pm', Uniform_pm)
    #print('Uniform_pos', Uniform_pos)
    
    
    
    
    # Compute probability density for each star’s proper motion
    pm_values = np.column_stack((pmra_final, pmdec_final))
    pos_values = np.column_stack((ra_final, dec_final))
    P_pm_cluster = mvn_pm.pdf(pm_values)
    P_pos_cluster = mvn_pos.pdf(pos_values)
    
    # Normalize probabilities
    membership_probs =  (P_pm_cluster*P_pos_cluster)/((P_pm_cluster*P_pos_cluster)+(Uniform_pm*Uniform_pos))
    
    #print('statistic', membership_probs)
    

    
  
    plt.figure(figsize=(9, 7))
    
    # Separate stars based on membership probability threshold (0.8 seems to be a common value from literature)
    P=0.9
    high_prob = membership_probs > P
    low_prob = ~high_prob
    
    
    plt.scatter(pmra_final[high_prob], pmdec_final[high_prob], 
                color='red', marker='+', s=50, label='Cluster Members (p > 0.9)', alpha=0.8)
    
 
    plt.scatter(pmra_final[low_prob], pmdec_final[low_prob], 
                color='black', marker='o', s=20, alpha=0.5, label='Field Stars (p ≤ 0.9)')
    
    
    #Density contours for reference
    sns.kdeplot(x=pmra_final, y=pmdec_final, levels=20, 
               linewidths=1, color="gray", alpha=0.5, fill=True)
      
    #plt.axvline(median_pm[0], linestyle="--", color="gray", alpha=0.6)
    #plt.axhline(median_pm[1], linestyle="--", color="gray", alpha=0.6)
    #plt.title("Vector Point Diagram (VPD) with Membership Probabilities", fontsize=14)
    plt.xlabel(r'$\mu_\alpha \cos(\delta)$ [mas/yr]', fontsize=16)
    plt.ylabel(r'$\mu_\delta$ [mas/yr]', fontsize=16)
    plt.legend(frameon=True, loc='upper left', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.grid(True, linestyle='--', alpha=0.6)   
    plt.show()
    
    
    plt.figure(figsize=(9, 7))
    
    # Separate stars based on membership probability threshold (0.8 seems to be a common value from literature)
   
    
    plt.scatter(ra_final[high_prob], dec_final[high_prob], 
                color='red', marker='+', s=50, label='Cluster Members (p > 0.9)', alpha=0.8)
    
 
    plt.scatter(ra_final[low_prob], dec_final[low_prob], 
                color='black', marker='o', s=20, alpha=0.5, label='Field Stars (p ≤ 0.9)')
    
    
    #Density contours for reference
    sns.kdeplot(x=ra_final, y=dec_final, levels=20, 
               linewidths=1, color="gray", alpha=0.5, fill=True)
      
    
    #plt.title("Vector Point Diagram (VPD) with Membership Probabilities", fontsize=14)
    plt.xlabel(r'$RA$ [deg]', fontsize=16)
    plt.ylabel(r'$DEC$ [deg]', fontsize=16)
    plt.legend(frameon=True, loc='upper left', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.grid(True, linestyle='--', alpha=0.6)   
    plt.show()
    
    
    member=high_prob
    pmra_member=pmra_final[member]
    pmdec_member=pmdec_final[member]
    ra_member=ra_final[member]
    dec_member=dec_final[member]
    mean_pm = np.array([np.mean(pmra_member), np.mean(pmdec_member)])  
    cov_pm = np.cov(np.vstack([pmra_member, pmdec_member]))
    mean_pos=np.array([np.mean(ra_member), np.mean(dec_member)])
    cov_pos=np.cov(np.vstack([ra_member, dec_member]))
    print('mean_pm',mean_pm)
    print('cov_pm',cov_pm)
    print('mean_pos', mean_pos)
    print('cov_pos', cov_pos)
    
    
    
        
    #Membership Probability Histogram
    def plot_membership_histogram(membership_probs):
        """
        Plot a histogram of membership probabilities in steps of 0.1.
        """
        bins = np.arange(0.8, 1.0, 0.005)  
        
        plt.figure(figsize=(9, 6))
        plt.hist(membership_probs, bins=bins, color='black', edgecolor='black', label='Data')        
        #cdf = np.cumsum(np.histogram(membership_probs, bins=bins)[0]) / len(membership_probs)
        #plt.plot(bins[:-1] + 0.05, color='red', alpha=0.5)
        plt.axvline(0.9, color='red', linestyle='--', alpha=0.5)

        #mean_prob = np.mean(membership_probs)
        #median_prob = np.median(membership_probs)
        #plt.axvline(mean_prob, color='green', linestyle='dotted', label=f'Mean: {mean_prob:.2f}')
        #plt.axvline(median_prob, color='purple', linestyle='dotted', label=f'Median: {median_prob:.2f}')
        plt.xlabel("Membership Statistic", fontsize=16)
        plt.ylabel("Number of Stars", fontsize=16)
        #plt.title("Histogram of Membership Probabilities", fontsize=14)
        
        ticks=np.arange(0.8,1.0,0.025)
        plt.xticks(ticks, fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='y', alpha=0.6)
        #plt.legend(loc='upper left')
        plt.show()    
    plot_membership_histogram(membership_probs)
    membership_data = np.column_stack((star_ids_V, pmra_final, pmdec_final, membership_probs))
    
    '''
         # Scatter Plot for Spatial Location
    figB, ax5 = plt.subplots(figsize=(5, 6))
    #ax5.scatter(ra_final, dec_final, s=12, color='black', label='Stars')
    sns.kdeplot(x=ra_final, y=dec_final, levels=20, linewidths=1, color="gray", fill=True, alpha=0.5)    
    ax5.set_xlabel(r'$RA$')
    ax5.set_ylabel(r'$DEC$')
    #ax5.set_title('')
    ax5.grid(True, linestyle="--", alpha=0.6)  
    ax5.invert_yaxis()    
    # Compute Mean & Covariance
    mean_pos = np.array([np.median(ra_final), np.median(dec_final)])  # Using Median
    cov_pos = np.cov(np.vstack([ra_final, dec_final]))  # Ensure correct shape    
    # Multivariate Normal Distribution for Cluster Proper Motion
    mvn = multivariate_normal(mean=mean_pos, cov=cov_pos)    
    # Compute Probability Density for Each Star’s Proper Motion
    pos_values = np.column_stack((ra_final, dec_final))
    membership_probs_spatial = mvn.pdf(pos_values)    
    # Normalize Probabilities
    membership_probs_spatial = membership_probs_spatial / np.max(membership_probs_spatial)    
    # Membership Classification
    P = 0.8
    high_prob = membership_probs_spatial > P
    low_prob = ~high_prob
    
    ax5.scatter(ra_final[high_prob], dec_final[high_prob], 
                color='red', marker='+', s=50, label='Cluster Members (p > 0.8)', alpha=0.8)    
    # Field Stars
    ax5.scatter(ra_final[low_prob], dec_final[low_prob], 
                color='black', marker='o', s=20, alpha=0.5, label='Field Stars (p ≤ 0.8)')
    
    # Marking Mean (or Median) Proper Motion
    #2ax5.scatter(mean_pos[0], mean_pos[1], color='gold', marker='*', s=150, label='Median Proper Motion')
    
    # Grid and Layout Enhancements
    ax5.axvline(mean_pos[0], linestyle="--", color="gray", alpha=0.6)
    ax5.axhline(mean_pos[1], linestyle="--", color="gray", alpha=0.6)
    #ax5.title("Spatial Location with Membership Probabilities", fontsize=14)
    ax5.set_xlabel('RA[deg]', fontsize=16)
    ax5.set_ylabel('DEC[deg]', fontsize=16)
    #ax5.set_legend(frameon=True, loc='upper left', fontsize=11)
    #ax5.grid(True, linestyle='--', alpha=0.6)   
    plt.show()'''
    
 
       

    def plot_member_cmd(B_min_V_cal, V_calibrated, membership_probs, B_min_V_errors, V_cal_errors):
        """
        Plots a CMD for only the stars that are considered members.
    
        """
        # Convert lists to numpy arrays for filtering
        B_min_V_cal = np.array(B_min_V_cal)
        V_calibrated = np.array(V_calibrated)
        membership_probs = np.array(membership_probs)
        B_V_errors = np.array(B_min_V_errors)
        V_errors = np.array(V_cal_errors)
        
     


        
        # Apply membership probability filter (strict inclusion)
        member_mask = high_prob  # Only select members above threshold
        B_min_V_members = B_min_V_cal[member_mask]
        V_members = V_calibrated[member_mask]
        prob_members = membership_probs[member_mask]
        
        B_V_errors_members = B_V_errors[member_mask]
        V_errors_members = V_errors[member_mask]
    
        # Plot CMD with members only
        plt.figure(figsize=(6, 8))
    
        sc = plt.scatter(
            B_min_V_members, V_members,
            s=20,
            edgecolor='black', color='blue'
        )
        
        # Add error bars (keep them neutral in color)
        plt.errorbar(B_min_V_members, V_members, yerr=V_errors_members, xerr=B_V_errors_members, fmt='none', ecolor='black')
        
        # Labels and formatting
        plt.xlabel(r'$(B-V)_{inst}$', fontsize=16)
        plt.ylabel(r'$V_{inst}$', fontsize=16)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().invert_yaxis()
        
        # Add colorbar
        #plt.colorbar(sc, label='Membership Probability')
            
        
       
        # --- Isochrone Overlay Section ---
        file2 = read_mist_models.ISOCMD('MIST_iso_67b0b7100ff8f.iso.cmd')
        isocmd = file2
        
        
        # Define age range for the isochrones
        start_age = 8.2
        end_age = 8.2
        step = 0.1
        ages = np.arange(start_age, end_age + step, step)
        
        # Loop over each age and overlay the corresponding isochrone
        for age in ages:
            age_ind = isocmd.age_index(age)
            B = isocmd.isocmds[age_ind]['Bessell_B']  # Isochrone B magnitudes
            V = isocmd.isocmds[age_ind]['Bessell_V']  # Isochrone V magnitudes
            BminV = np.array(B - V)                   # Compute (B-V)
            
            # Apply extinction and distance modulus adjustments:
            Rv = 3.1
            Av = 0.98     #2.46
            colour_excess = Av / Rv       # Horizontal (color) shift
            mu = 9.6            # Vertical shift (distance modulus) 8.5
            
            V_adjusted = np.array(V) + mu + Av
            B_V_adjusted = np.array(BminV) + colour_excess
            
            # Filter isochrone points to the relevant range:
            B_V_plot = []
            V_plot = []
            for j, mag in enumerate(B_V_adjusted):
                if 0.1< mag < 2.25 and 6 < V_adjusted[j] < 15:
                    B_V_plot.append(mag)
                    V_plot.append(V_adjusted[j])
            
            distance = 10**((mu+5)/5)
            print('distance[pc] = ', distance)
            print('Av', Av)
            print('Excess', colour_excess)
            
            # Plot the isochrone with a label indicating its age.
            plt.plot(B_V_plot, V_plot, lw=1.5, label=f'log(age) = {age:.1f}', alpha=0.5)
        
        # Restore the original axis limits (so the scatter data remains centered)
       
        
        # Add a legend to distinguish the isochrones
        plt.legend(loc='center right', fontsize=16)   
        plt.show()
        
        V_members = np.ravel(V_members)
        
        def V_band_Histogram(V_members):
            bins = np.arange(6, 18, 0.2)          
            plt.figure(figsize=(9, 6))
            plt.hist(V_members, bins=bins, color='black', edgecolor='black', label='Data')
            plt.xlabel(r'$(V)_{inst}$', fontsize=16)
            plt.ylabel("Number of Stars", fontsize=16)
            #plt.title("Histogram of $(V)_{cal}$", fontsize=14)
            plt.xticks(np.arange(6, 19, 1), fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(axis='y', alpha=0.6)
            #plt.legend()
            plt.show()
        V_band_Histogram(V_calibrated)
        
        
        def V_mag_Members(V_members, membership_probs):
            bins = np.arange(6, 18, 0.2)          
            plt.figure(figsize=(9, 6))
            plt.scatter(V_members, membership_probs, color='black', edgecolor='black', label='Data')
            plt.xlabel(r'$(V)_{inst}$', fontsize=16)
            plt.ylabel("Membership Probability", fontsize=16)
            #plt.title("Histogram of $(V)_{cal}$", fontsize=14)
            plt.xticks(np.arange(6, 19, 1), fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(axis='y', alpha=0.6)
            #plt.legend()
            plt.show()
          
            
        V_mag_Members(V_calibrated, membership_probs)
        
        
    plot_member_cmd(B_min_V_cal, V_calibrated, membership_probs, B_min_V_errors, V_cal_errors)
    

        
Calibration()