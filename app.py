import numpy as np 
import pandas as pd
from scipy.linalg import lstsq
import astropy.units as u 
from astropy.time import Time
from astropy.constants import G,M_sun
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    SkyCoord,
    GCRS,
    get_body_barycentric_posvel,
)

"""
A simple script to determine the orbit of (1)Ceres
using the Herrick-Gibbs method. 
"""
mu_sun_si = (G*M_sun).to(u.m**3/u.s**2).value #Standard gravitational parameter of the Sun in SI
wls_w_list = np.array([0.5,0.5,0.5,0.5])
obs_coords = [(-29.16163056,-67.49933361,1107), #Chilecito,Argentina
            (28.07616944, -16.558,207),#San Isidro,Canary Islands
            (64.13548333,-21.89546667,37),#Reykjavik,Iceland
            (33.96095,-83.37793611,232)]#Athens,Georgia,USA

epoch = Time([ "2000-01-01 00:00:00" , "2000-01-01 00:00:00" , "2000-01-01 00:00:00" ], scale = "tt") #J2000

def loc_eci(lat,lon,alt,t):  #Obtaining Earth-Centered Inertial coordinates of the observer on Earth at a given time
    Loc = EarthLocation.from_geodetic(lon*u.deg,lat*u.deg,alt*u.m)
    gcrs = Loc.get_gcrs(obstime=t)
    return gcrs.cartesian.xyz.to(u.m).value

def loc_unit_vec_radec(ra_hours, dec_deg, t, equinox="J2000"):
    if equinox == "obs":
        sc = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_deg * u.deg, frame="fk5", equinox=t)
    else:
        sc = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_deg * u.deg, frame="fk5", equinox="J2000")
    sc_gcrs = sc.transform_to(GCRS(obstime=t))
    coords = sc_gcrs.cartesian
    v = np.array([coords.x.value, coords.y.value, coords.z.value])
    return v / np.linalg.norm(v)

def loc_unit_vec_altaz(az, alt, lat, lon, alt_m, t):
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, alt_m * u.m)
    altaz = AltAz(obstime=t, location=loc)
    sc = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame=altaz)
    sc_gcrs = sc.transform_to(GCRS(obstime=t))
    coords = sc_gcrs.cartesian
    v = np.array([coords.x.value, coords.y.value, coords.z.value])
    return v / np.linalg.norm(v)

def geocentric_ceres(R_list,rho_list,wls_w_list): # Solve for Geocentric position of Ceres using weighted least squares
    A_blocks, b_blocks, w_blocks = [], [], []
    sigma_rad = np.deg2rad(wls_w_list / 3600.0)
    for i in range(len(R_list)):
        rho = rho_list[i].reshape(3, 1)
        P_i = np.eye(3) - rho @ rho.T
        R_i = R_list[i]
        w_i = 1.0 / sigma_rad[i]**2
        A_blocks.append(P_i)
        b_blocks.append(P_i @ R_i)
        w_blocks.extend([w_i, w_i, w_i])

    A = np.vstack(A_blocks)
    b = np.concatenate(b_blocks)

    W_sqrt = np.sqrt(w_blocks)
    A_weighted = A * W_sqrt[:, None]
    b_weighted = b * W_sqrt

    r_geo, residuals, rank, s = lstsq(A_weighted, b_weighted, cond=None)
    cov_r = np.linalg.inv(A_weighted.T @ A_weighted)
    return r_geo, cov_r

def pairwise_geocentric_solution(R_list, rho_list, min_range_au=1.0, max_range_au=5.0, max_gap_km=1000.0):
    midpoints = []
    for i in range(len(R_list)):
        for j in range(i + 1, len(R_list)):
            R1, R2 = R_list[i], R_list[j]
            rho1, rho2 = rho_list[i], rho_list[j]
            A = np.vstack([rho1, -rho2]).T
            s, t = np.linalg.lstsq(A, R2 - R1, rcond=None)[0]
            p1 = R1 + s * rho1
            p2 = R2 + t * rho2
            midpoint = 0.5 * (p1 + p2)
            gap = np.linalg.norm(p1 - p2)
            au = np.linalg.norm(midpoint) / 1.495978707e11
            if gap < max_gap_km * 1000.0 and min_range_au < au < max_range_au:
                midpoints.append((midpoint, gap))

    if len(midpoints) >= 2:
        mids = np.array([m for m, _ in midpoints])
        weights = np.array([1.0 / (gap + 1e-6) for _, gap in midpoints])
        return np.average(mids, axis=0, weights=weights)

    r_geo, _ = geocentric_ceres(R_list, rho_list, wls_w_list)
    return r_geo

def geo_to_helio(r_geo,t): #Converts Ceres' Geocentric position to Heliocentric position.
    E_bary = get_body_barycentric_posvel("earth", Time(t))
    sun_bary = get_body_barycentric_posvel("sun", Time(t))
    r_helio = r_geo + E_bary[0].xyz.to(u.m).value - sun_bary[0].xyz.to(u.m).value
    return r_helio 

def h_gibbs(r1, r2, r3, t1, t2, t3):
    dt12 = (t2 - t1).to(u.s).value   
    dt23 = (t3 - t2).to(u.s).value
    dt13 = (t3 - t1).to(u.s).value
    r1m, r2m, r3m = np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)
    D1 = -dt23 * (1.0 / (dt12 * dt13) + mu_sun_si / (12.0 * r1m**3))
    D2 = (dt23 - dt12) * (1.0 / (dt12 * dt23) + mu_sun_si / (12.0 * r2m**3))
    D3 =  dt12 * (1.0 / (dt23 * dt13) + mu_sun_si / (12.0 * r3m**3))
    
    # Check: D1 + D2 + D3 ≈ 0, allowing a small numeric residual from near-degenerate geometry
    assert np.isclose(D1 + D2 + D3, 0.0, rtol=1e-2, atol=1e-2), \
        f"Herrick-Gibbs coefficient sum check failed: {D1+D2+D3:.3e}"
    v2 = D1 * r1 + D2 * r2 + D3 * r3
    return v2

def keplerian_elements(r2, v2, mu=mu_sun_si):
    r = np.linalg.norm(r2)
    v = np.linalg.norm(v2)
    
    h_vec = np.cross(r2, v2)           # specific angular momentum
    h     = np.linalg.norm(h_vec)
    if h==0:
        raise ValueError('Invalid value for angular momentum')
    
    N_vec = np.cross([0, 0, 1], h_vec)  #Ascending Node vector
    N     = np.linalg.norm(N_vec)
    
    e_vec = ((v**2 - mu/r) * r2 - np.dot(r2, v2) * v2) / mu
    e     = np.linalg.norm(e_vec)
    
    eps   = v**2 / 2.0 - mu / r  #Specific energy from the vis-viva equation
    if eps >0:
        raise ValueError(f'Hyperbolic Orbit: eps={eps:.3e} m^2/s^2')
    elif abs(eps) <=0:
        raise ValueError(f'Parabolic Orbit: eps={eps:.3e} m^2/s^2') 
    a = -mu / (2.0 * eps)
#Inclination 
    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))
#Right Ascension of Ascending Node (RAAN)
    Omega = np.degrees(np.arccos(np.clip(N_vec[0] / N, -1, 1)))
    if N_vec[1] < 0:
        Omega = 360.0 - Omega
#Argument of Perihelion.
    if e < 1e-10:
        omega = 0.0 #Circular orbit. 
    else:
        omega = np.degrees(np.arccos(np.clip(np.dot(N_vec, e_vec) / (N * e), -1, 1)))
        if e_vec[2] < 0: omega = 360.0 - omega
#True Anomaly
    nu = np.degrees(np.arccos(np.clip(np.dot(e_vec, r2) / (e * r), -1, 1))) 

    if np.dot(r2, v2) < 0:
        nu = 360.0 - nu
    T_yr = 2 * np.pi * np.sqrt(a**3 / mu) / (365.25 * 86400)

    return a, e, i, Omega, omega, nu, T_yr

def keplerian_to_equinoctial(a,e,i_deg,Omega_deg,omega_deg,nu_deg):
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(np.radians(nu_deg / 2)))
    M = E - e * np.sin(E)
    
    e_x = e * np.cos(np.radians(Omega_deg + omega_deg)) #x_eccentricity
    e_y = e*np.sin(np.radians(Omega_deg + omega_deg)) #y_eccentricity
    h_x = np.tan(np.radians(i_deg/2))*np.cos(np.radians(Omega_deg)) #x_inclination
    h_y = np.tan(np.radians(i_deg/2))*np.sin(np.radians(Omega_deg)) #y_inclination
    L = np.radians(Omega_deg + omega_deg)+M #Mean Longitude in radians
    return a, e_x, e_y, h_x, h_y, L

def propagate_orbit(a, e_x, e_y, h_x, h_y, L,t,t_epoch,mu = mu_sun_si ):
    n = np.sqrt(mu / a**3)  # mean motion
    L_t = L + n*(t - t_epoch).to(u.s).value
    return a, e_x, e_y, h_x, h_y, L_t

def get_cartesian_at_time(equi, t_target, t_epoch):
    a, ex, ey, hx, hy, L0 = equi
    mu = mu_sun_si
    
    # 1. Propagation
    n = np.sqrt(mu / a**3)
    dt = (t_target - t_epoch).to(u.s).value
    Lt = L0 + n * dt
    
    # 2. Solve Kepler's Eq for F (Newton-Raphson)
    F = Lt
    for _ in range(10): # Usually converges in 5-10 iterations
        f_val = F + ey*np.cos(F) - ex*np.sin(F) - Lt
        f_prime = 1 - ey*np.sin(F) - ex*np.cos(F)
        F = F - f_val/f_prime
        
    # 3. Orbital Plane Coordinates
    beta = 1.0 / (1.0 + np.sqrt(1 - ex**2 - ey**2))
    Xw = a * ((1 - ey**2 * beta) * np.cos(F) + ex * ey * beta * np.sin(F) - ex)
    Yw = a * ((1 - ex**2 * beta) * np.sin(F) + ex * ey * beta * np.cos(F) - ey)
    
    # 4. Final 3D Cartesian
    denom = 1 + hx**2 + hy**2
    h_prod = hx * Xw - hy * Yw
    
    x = Xw - (2 * hx / denom) * h_prod
    y = Yw + (2 * hy / denom) * h_prod
    z = (2 / denom) * h_prod
    
    return np.array([x, y, z])

def get_geocentric_cartesian(equi, t_target, t_epoch):
    r_ceres_sun = get_cartesian_at_time(equi, t_target, t_epoch)
    earth_bary = get_body_barycentric_posvel("earth", t_target)
    sun_bary = get_body_barycentric_posvel("sun", t_target)
    
    r_earth_sun = (earth_bary[0].xyz - sun_bary[0].xyz).to(u.m).value
    r_geo = r_ceres_sun - r_earth_sun
    
    return r_geo

def load_and_parse_file(file_path):
    df_raw = pd.read_excel(file_path, header=1)
    set_indices = [(0, 4), (6, 10), (12, 16)]
    processed_sets = []

    for start, end in set_indices:
        subset = df_raw.iloc[start:end].copy()
        subset = subset.rename(columns={
            'Lati': 'Lat',
            'Longi': 'Lon',
            'Time': 'Time_Lcl',
            'UTC Time Zone': 'UTC_Off',
            'Elevation(m)': 'Alt_m',
        })
        subset = subset.dropna(subset=['Date', 'Time_Lcl', 'UTC_Off']).copy()

        times = []
        for _, row in subset.iterrows():
            t_base = Time(row['Date'], scale='utc')
            t_utc = t_base + (float(row['Time_Lcl']) - float(row['UTC_Off'])) * u.hour
            times.append(t_utc)
        subset['Time_UTC'] = times
        subset['RA_obs'] = subset['RA'].astype(float)
        subset['Dec_obs'] = subset['Dec'].astype(float)
        subset['RA_deg'] = subset['RA_J2000'].astype(float) * 15.0
        subset['Dec_deg'] = subset['Dec_J2000'].astype(float)
        processed_sets.append(subset)
    
    return processed_sets

def main(file_path):
    data_sets = load_and_parse_file(file_path)
    
    r_helio_list = []
    t_list = []

    print(f"Processing {len(data_sets)} observation sets...")

    for subset in data_sets:
        R_obs_set = []
        rho_hat_set = []
        
        # Triangulate using all 4 stations in the set
        for _, row in subset.iterrows():
            t = row['Time_UTC']
            R_obs_set.append(loc_eci(row['Lat'], row['Lon'], row['Alt_m'], t))
            rho_hat_set.append(loc_unit_vec_radec(row['RA_obs'], row['Dec_obs'], t, equinox="obs"))
        r_geo = pairwise_geocentric_solution(R_obs_set, rho_hat_set)
        
        t_avg = Time(np.mean([t.jd for t in subset['Time_UTC']]), format='jd')
        
        # Convert to Heliocentric
        r_helio = geo_to_helio(r_geo, t_avg)
        r_helio_list.append(r_helio)
        t_list.append(t_avg)

    # 2. Herrick-Gibbs to find velocity at middle point (t2)
    try:
        v2 = h_gibbs(r_helio_list[0], r_helio_list[1], r_helio_list[2], 
                    t_list[0], t_list[1], t_list[2])
        results = keplerian_elements(r_helio_list[1], v2)
    except ValueError as err:
        print(f"Herrick-Gibbs produced non-elliptical orbit: {err}")
        dt23 = (t_list[2] - t_list[1]).to(u.s).value
        v2 = (r_helio_list[2] - r_helio_list[1]) / dt23
        print("Using finite-difference velocity between sets 2 and 3 as fallback.")
        results = keplerian_elements(r_helio_list[1], v2)

    a, e, i, Omega, omega, nu, T_yr = results
    
    print("\n--- Determined Keplerian Elements ---")
    print(f"Semi-major axis: {a/1.495978707e11:.4f} AU")
    print(f"Eccentricity:    {e:.4f}")
    print(f"Period:          {T_yr:.2f} years")

    equi_elements = keplerian_to_equinoctial(a, e, i, Omega, omega, nu)
    print("\n--- Final Equinoctial Elements (Propagation-Ready) ---")
    names = ['a (m)', 'ex', 'ey', 'hx', 'hy', 'L (rad)']
    for name, val in zip(names, equi_elements):
        print(f"{name}: {val:.6e}")
    

    return equi_elements, t_list[1]

if __name__ == "__main__":
    file_path = r"C:\Users\USER\OneDrive\Desktop\(1)Ceres_Obv.xlsx"
    equi_elements, epoch = main(file_path)
    t_target = Time("2025-11-12 00:00:00")
    pos_geo = get_geocentric_cartesian(equi_elements, t_target, epoch)



