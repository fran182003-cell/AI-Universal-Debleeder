import numpy as np
import scipy.signal as sig
from scipy.special import sph_harm, spherical_jn, spherical_yn

from typing import Union, Tuple, Dict
ArrayLike = Union[float, np.ndarray]

def sph2cart(
    r: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike
) -> Tuple[ArrayLike,ArrayLike,ArrayLike]:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r
        Radial distance from the origin.
    theta
        Polar angle in radians.
    phi
        Azimuthal angle in radians.

    Returns
    -------
    x, y, z
        Cartesian coordinates.
    
    Notes
    -----
    The spherical coordinate system used here follows the physics convention:
    - ``theta`` is the polar angle measured from the **positive z-axis**:
        * ``theta = 0`` → positive z-axis
        * ``theta = π/2`` → xy-plane
        * ``theta = π`` → negative z-axis
    - ``phi`` is the azimuthal angle measured in the **xy-plane** from the positive x-axis:
        * ``phi = 0`` → positive x-axis
        * ``phi = π/2`` → positive y-axis
        * ``phi = -π/2`` → negative y-axis
    - ``theta ∈ [0, π]`` and ``phi ∈ [-π, π]``.
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    
    return x,y,z

def cart2sph(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike
) -> Tuple[ArrayLike,ArrayLike,ArrayLike]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x
        Cartesian x-coordinate(s).
    y
        Cartesian y-coordinate(s).
    z
        Cartesian z-coordinate(s).

    Returns
    -------
    r, theta, phi
        Spherical coordinates corresponding to the input:
        - ``r``: radial distance from the origin
        - ``theta``: polar angle in radians
        - ``phi``: azimuthal angle in radians
    
    Notes
    -----
    The spherical coordinate system follows the physics convention:
    - ``theta`` is the polar angle measured from the **positive z-axis**:
        * ``theta = 0`` → positive z-axis
        * ``theta = π/2`` → xy-plane
        * ``theta = π`` → negative z-axis
    - ``phi`` is the azimuthal angle measured in the **xy-plane** from the positive x-axis:
        * ``phi = 0`` → positive x-axis
        * ``phi = π/2`` → positive y-axis
        * ``phi = -π/2`` → negative y-axis
    - ``theta ∈ [0, π]`` and ``phi ∈ [-π, π]``.

    This function ensures numerical stability by avoiding division by zero
    when computing ``theta`` for points at the origin.
    """
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/np.maximum(r,np.finfo(float).eps))
    phi = np.arctan2(y,x)

    return r,theta,phi

def computeSH(
    theta: ArrayLike,
    phi: ArrayLike,
    order: int = 3
) -> Tuple[Dict[int, Dict[int, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Compute spherical harmonics up to a given order using physics-convention angles.

    Parameters
    ----------
    theta
        Polar angle(s) in radians, measured from the positive z-axis.
    phi
        Azimuthal angle(s) in radians, measured counter-clockwise in the xy-plane from the positive x-axis.
    order
        Maximum degree of spherical harmonics (default is 3).

    Returns
    -------
    sh : dict
        Nested dictionary where sh[n][m] contains the spherical harmonic of
        degree n and order m evaluated at the meshgrid of input angles.
    theta : ndarray
        2D array of polar angles after meshing with `phi`.
    phi : ndarray
        2D array of azimuthal angles after meshing with `theta`.

    Notes
    -----
    - The input angles follow the **physics convention** (theta = polar, phi = azimuthal).
    - The function internally calls `scipy.special.sph_harm`, which expects
      azimuthal angle first and polar angle second, so the arguments are
      swapped internally.
    - Keys satisfy ``0 <= n <= order`` and ``-n <= m <= n``.
    - The returned `theta` and `phi` arrays correspond to the 2D meshgrid
      used in evaluating the harmonics.

    Examples
    --------
    >>> th = np.linspace(0, np.pi, 5)
    >>> ph = np.linspace(0, 2*np.pi, 5)
    >>> sh, t, p = computeSH(th, ph, order=2)
    >>> sh[1][0]
    array([...], dtype=complex)
    """
    theta, phi = np.meshgrid(theta,phi)
    theta = theta.squeeze()
    phi = phi.squeeze()
    sh = {}
    for n in range(order+1):
        sh[n] = {}
        for m in range(-n,n+1,1):
            sh[n][m] = sph_harm(m,n,phi,theta)

    return sh, theta, phi

def SHMatrix(
    sampling_points: np.ndarray,
    order: int = 3
) -> np.ndarray:
    """
    Construct the spherical harmonics matrix up to a given order:

    Parameters
    ----------
    sampling_points
        2D array of shape (Q, 2), where Q is the number of sampling points.
        - First column: polar angles (theta), in radians, measured from the positive z-axis.
        - Second column: azimuthal angles (phi), in radians, measured counter-clockwise
          in the xy-plane from the positive x-axis.
    order
        Maximum degree of spherical harmonics (default is 3).

    Returns
    -------
    Y : ndarray
        Complex-valued array of shape (Q, (order + 1)^2). Each row corresponds to
        one sampling point, and each column corresponds to a spherical harmonic
        basis function ordered by increasing order n, then degree m.
    
        Specifically:

            [Y00(theta1,phi1)    Y1-1(theta1,phi1)   ...     YNN(theta1,phi1)]   
        Y = [Y00(theta2,phi2)    Y1-1(theta2,phi2)   ...     YNN(theta2,phi2)]
            [                            ...                                 ]
            [Y00(thetaQ,phiQ)    Y1-1(thetaQ,phiQ)   ...     YNN(thetaQ,phiQ)]
    
    Notes
    -----
    - The number of spherical harmonic basis functions up to order `n` is (n + 1)².
    - The ordering of columns follows the standard convention:
        (0,0), (1,-1), (1,0), (1,1), (2,-2), …, (order, order).
    - Input angles follow the physics convention: theta = polar, phi = azimuth.

    Examples
    --------
    >>> points = np.array([
    ...     [np.pi/2, 0.0],
    ...     [np.pi/2, np.pi/2]
    ... ])
    >>> Y = SHMatrix(points, order=1)
    >>> Y.shape
    (2, 4)
    """
    Y = np.zeros([len(sampling_points),(order + 1)**2])*(1+1j)

    for q,(theta, phi) in enumerate(sampling_points):
        # Evaluate the complex spherical harmonics at the given sampling point
        Yq,_,_ = computeSH(theta,phi,order=order)
        # Fill the corresponding row in the matrix
        for n in range(order+1):
            for m in range(-n,n+1,1):
                row_index = n*(n+1)+m
                Y[q,row_index] = Yq[n][m]

    return Y

def bessel_1(
    x: ArrayLike,
    order: int = 3
) -> Dict[int, np.ndarray]:
    """
    Compute spherical Bessel functions of the first kind up to a given order.

    Parameters
    ----------
    x
        Real argument(s). A small epsilon is added internally for numerical stability.
    order
        Maximum order of the spherical Bessel functions (default is 3).

    Returns
    -------
    jn : dict
        Dictionary where jn[n] contains the spherical Bessel function
        j_n(x) evaluated at the input for 0 ≤ n ≤ `order`.
    """
    x = x + np.finfo(float).eps
    jn = {}
    for n in range(order+1):
        jn[n] = spherical_jn(n,x)
    
    return jn

def bessel_2(
    x: ArrayLike,
    order: int = 3
) -> Dict[int, np.ndarray]:
    """
    Compute spherical Bessel functions of the second kind up to a given order.

    Parameters
    ----------
    x
        Real argument(s). A small epsilon is added internally for numerical stability.
    order
        Maximum order of the spherical Bessel functions (default is 3).

    Returns
    -------
    yn : dict
        Dictionary where yn[n] contains the spherical Bessel function
        y_n(x) evaluated at the input for 0 ≤ n ≤ `order`.
    """
    x = x + np.finfo(float).eps
    yn = {}
    for n in range(order+1):
        yn[n] = spherical_yn(n,x)
    
    return yn

def hankel_1(
    x: ArrayLike,
    order: int = 3
) -> Dict[int, np.ndarray]:
    """
    Compute spherical Hankel functions of the first kind up to a given order.

    Parameters
    ----------
    x
        Real argument(s).
    order
        Maximum order of the spherical Hankel functions (default is 3).

    Returns
    -------
    h1 : dict
        Dictionary where h1[n] contains the spherical Hankel function
        h_n^(1)(x) evaluated at the input for 0 ≤ n ≤ `order`.

    Notes
    -----
    The spherical Hankel function of the first kind is defined as:

        h_n^(1)(x) = j_n(x) + i y_n(x)

    where j_n and y_n are the spherical Bessel functions of the first
    and second kind, respectively and i = sqrt(-1).
    """
    jn = bessel_1(x,order=order)
    yn = bessel_2(x,order=order)
    h_1 = {}
    for n in range(order+1):
       h_1[n] = jn[n] + 1j*yn[n]
    
    return h_1

def hankel_2(
    x: ArrayLike,
    order: int = 3
) -> Dict[int, np.ndarray]:
    """
    Compute spherical Hankel functions of the second kind up to a given order.

    Parameters
    ----------
    x
        Real argument(s).
    order
        Maximum order of the spherical Hankel functions (default is 3).

    Returns
    -------
    h2 : dict
        Dictionary where h2[n] contains the spherical Hankel function
        h_n^(2)(x) evaluated at the input for 0 ≤ n ≤ `order`.

    Notes
    -----
    The spherical Hankel function of the second kind is defined as:

        h_n^(2)(x) = j_n(x) - i y_n(x)

    where j_n and y_n are the spherical Bessel functions of the first
    and second kind, respectively, and i = sqrt(-1).
    """
    jn = bessel_1(x,order=order)
    yn = bessel_2(x,order=order)
    h_2 = {}
    for n in range(order+1):
        h_2[n] = jn[n] - 1j*yn[n]
    
    return h_2

def b_function(
    k: ArrayLike,
    r: float,
    ra: float,
    order: int = 3,
    type: str = "rigid"
) -> Dict[int, np.ndarray]:
    """
    Compute the radial function b_n(kr) relating the coefficients of the
    spherical Fourier transform of the sound pressure on a spherical array
    to the plane wave density function.

    Parameters
    ----------
    k
        Wavenumber (2π / λ), where λ is the wavelength. Can be scalar or array.
    r
        Radial distance from the origin where the pressure is evaluated.
    ra
        Radius of the spherical array.
    order
        Maximum order of the spherical expansion (default is 3).
    type
        Boundary condition of the spherical array:
        - ``"rigid"`` : rigid sphere (default),
        - ``"open"`` : open sphere.

    Returns
    -------
    bn : dict
        Dictionary where bn[n] contains the radial function b_n(kr)
        evaluated at the input for 0 ≤ n ≤ `order`.

    Notes
    -----
    - For a rigid sphere, the radial function is given by:

        b_n(kr) = 4π i^n [j_n(k*r) - [j_n'(k*ra) / h_n^(2)'(k*ra)]*h_n^(2)(k*r)] ]

      where j_n, y_n are spherical Bessel functions of the first and second kind,
      and h_n^(2) is the spherical Hankel function of the second kind.

    - For an open sphere:

        b_n(k, r) = 4π i^n j_n(kr)

    References
    ----------
    Rafaely, B. *Fundamentals of Spherical Array Processing*. Springer, 2015. (see p. 51)
    """
    bn = {}
    jn = bessel_1(k*r,order=order)
    # Compute the function b for rigid spheres for orders up to `order`
    if type == 'rigid':
        hn_2 = hankel_2(k*r,order=order)
        for n in range(order+1):
            num = spherical_jn(n,(k*ra) + np.finfo(float).eps,derivative=True)
            dem = num -1j*spherical_yn(n,k*ra + np.finfo(float).eps,derivative=True)
            bn[n] = 4*np.pi*(1j)**n*(jn[n] - num/dem*hn_2[n])
    # Compute the function b for open spheres for orders up to `order`
    elif type == 'open':
        for n in range(order+1):
            bn[n] = 4*np.pi*(1j)**n*jn[n]
    return bn

def AmpModMatrix(
    f: np.ndarray,
    b: Dict[int, np.ndarray],
    max_gain_db: float = 40,
    dtype: np.dtype = np.complex128
) -> Tuple[np.ndarray,Dict[int,int]]:
    """
    Compute the amplitude modulation equalization matrix for plane wave decomposition.

    Parameters
    ----------
    f
        1D array of frequency values (Hz).
    b
        Dictionary of radial functions b_n(f), where each entry corresponds to
        a spherical harmonic order n. Typically computed using :func:`b_function`.
    max_gain_db
        Maximum allowed gain in decibels. Used to determine crossover frequencies
        beyond which equalization is applied (default is 40 dB).
    dtype
        Data type of the output matrix (default is ``np.complex128``).

    Returns
    -------
    W : ndarray
        Equalization matrix of shape ``((N+1)², len(f))``, where ``N`` is the
        maximum spherical harmonic order in `b`.
    crossover_indices : dict
        Dictionary mapping spherical harmonic order n to the index in `f`
        at which equalization starts being applied.

    Notes
    -----
    - For each order n, the algorithm finds the lowest frequency at which the
      radial function magnitude |b_n(f)| is above ``-max_gain_db`` dB. This
      frequency defines the crossover point.
    - From the crossover index onward, the multiplicative inverse of b_n(f) is
      applied to the corresponding rows of the equalization matrix.

    Examples
    --------
    >>> f = np.linspace(100, 5000, 100)
    >>> b = b_function(2*np.pi*f/343, r=0.1, ra=0.1, order=3, type="rigid")
    >>> W, crossover_indices = AmpModMatrix(f, b)
    """
    N = len(b) - 1
    # Preallocate memory for matrix W
    W = np.zeros([(N+1)**2,len(f)],dtype=dtype)
    crossover_indices = {}
    # Fill the matrix
    W[0,::] = 1/b[0]
    for n in range(1,N+1):
        crossover_order_n = f[20*np.log10(abs(b[n])) >= -max_gain_db].min()
        crossover_order_n_index = np.where(f == crossover_order_n)[0][0]
        # Row indices depend on the SH order
        row_indices = np.arange(n**2,(n+1)**2)
        W[row_indices,crossover_order_n_index::] = 1/b[n][crossover_order_n_index::]
        # Add the crossover indices to the list
        crossover_indices.update({n:crossover_order_n_index}) 

    return W,crossover_indices

def MaxDirectivityBeamformerCoeffMatrix(
    f: np.ndarray,
    crossover_indices: Dict[int,int]
) -> np.ndarray:
    """
    Compute the coefficient matrix for the maximum directivity beamformer.

    Parameters
    ----------
    f
        1D array of frequency values (Hz).
    crossover_indices
        Dictionary mapping spherical harmonic order n to the index in `f`
        where equalization (from :func:`AmpModMatrix`) starts.
    
    Returns
    -------
    D : ndarray
        Square diagonal matrix of shape ``(len(f), len(f))`` containing the
        beamformer coefficients.

    Notes
    -----
    - The diagonal entries are piecewise constant and depend on the SH order.
    - For order n, the coefficient is:

        D = 4π / ( (n+1)² )
     
    Examples
    --------
    >>> f = np.linspace(0, 5000, 100)
    >>> crossover_indices = {1: 20, 2: 50, 3: 75}
    >>> D = MaxDirectivityBeamformerCoeffMatrix(f, crossover_indices)
    >>> D.shape
    (100, 100)
    """
    diag = np.zeros(len(f))
    # Sort the crossover indices by order
    sorted_orders = sorted(crossover_indices.keys())

    # Assign values to diag array based on crossover indices
    start_idx = 0
    for order in sorted_orders:
        end_idx = crossover_indices[order]
        # `order` here corresponds to the *next upper order* (n+1).
        # This means that when computing the coefficients for n = 0,
        # `order` is 1, coefficient = 4π / (1+1)^2.
        # When computing for n = 1, `order` is 2, coefficient = 4π / (2+1)^2, etc.
        diag[start_idx:end_idx] = 4 * np.pi / (order**2)
        start_idx = end_idx  # Update start index for the next range

    # Assign the last range (beyond the last crossover point)
    diag[start_idx:] = 4 * np.pi / ((sorted_orders[-1] + 1)**2)
    
    # Return the diagonal matrix with the beamformer coefficients
    return np.diag(diag)

def MaxDirectivityBeamformer(
    x: np.ndarray,
    fs: int,
    ldirs: np.ndarray,
    sampling_points:np.ndarray,
    order: int = 3,
    r: float = 49e-3,
    win_len: int = 2048
) -> np.ndarray:
    """
    Implement a maximum directivity beamformer for spherical microphone arrays.

    Parameters
    ----------
    x
        Input microphone signals of shape ``(Q, L)``, where ``Q`` is the number of
        microphones (sampling points) and ``L`` is the signal length in samples.
    fs
        Sampling frequency in Hz.
    ldirs
        Steering directions of shape ``(v, 2)``, where ``v`` is the number of
        look directions. Each row contains ``(theta, phi)`` in radians:
        - ``theta`` : polar angle (0 at +z, π/2 in xy-plane),
        - ``phi``   : azimuth angle, measured counter-clockwise from +x axis.
    sampling_points
        Array of shape ``(Q, 2)`` with the microphone positions in spherical
        coordinates. Each row contains ``(theta, phi)`` in radians.
    order
        Maximum spherical harmonic order (default is 3).
    r
        Radius of the spherical microphone array in meters (default is 49e-3).
    win_len
        Window length (STFT frame size) in samples (default is 2048).

    Returns
    -------
    y : ndarray
        Beamformed signals of shape ``(v, L)``, where ``v`` is the number of
        steering directions and ``L`` is the signal length in samples.
    
    Notes
    -----
    - The implementation is based on frequency-domain processing:
      1. Compute the STFT of the input signals.
      2. Compute the spherical harmonics transform coefficients.
      3. Apply amplitude modulation equalization and beamformer weights.
      4. Reconstruct the time-domain signals using ISTFT.
    - Each row of the output corresponds to the beamformer output for one
      steering direction in `ldirs`.

    Examples
    --------
    >>> x = np.random.randn(19, 48000)   # 16-mic array, 1 s at 48 kHz
    >>> fs = 48000
    >>> ldirs = np.array([[np.pi/2, 0], [np.pi/2, np.pi/2]])  # steer to +x and +y
    >>> sampling_points = np.random.rand(19, 2) * [np.pi, 2*np.pi]
    >>> y = MaxDirectivityBeamformer(x, fs, ldirs, sampling_points, order=3)
    >>> y.shape
    (2, 48000)
    """
    # Spherical harmonics matrix for the sampling points
    Y_q = SHMatrix(sampling_points,order=order)
    # Spherical harmonics matrix for the look directions
    Y_l = SHMatrix(ldirs,order=order)
    # Compute the STFT of the input signals
    f,t,x_stft = sig.stft(x,fs=fs,nperseg=win_len)
    # Compute matrix W for plane-wave decomposition amplitude modulation equalization
    b = b_function(k=2*np.pi*f/343,r=r,ra=r,order=order)
    W,crossover_indices = AmpModMatrix(f,b,dtype=x_stft.dtype)
    # Obtain the beamformer coefficient matrix D
    D = MaxDirectivityBeamformerCoeffMatrix(f,crossover_indices)
    # Compute the spherical harmonics transform coefficients
    p_nm = np.einsum('NQ,Qft -> Nft',np.linalg.pinv(Y_q),x_stft,optimize=True)
    # Compute the beamformer
    y_stft = np.einsum('vN,Nft,Nf->vft',Y_l,p_nm,W@D,optimize=True)
    # Compute the inverse STFT to obtain the time-domain signals
    _,y = sig.istft(y_stft,fs=fs,nperseg=win_len)

    return y