import numpy as np


def imnoise2New(f, noise_type, a=None, b=None):
    """
    Python transcription of DIPUM imnoise2.m
    f : input image, double in [0,1]
    noise_type : string
    a, b : parameters (same meaning as MATLAB)
    """

    f = np.asarray(f, dtype=np.float64)
    M, N = f.shape

    noise_type = noise_type.lower()

    # ---------------- Salt & Pepper ----------------
    if noise_type in ['salt & pepper', 'salt', 'pepper']:
        if a is None:
            a = 0.05
        if b is None:
            b = 0.05

        if a + b > 1:
            raise ValueError("The sum (a + b) must not exceed 1")

        R = np.full((M, N), 0.5)
        X = np.random.rand(M, N)

        R[X <= b] = 0
        R[(X > b) & (X <= a + b)] = 1

        fn = f.copy()
        fn[R == 0] = 0
        fn[R == 1] = 1

        return np.clip(fn, 0, 1), R

    # ---------------- Gaussian ----------------
    elif noise_type == 'gaussian':
        if a is None:
            a = 0
        if b is None:
            b = 0.01

        R = a + b * np.random.randn(M, N)
        fn = f + R

    # ---------------- Lognormal ----------------
    elif noise_type == 'lognormal':
        if a is None:
            a = 1
        if b is None:
            b = 0.25

        R = np.exp(b * np.random.randn(M, N) + a)
        fn = f + R

    # ---------------- Rayleigh ----------------
    elif noise_type == 'rayleigh':
        if a is None:
            a = 0
        if b is None:
            b = 1

        R = a + np.sqrt(-b * np.log(1 - np.random.rand(M, N)))
        fn = f + R

    # ---------------- Exponential ----------------
    elif noise_type == 'exponential':
        if a is None:
            a = 1
        if a <= 0:
            raise ValueError("Parameter a must be positive")

        R = exponential_noise(M, N, a)
        fn = f + R

    # ---------------- Erlang ----------------
    elif noise_type == 'erlang':
        if a is None:
            a = 2
        if b is None:
            b = 5

        if int(b) != b or b <= 0:
            raise ValueError("Parameter b must be a positive integer")

        R = erlang_noise(M, N, a, int(b))
        fn = f + R

        # ---------------- Uniform ----------------

    elif noise_type == 'uniform':

        if a is None:
            a = 0

        if b is None:
            b = 0.2

        if b <= a:
            raise ValueError("For uniform noise, b must be > a")

        R = a + (b - a) * np.random.rand(M, N)

        fn = f + R
    else:
        raise ValueError("Unknown distribution type")

    # MATLAB behavior: clamp to [0,1] WITHOUT renormalizing
    fn = np.clip(fn, 0, 1)

    return fn, R


# ------------------------------------------------------------------
def exponential_noise(M, N, a):
    k = -1 / a
    return k * np.log(1 - np.random.rand(M, N))


# ------------------------------------------------------------------
def erlang_noise(M, N, a, b):
    k = -1 / a
    R = np.zeros((M, N))
    for _ in range(b):
        R += k * np.log(1 - np.random.rand(M, N))
    return R