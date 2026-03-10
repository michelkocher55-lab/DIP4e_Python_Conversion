from typing import Any
import numpy as np


def wavefilter(wname: Any, kind: Any = "d"):
    """
    Create wavelet decomposition and reconstruction filters.
    MATLAB-faithful translation of DIPUM wavefilter.m.
    """
    if not isinstance(wname, str):
        raise ValueError("WNAME must be a string.")

    name = wname.lower()

    if name in ("haar", "db1"):
        ld = np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0)
        hd = np.array([-1.0, 1.0], dtype=float) / np.sqrt(2.0)
        lr = ld.copy()
        hr = -hd

    elif name == "db4":
        ld = np.array(
            [
                -1.059740178499728e-002,
                3.288301166698295e-002,
                3.084138183598697e-002,
                -1.870348117188811e-001,
                -2.798376941698385e-002,
                6.308807679295904e-001,
                7.148465705525415e-001,
                2.303778133088552e-001,
            ],
            dtype=float,
        )
        t = np.arange(8, dtype=float)
        hd = np.cos(np.pi * t) * ld[::-1]
        lr = ld[::-1]
        hr = np.cos(np.pi * t) * ld

    elif name == "sym4":
        ld = np.array(
            [
                -7.576571478927333e-002,
                -2.963552764599851e-002,
                4.976186676320155e-001,
                8.037387518059161e-001,
                2.978577956052774e-001,
                -9.921954357684722e-002,
                -1.260396726203783e-002,
                3.222310060404270e-002,
            ],
            dtype=float,
        )
        t = np.arange(8, dtype=float)
        hd = np.cos(np.pi * t) * ld[::-1]
        lr = ld[::-1]
        hr = np.cos(np.pi * t) * ld

    elif name == "bior6.8":
        ld = np.array(
            [
                0,
                1.908831736481291e-003,
                -1.914286129088767e-003,
                -1.699063986760234e-002,
                1.193456527972926e-002,
                4.973290349094079e-002,
                -7.726317316720414e-002,
                -9.405920349573646e-002,
                4.207962846098268e-001,
                8.259229974584023e-001,
                4.207962846098268e-001,
                -9.405920349573646e-002,
                -7.726317316720414e-002,
                4.973290349094079e-002,
                1.193456527972926e-002,
                -1.699063986760234e-002,
                -1.914286129088767e-003,
                1.908831736481291e-003,
            ],
            dtype=float,
        )
        hd = np.array(
            [
                0,
                0,
                0,
                1.442628250562444e-002,
                -1.446750489679015e-002,
                -7.872200106262882e-002,
                4.036797903033992e-002,
                4.178491091502746e-001,
                -7.589077294536542e-001,
                4.178491091502746e-001,
                4.036797903033992e-002,
                -7.872200106262882e-002,
                -1.446750489679015e-002,
                1.442628250562444e-002,
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        )
        t = np.arange(18, dtype=float)
        lr = np.cos(np.pi * (t + 1.0)) * hd
        hr = np.cos(np.pi * t) * ld

    elif name == "jpeg9.7":
        ld = np.array(
            [
                0,
                0.02674875741080976,
                -0.01686411844287495,
                -0.07822326652898785,
                0.2668641184428723,
                0.6029490182363579,
                0.2668641184428723,
                -0.07822326652898785,
                -0.01686411844287495,
                0.02674875741080976,
            ],
            dtype=float,
        )
        hd = np.array(
            [
                0,
                0.09127176311424948,
                -0.05754352622849957,
                -0.5912717631142470,
                1.115087052456994,
                -0.5912717631142470,
                -0.05754352622849957,
                0.09127176311424948,
                0,
                0,
            ],
            dtype=float,
        )
        t = np.arange(10, dtype=float)
        lr = np.cos(np.pi * (t + 1.0)) * hd
        hr = np.cos(np.pi * t) * ld

    else:
        raise ValueError("Unrecognizable wavelet name (WNAME).")

    if not isinstance(kind, str):
        raise ValueError("TYPE must be a string.")

    k = kind.lower()[0]
    if k == "d":
        return ld, hd
    if k == "r":
        return lr, hr
    raise ValueError("Unrecognizable filter TYPE.")
