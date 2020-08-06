from math import *
'''
#include "ts07d.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

# ts07d.f -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
#


# Common Block Declarations #

struct {
    doublereal pdyn
} input_

#define input_1 input_

struct {
    doublereal a[101]
} param_

#define param_1 param_

union {
    struct {
	doublereal d__
    } _1
    struct {
	doublereal d0
    } _2
} tail_

#define tail_1 (tail_._1)
#define tail_2 (tail_._2)

struct {
    doublereal xkappa1, xkappa2
} birkpar_

#define birkpar_1 birkpar_

struct {
    doublereal g, tw
} g_

#define g_1 g_

struct {
    doublereal rh0
} rh0_

#define rh0_1 rh0_

struct {
    doublereal tss[400]	# was [80][5] #
} tss_

#define tss_1 tss_

struct {
    doublereal tso[1600]	# was [80][5][4] #
} tso_

#define tso_1 tso_

struct {
    doublereal tse[1600]	# was [80][5][4] #
} tse_

#define tse_1 tse_

struct {
    doublereal dphi, b, rho_0__, xkappa
} dphi_b_rho0__

#define dphi_b_rho0__1 dphi_b_rho0__

struct {
    integer m
} modenum_

#define modenum_1 modenum_

struct {
    doublereal dtheta
} dtheta_

#define dtheta_1 dtheta_

# Table of constant values #

static integer c__101 = 101
static doublereal c_b4 = .155
static integer c__5 = 5
static integer c__1 = 1
static integer c__0 = 0
static integer c__14 = 14
static integer c__2 = 2
static doublereal c_b89 = .14

int combo_1(int i, int j) {
	return i + 5*j
}

int combo_2(int i, int j, int k) {
	return 80*i + 400*j + k
}

void update()
{
	bool DEBUG = false

	// LOAD TSS
	for (int i = 0 i < 5 ++i) {
		std::ostringstream oss
		oss << "static/tailamebhr" << i + 1 << ".par"
		std::string filename = oss.str()
		std::ifstream file(filename)
		if (file.is_open()) {
			std::string fileline
			for (int j = 0 j < 80 ++j) {
				getline(file, fileline)
				tss_1.tss[combo_1(i, j)] = std::stod(fileline)
			}
			file.close()
		}
	}

	// LOAD TSO
	for (int i = 0 i < 5 ++i) {
		for (int j = 0 j < 4 ++j) {
			std::ostringstream oss
			oss << "static/tailamhr_o_" << i + 1 << j + 1 << ".par"
			std::string filename = oss.str()
			std::ifstream file(filename)
			if (file.is_open()) {
				std::string fileline
				for (int k = 0 k < 80 ++k) {
					getline(file, fileline)
					tso_1.tso[combo_2(i, j, k)] = std::stod(fileline)
				}
				file.close()
			}
		}
	}

	// LOAD TSE
	for (int i = 0 i < 5 ++i) {
		for (int j = 0 j < 4 ++j) {
			std::ostringstream oss
			oss << "static/tailamhr_e_" << i + 1 << j + 1 << ".par"
			std::string filename = oss.str()
			std::ifstream file(filename)
			if (file.is_open()) {
				std::string fileline
				for (int k = 0 k < 80 ++k) {
					getline(file, fileline)
					tse_1.tse[combo_2(i, j, k)] = std::stod(fileline)
				}
				file.close()
			}
		}
	}
	
	// LOAD PARAM
	std::ostringstream oss
	oss << "dynamic/2008_086_01_00.par"
	std::string filename = oss.str()
	std::ifstream file(filename)
	if (file.is_open()) {
		std::string fileline
		for (int i = 0 i < 101 ++i) {
			getline(file, fileline)
			param_1.a[i] = std::stod(fileline)
		}
		file.close()
	}

	if (DEBUG) {
		for (int i = 0 i < sizeof(tss_1.tss) / sizeof(0.0) ++i) {
			std::cout << "TSS LINE " << i << ": " << tss_1.tss[i] << std::endl
		}

		for (int i = 0 i < sizeof(tso_1.tso) / sizeof(0.0) ++i) {
			std::cout << "TSO LINE " << i << ": " << tso_1.tso[i] << std::endl
		}

		for (int i = 0 i < sizeof(tso_1.tso) / sizeof(0.0) ++i) {
			std::cout << "TSE LINE " << i << ": " << tse_1.tse[i] << std::endl
		}

		for (int i = 0 i < sizeof(param_1.a) / sizeof(0.0) ++i) {
			std::cout << "PARAM LINE " << i << ": " << param_1.a[i] << std::endl
		}
	}
}


# *********************************************************************** #

# Subroutine # int ts07d_july_2017__(integer *iopt, doublereal *parmod, 
	doublereal *ps, doublereal *x, doublereal *y, doublereal *z__, 
	doublereal *bx, doublereal *by, doublereal *bz)
{
    static doublereal bxcf, bycf, bzcf, bxr11, byr11, bzr11, bxr12, byr12, 
	    bxte[20]	# was [5][4] #, byte[20]	# was [5][4] #, 
	    bzte[20]	# was [5][4] #, bzr12, bxto[20]	# was [5][4] 
	    #, byto[20]	# was [5][4] #, bzto[20]	# was [5][4] 
	    #, bxts[5], byts[5], bzts[5], bxr21a, byr21a, bzr21a, bxr21s, 
	    byr21s, bzr21s
    extern # Subroutine # int extern_(integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *)


#  July 2017, G.K.Stephens, This routine was updated to be a double precision subroutine. #
#  To indicate the update, the subroutine was renamed from EXTMODEL to TS07D_JULY_2017. #
#  Additionally, this July 2017 update incorporates an upgraded version of the Bessel function evaluator, #
#  provided by Jay Albert, which significantly speeds up the model. We thank Jay Albert for these #
#  contributions. #

#  This subroutine computes and returns the compoents of the external magnetic field vector in the #
#  GSM coordinate system due to extraterrestrial currents, as described in the following references. #
#  To compute the total magnetic field, the magnetic field vector from the Earth's internal sources #
#  must be added. To compute this see Dr. Tsyganenko's most recent implementation of the IGRF #
#  model in the GEOPACK package (Geopack-2008_dp.for). #

#  References: (1) Tsyganenko, N. A., and M. I. Sitnov (2007), Magnetospheric configurations from a #
#                  high-resolution data-based magneticfield model, J. Geophys. Res., 112,A06225, #
#                  doi:10.1029/2007JA012260. #

#              (2) Sitnov, M. I., N. A. Tsyganenko, A. Y. Ukhorskiy, B. J. Anderson, H. Korth, #
#                  A. T. Y. Lui, and P. C. Brandt (2010),Empirical modeling of a CIR‐driven magnetic #
#                  storm, J. Geophys. Res., 115, A07231, doi:10.1029/2009JA015169. #

#  Inputs: #
#     IOPT - An option flag that allows to break out the magnetic field vector contributions from the #
#        individual modules. #
#        IOPT=0 - The total external magnetic field (most common) #
#        IOPT=1 - The field due to only the shielding of the dipole field on the magnetopause #
#        IOPT=2 - The field due to only the equatorial currents and its shielding field #
#        IOPT=3 - The field due to only the Birkeland currents and its shielding field #
#     PS - The Geodipole tilt angle in radians. #
#     PARMOD - A 10-element array, in this model this input is not used and will have no impact on the #
#        model evaluation. It is kept here because the TRACE_08 routine in the Geopack package requires #
#        a consistent signature with other empirical magnetic field models that do use this parameter. #
#     X,Y,Z - The Cartesian Geocentric position where the model will be evaluated in the GSM coordinate #
#        system in units of Earth Radii (1 RE=6371.2 KM). #

#  Common Block Inputs: #
#     /INPUT/ PDYN - The solar wind dynamic pressure in nanoPascals (nPa) #
#     /PARAM/ A - An 101-element array containing the variable (time-dependent) coefficients and #
#        parameters used in evaluating the model #
#     /TSS/ TSS - An 80x5-element array containing the static (time-independent) coefficients that are #
#        used to shield the symmetric equatorial expansions #
#     /TSO/ TSO - An 80x5x4-element array containing the static (time-independent) coefficients that are #
#        used to shield the ODD axis-symmetric equatorial expansions #
#     /TSE/ TSE - An 80x5x4-element array containing the static (time-independent) coefficients that are #
#        used to shield the EVEN axis-symmetric equatorial expansions #

#  Outputs: #
#     BX,BY,BZ - the evaluated magnetic field vector in the GSM coordinate system in units of nanoTesla (nT) 
#


    # Parameter adjustments #
    --parmod

    # Function Body #
    extern_(iopt, param_1.a, &c__101, ps, &input_1.pdyn, x, y, z__, &bxcf, &
	    bycf, &bzcf, bxts, byts, bzts, bxto, byto, bzto, bxte, byte, bzte,
	     &bxr11, &byr11, &bzr11, &bxr12, &byr12, &bzr12, &bxr21a, &byr21a,
	     &bzr21a, &bxr21s, &byr21s, &bzr21s, bx, by, bz)
    return 0

} # ts07d_july_2017__ #



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Subroutine # int extern_(integer *iopgen, doublereal *a, integer *ntot, 
	doublereal *ps, doublereal *pdyn, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bxcf, doublereal *bycf, doublereal *bzcf,
	 doublereal *bxts, doublereal *byts, doublereal *bzts, doublereal *
	bxto, doublereal *byto, doublereal *bzto, doublereal *bxte, 
	doublereal *byte, doublereal *bzte, doublereal *bxr11, doublereal *
	byr11, doublereal *bzr11, doublereal *bxr12, doublereal *byr12, 
	doublereal *bzr12, doublereal *bxr21a, doublereal *byr21a, doublereal 
	*bzr21a, doublereal *bxr21s, doublereal *byr21s, doublereal *bzr21s, 
	doublereal *bx, doublereal *by, doublereal *bz)
{
    # Initialized data #

    static doublereal a0_a__ = 34.586
    static doublereal a0_s0__ = 1.196
    static doublereal a0_x0__ = 3.4397

    # System generated locals #
    doublereal d__1

    # Builtin functions #
    // double pow_dd(doublereal *, doublereal *), sin(doublereal), sqrt(
	    //doublereal)

    # Local variables #
    extern # Subroutine # int deformed_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *)
    static doublereal p_factor__
    extern # Subroutine # int shlcar3x3_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), birk_tot__(doublereal *, doublereal *, doublereal *
	    , doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), birtotsy_(doublereal *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *)
    static integer k, l
    static doublereal s0, x0, am, tx, ty, tz, xx, yy, zz
    static integer ind
    static doublereal bbx, bby, bbz, cfx, cfy, cfz, sps, a_r11__, a_r12__, 
	    a_r21a__, a_r21s__, bxr22a, byr22a, bzr22a, xappa, bxr11s, byr11s,
	     bzr11s, bxr12s, byr12s, bzr12s, bxr22s, byr22s, bzr22s, pdyn_0__,
	     xappa3


#   IOPGEN - GENERAL OPTION FLAG:  IOPGEN=0 - CALCULATE TOTAL FIELD #
#                                  IOPGEN=1 - DIPOLE SHIELDING ONLY #
#                                  IOPGEN=2 - TAIL FIELD ONLY #
#                                  IOPGEN=3 - BIRKELAND FIELD ONLY #
#                                  IOPGEN=4 - RING CURRENT FIELD ONLY #




# THE COMMON BLOCK FORWARDS TAIL SHEET THICKNESS #
#  SCALING FACTORS FOR BIRKELAN #

    # Parameter adjustments #
    --a
    --bxts
    --byts
    --bzts
    bxto -= 6
    byto -= 6
    bzto -= 6
    bxte -= 6
    byte -= 6
    bzte -= 6

    # Function Body #
#   SHUE ET A #
    d__1 = *pdyn / 2.
    xappa = pow(d__1, c_b4)
#   0.155 is the value obtained in TS05 #
# Computing 3rd power #
    d__1 = xappa
    xappa3 = d__1 * (d__1 * d__1)
    tail_1.d__ = a[96]
    rh0_1.rh0 = a[97]
    g_1.g = a[98]
    birkpar_1.xkappa1 = a[99]
    birkpar_1.xkappa2 = a[100]
    g_1.tw = a[101]

#   THIS PARAMETER CONTROLS THE IMF-INDUCED T #
    xx = *x * xappa
# pressure scaling has been reinstated here #
    yy = *y * xappa
    zz = *z__ * xappa
#     print *,XAPPA,PDYN #

    sps = sin(*ps)

    x0 = a0_x0__ / xappa
# pressure scaling has been reinstated, even thou #
    am = a0_a__ / xappa
# pressure scaling has been reinstated, even thou #
    s0 = a0_s0__

#   CALCULATE THE IMF CLOCK ANGLE: #

#        IF (BYIMF.EQ.0.D0.AND.BZIMF.EQ.0.D0) THEN #
#            THETA=0.D0 #
#         ELSE #
#            THETA=DATAN2(BYIMF,BZIMF) #
#            IF (THETA.LE.0.D0) THETA=THETA+6.283185307D0 #
#        ENDIF #

#       CT=COS(THETA) #
#       ST=SIN(THETA) #
#       YS=Y*CT-Z*ST #
#       ZS=Z*CT+Y*ST #

#       STHETAH=SIN(THETA/2.)**2 #

#  CALCULATE "IMF" COMPONENTS OUTSIDE THE MAGNETOPAUSE LAYER (HENCE BEGIN WITH "O") #
#  THEY ARE NEEDED ONLY IF THE POINT (X,Y,Z) IS WITHIN THE TRANSITION MAGNETOPAUSE LAYER #
#  OR OUTSIDE THE MAGNETOSPHERE: #

#      FACTIMF=A(24)+A(25)*STHETAH #

#      OIMFX=0.D0 #
#      OIMFY=BYIMF*FACTIMF #
#      OIMFZ=BZIMF*FACTIMF #

# ===================================================================== #
#  THIS FRAGMENT (BETWEEN THE ===== LINES) DISABLES THE CALCULATION OF THE MAGNETOPAUSE POSITION #
#  IT SHOULD BE USED ONLY FOR THE FITTING (WE ASSUME THAT NO POINTS FROM THE SHEATH ARE PRESENT #
#  IN THE DATASET, WHICH ITSELF IS STILL A QUESTION). #

#  REMOVE IT IN THE FINAL VERSION. #
#      SIGMA=0.D0 #
#      GOTO 1111 #
# ====================================================================== #

#      R=SQRT(X**2+Y**2+Z**2) #
#      XSS=X #
#      ZSS=Z #
#  1   XSOLD=XSS      !   BEGIN ITERATIVE SEARCH OF UNWARPED COORDS (TO FIND SIGMA) #
#      ZSOLD=ZSS #
#      RH=RH0+RH2*(ZSS/R)**2 #
#      SINPSAS=SPS/(1.D0+(R/RH)**3)**0.33333333D0 #
#      COSPSAS=DSQRT(1.D0-SINPSAS**2) #
#      ZSS=X*SINPSAS+Z*COSPSAS #
#      XSS=X*COSPSAS-Z*SINPSAS #
#      DD=DABS(XSS-XSOLD)+DABS(ZSS-ZSOLD) #
#      IF (DD.GT.1.D-6) GOTO 1 #
#                                END OF ITERATIVE SEARCH #
#      RHO2=Y**2+ZSS**2 #
#      ASQ=AM**2 #
#      XMXM=AM+XSS-X0 #
#      IF (XMXM.LT.0.) XMXM=0. ! THE BOUNDARY IS A CYLINDER TAILWARD OF X=X0-AM #
#      AXX0=XMXM**2 #
#      ARO=ASQ+RHO2 #
#      SIGMA=DSQRT((ARO+AXX0+SQRT((ARO+AXX0)**2-4.*ASQ*AXX0))/(2.*ASQ)) #
# ================================================================== #
# 1111 CONTINUE  !!!!!!!!!!!!  REMOVE IN THE FINAL VERSION #
# ================================================================== #

#   NOW, THERE ARE THREE POSSIBLE CASES: #
#    (1) INSIDE THE MAGNETOSPHERE   (SIGMA #
#    (2) IN THE BOUNDARY LAYER #
#    (3) OUTSIDE THE MAGNETOSPHERE AND B.LAYER #
#       FIRST OF ALL, CONSIDER THE CASES (1) AND (2): #

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#      IF (SIGMA.LT.S0+DSIG) THEN  !  CASES (1) OR (2) CALCULATE THE MODEL FIELD #
#                                   (WITH THE POTENTIAL "PENETRATED" INTERCONNECTION FIELD): #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    if (*iopgen <= 1) {
#     print *,XX,YY,ZZ #
	shlcar3x3_(&xx, &yy, &zz, ps, &cfx, &cfy, &cfz)
#  DIPOLE SHIEL #
	*bxcf = cfx * xappa3
	*bycf = cfy * xappa3
	*bzcf = cfz * xappa3
    } else {
	*bxcf = 0.
	*bycf = 0.
	*bzcf = 0.
    }
#  DONE #
    if (*iopgen == 0 || *iopgen == 2) {
	deformed_(ps, &xx, &yy, &zz, &bxts[1], &byts[1], &bzts[1], &bxto[6], &
		byto[6], &bzto[6], &bxte[6], &byte[6], &bzte[6])
#  TAIL FIELD (THREE MO #
    } else {
	for (k = 1 k <= 5 ++k) {
	    bxts[k] = 0.
	    byts[k] = 0.
	    bzts[k] = 0.
# L11: #
	}
	for (k = 1 k <= 5 ++k) {
	    for (l = 1 l <= 4 ++l) {
		bxto[k + l * 5] = 0.
		byto[k + l * 5] = 0.
		bzto[k + l * 5] = 0.
		bxte[k + l * 5] = 0.
		byte[k + l * 5] = 0.
		bzte[k + l * 5] = 0.
# L13: #
	    }
# L12: #
	}
    }
    if (*iopgen == 0 || *iopgen == 3) {
	birk_tot__(ps, &xx, &yy, &zz, bxr11, byr11, bzr11, bxr12, byr12, 
		bzr12, bxr21a, byr21a, bzr21a, &bxr22a, &byr22a, &bzr22a)
#   BIRKELA #
	birtotsy_(ps, &xx, &yy, &zz, &bxr11s, &byr11s, &bzr11s, &bxr12s, &
		byr12s, &bzr12s, bxr21s, byr21s, bzr21s, &bxr22s, &byr22s, &
		bzr22s)
#                                                                        (TWO MODES FOR R1s AND TWO MODES FOR
 R2s) #
#                                                                        (but we actually use from here only 
R2s modes) #

    } else {
	*bxr11 = 0.
	*byr11 = 0.
	*bzr11 = 0.
	*bxr12 = 0.
	*byr12 = 0.
	*bzr12 = 0.
	*bxr21a = 0.
	*byr21a = 0.
	*bzr21a = 0.
	*bxr21s = 0.
	*byr21s = 0.
	*bzr21s = 0.
    }

# ----------------------------------------------------------- #

#    NOW, ADD UP ALL THE COMPONENTS: #
    a_r11__ = a[92]
    a_r12__ = a[93]
    a_r21a__ = a[94]
    a_r21s__ = a[95]
    tx = 0.
    ty = 0.
    tz = 0.
# --- New tail structure ------------- #
    pdyn_0__ = 2.
#   AVERAGE PRESSURE USED FOR NORMALIZATION #
    p_factor__ = sqrt(*pdyn / pdyn_0__) - 1.
    ind = 1
    for (k = 1 k <= 5 ++k) {
	++ind
	tx += (a[ind] + a[ind + 45] * p_factor__) * bxts[k]
#   2 - 6  &  47 #
	ty += (a[ind] + a[ind + 45] * p_factor__) * byts[k]
	tz += (a[ind] + a[ind + 45] * p_factor__) * bzts[k]
# L911: #
    }
    for (k = 1 k <= 5 ++k) {
	for (l = 1 l <= 4 ++l) {
	    ++ind
	    tx += (a[ind] + a[ind + 45] * p_factor__) * bxto[k + l * 5]
#   7 -26  &  52 #
	    ty += (a[ind] + a[ind + 45] * p_factor__) * byto[k + l * 5]
	    tz += (a[ind] + a[ind + 45] * p_factor__) * bzto[k + l * 5]
	    tx += (a[ind + 20] + a[ind + 65] * p_factor__) * bxte[k + l * 5]
#   27 -46  & #
	    ty += (a[ind + 20] + a[ind + 65] * p_factor__) * byte[k + l * 5]
	    tz += (a[ind + 20] + a[ind + 65] * p_factor__) * bzte[k + l * 5]
# L913: #
	}
# L912: #
    }
    bbx = a[1] * *bxcf + tx + a_r11__ * *bxr11 + a_r12__ * *bxr12 + a_r21a__ *
	     *bxr21a + a_r21s__ * *bxr21s
    bby = a[1] * *bycf + ty + a_r11__ * *byr11 + a_r12__ * *byr12 + a_r21a__ *
	     *byr21a + a_r21s__ * *byr21s
    bbz = a[1] * *bzcf + tz + a_r11__ * *bzr11 + a_r12__ * *bzr12 + a_r21a__ *
	     *bzr21a + a_r21s__ * *bzr21s

#   ----------------------------------------------------------- #

#   AND WE HAVE THE TOTAL EXTERNAL FIELD. #

#  NOW, LET US CHECK WHETHER WE HAVE THE CASE (1). IF YES - WE ARE DONE: #

#      IF (SIGMA.LT.S0-DSIG) THEN    !  (X,Y,Z) IS INSIDE THE MAGNETOSPHERE #
#       BX=BBX #
#       BY=BBY #
#       BZ=BBZ #
#                     ELSE           !  THIS IS THE MOST COMPLEX CASE: WE ARE INSIDE #
#                                             THE INTERPOLATION REGION #
#       FINT=0.5*(1.-(SIGMA-S0)/DSIG) #
#       FEXT=0.5*(1.+(SIGMA-S0)/DSIG) #

#       CALL DIPOLE (PS,X,Y,Z,QX,QY,QZ) #
#       BX=(BBX+QX)*FINT+OIMFX*FEXT -QX #
#       BY=(BBY+QY)*FINT+OIMFY*FEXT -QY #
#       BZ=(BBZ+QZ)*FINT+OIMFZ*FEXT -QZ #

#        ENDIF  !   THE CASES (1) AND (2) ARE EXHAUSTED THE ONLY REMAINING #
#                      POSSIBILITY IS NOW THE CASE (3): #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#        ELSE #
#                CALL DIPOLE (PS,X,Y,Z,QX,QY,QZ) #
#                BX=OIMFX-QX #
#                BY=OIMFY-QY #
#                BZ=OIMFZ-QZ #
#        ENDIF #

    *bx = bbx
    *by = bby
    *bz = bbz
    return 0
} # extern_ #


# XXXXXXXXXXXXXXXXXXXXXXXXXXX11/15/05 16:06 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

# Subroutine # int shlcar3x3_(doublereal *x, doublereal *y, doublereal *z__,
	 doublereal *ps, doublereal *bx, doublereal *by, doublereal *bz)
{
    # Initialized data #

    static doublereal a[50] = { -901.2327248,895.8011176,817.6208321,
	    -845.5880889,-83.73539535,86.58542841,336.8781402,-329.3619944,
	    -311.294712,308.6011161,31.94469304,-31.30824526,125.8739681,
	    -372.3384278,-235.4720434,286.7594095,21.86305585,-27.42344605,
	    -150.4874688,2.669338538,1.395023949,-.5540427503,-56.85224007,
	    3.681827033,-43.48705106,5.103131905,1.073551279,-.6673083508,
	    12.21404266,4.177465543,5.799964188,-.3977802319,-1.044652977,
	    .570356001,3.536082962,-3.222069852,9.620648151,6.082014949,
	    27.75216226,12.44199571,5.122226936,6.982039615,20.12149582,
	    6.150973118,4.663639687,15.73319647,2.303504968,5.840511214,
	    .08385953499,.3477844929 }

    # System generated locals #
    doublereal d__1, d__2

    # Builtin functions #
    //double cos(doublereal), sin(doublereal), sqrt(doublereal), exp(doublereal)
	    //

    # Local variables #
    static doublereal a1, a2, a3, a4, a5, a6, a7, a8, a9, p1, p2, p3, r1, r2, 
	    r3, q1, q2, q3, s1, s2, s3, t1, t2, x1, z1, x2, z2, ct1, ct2, fx1,
	     fx2, fz1, hy1, hx1, hz1, hy2, fz2, hx2, st1, st2, hz2, fx3, hy3, 
	    fz3, hx3, hz3, fx4, hy4, fz4, hx4, hz4, fx5, hy5, fz5, hx5, hz5, 
	    fx6, hy6, fz6, hx6, hz6, fx7, hy7, fz7, hx7, hz7, fx8, hy8, fz8, 
	    hx8, hz8, fx9, hy9, fz9, hx9, hz9, cps, cyp, cyq, czr, czs, sps, 
	    syp, syq, szr, szs, s2ps, expr, exqs, sqpr, sqqs


#   THIS S/R RETURNS THE SHIELDING FIELD FOR THE EARTH'S DIPOLE, #
#   REPRESENTED BY  2x3x3=18 "CARTESIAN" HARMONICS, tilted with respect #
#   to the z=0 plane (see NB#4, p.74-74) #

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#  The 36 coefficients enter in pairs in the amplitudes of the "cartesian" #
#    harmonics (A(1)-A(36). #
#  The 14 nonlinear parameters (A(37)-A(50) are the scales Pi,Ri,Qi,and Si #
#   entering the arguments of exponents, sines, and cosines in each of the #
#   18 "Cartesian" harmonics  PLUS TWO TILT ANGLES FOR THE CARTESIAN HARMONICS #
#       (ONE FOR THE PSI=0 MODE AND ANOTHER FOR THE PSI=90 MODE) #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #



    p1 = a[36]
    p2 = a[37]
    p3 = a[38]
    r1 = a[39]
    r2 = a[40]
    r3 = a[41]
    q1 = a[42]
    q2 = a[43]
    q3 = a[44]
    s1 = a[45]
    s2 = a[46]
    s3 = a[47]
    t1 = a[48]
    t2 = a[49]

    cps = cos(*ps)
    sps = sin(*ps)
    s2ps = cps * 2.

#   MODIFIED HERE (INSTEAD OF SIN(3*PS) I TR #
    st1 = sin(*ps * t1)
    ct1 = cos(*ps * t1)
    st2 = sin(*ps * t2)
    ct2 = cos(*ps * t2)
#     print *,X,Z #
    x1 = *x * ct1 - *z__ * st1
#         print *,'X1=',X1 #
    z1 = *x * st1 + *z__ * ct1
    x2 = *x * ct2 - *z__ * st2
    z2 = *x * st2 + *z__ * ct2


#  MAKE THE TERMS IN THE 1ST SUM ("PERPENDICULAR" SYMMETRY): #

#       I=1 #

# Computing 2nd power #
    d__1 = p1
# Computing 2nd power #
    d__2 = r1
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p1)
    syp = sin(*y / p1)
    czr = cos(z1 / r1)
    szr = sin(z1 / r1)
#       print *,X1 #
    expr = exp(sqpr * x1)
    fx1 = -sqpr * expr * cyp * szr
    hy1 = expr / p1 * syp * szr
    fz1 = -expr * cyp / r1 * czr
    hx1 = fx1 * ct1 + fz1 * st1
    hz1 = -fx1 * st1 + fz1 * ct1
# Computing 2nd power #
    d__1 = p1
# Computing 2nd power #
    d__2 = r2
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p1)
    syp = sin(*y / p1)
    czr = cos(z1 / r2)
    szr = sin(z1 / r2)
    expr = exp(sqpr * x1)
    fx2 = -sqpr * expr * cyp * szr
    hy2 = expr / p1 * syp * szr
    fz2 = -expr * cyp / r2 * czr
    hx2 = fx2 * ct1 + fz2 * st1
    hz2 = -fx2 * st1 + fz2 * ct1
# Computing 2nd power #
    d__1 = p1
# Computing 2nd power #
    d__2 = r3
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p1)
    syp = sin(*y / p1)
    czr = cos(z1 / r3)
    szr = sin(z1 / r3)
    expr = exp(sqpr * x1)
    fx3 = -expr * cyp * (sqpr * z1 * czr + szr / r3 * (x1 + 1. / sqpr))
    hy3 = expr / p1 * syp * (z1 * czr + x1 / r3 * szr / sqpr)
# Computing 2nd power #
    d__1 = r3
    fz3 = -expr * cyp * (czr * (x1 / (d__1 * d__1) / sqpr + 1.) - z1 / r3 * 
	    szr)
    hx3 = fx3 * ct1 + fz3 * st1
    hz3 = -fx3 * st1 + fz3 * ct1

#       I=2: #

# Computing 2nd power #
    d__1 = p2
# Computing 2nd power #
    d__2 = r1
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p2)
    syp = sin(*y / p2)
    czr = cos(z1 / r1)
    szr = sin(z1 / r1)
    expr = exp(sqpr * x1)
    fx4 = -sqpr * expr * cyp * szr
    hy4 = expr / p2 * syp * szr
    fz4 = -expr * cyp / r1 * czr
    hx4 = fx4 * ct1 + fz4 * st1
    hz4 = -fx4 * st1 + fz4 * ct1
# Computing 2nd power #
    d__1 = p2
# Computing 2nd power #
    d__2 = r2
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p2)
    syp = sin(*y / p2)
    czr = cos(z1 / r2)
    szr = sin(z1 / r2)
    expr = exp(sqpr * x1)
    fx5 = -sqpr * expr * cyp * szr
    hy5 = expr / p2 * syp * szr
    fz5 = -expr * cyp / r2 * czr
    hx5 = fx5 * ct1 + fz5 * st1
    hz5 = -fx5 * st1 + fz5 * ct1
# Computing 2nd power #
    d__1 = p2
# Computing 2nd power #
    d__2 = r3
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p2)
    syp = sin(*y / p2)
    czr = cos(z1 / r3)
    szr = sin(z1 / r3)
    expr = exp(sqpr * x1)
    fx6 = -expr * cyp * (sqpr * z1 * czr + szr / r3 * (x1 + 1. / sqpr))
    hy6 = expr / p2 * syp * (z1 * czr + x1 / r3 * szr / sqpr)
# Computing 2nd power #
    d__1 = r3
    fz6 = -expr * cyp * (czr * (x1 / (d__1 * d__1) / sqpr + 1.) - z1 / r3 * 
	    szr)
    hx6 = fx6 * ct1 + fz6 * st1
    hz6 = -fx6 * st1 + fz6 * ct1

#       I=3: #

# Computing 2nd power #
    d__1 = p3
# Computing 2nd power #
    d__2 = r1
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p3)
    syp = sin(*y / p3)
    czr = cos(z1 / r1)
    szr = sin(z1 / r1)
    expr = exp(sqpr * x1)
    fx7 = -sqpr * expr * cyp * szr
    hy7 = expr / p3 * syp * szr
    fz7 = -expr * cyp / r1 * czr
    hx7 = fx7 * ct1 + fz7 * st1
    hz7 = -fx7 * st1 + fz7 * ct1
# Computing 2nd power #
    d__1 = p3
# Computing 2nd power #
    d__2 = r2
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p3)
    syp = sin(*y / p3)
    czr = cos(z1 / r2)
    szr = sin(z1 / r2)
    expr = exp(sqpr * x1)
    fx8 = -sqpr * expr * cyp * szr
    hy8 = expr / p3 * syp * szr
    fz8 = -expr * cyp / r2 * czr
    hx8 = fx8 * ct1 + fz8 * st1
    hz8 = -fx8 * st1 + fz8 * ct1
# Computing 2nd power #
    d__1 = p3
# Computing 2nd power #
    d__2 = r3
    sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyp = cos(*y / p3)
    syp = sin(*y / p3)
    czr = cos(z1 / r3)
    szr = sin(z1 / r3)
    expr = exp(sqpr * x1)
    fx9 = -expr * cyp * (sqpr * z1 * czr + szr / r3 * (x1 + 1. / sqpr))
    hy9 = expr / p3 * syp * (z1 * czr + x1 / r3 * szr / sqpr)
# Computing 2nd power #
    d__1 = r3
    fz9 = -expr * cyp * (czr * (x1 / (d__1 * d__1) / sqpr + 1.) - z1 / r3 * 
	    szr)
    hx9 = fx9 * ct1 + fz9 * st1
    hz9 = -fx9 * st1 + fz9 * ct1
    a1 = a[0] + a[1] * cps
    a2 = a[2] + a[3] * cps
    a3 = a[4] + a[5] * cps
    a4 = a[6] + a[7] * cps
    a5 = a[8] + a[9] * cps
    a6 = a[10] + a[11] * cps
    a7 = a[12] + a[13] * cps
    a8 = a[14] + a[15] * cps
    a9 = a[16] + a[17] * cps
    *bx = a1 * hx1 + a2 * hx2 + a3 * hx3 + a4 * hx4 + a5 * hx5 + a6 * hx6 + 
	    a7 * hx7 + a8 * hx8 + a9 * hx9
    *by = a1 * hy1 + a2 * hy2 + a3 * hy3 + a4 * hy4 + a5 * hy5 + a6 * hy6 + 
	    a7 * hy7 + a8 * hy8 + a9 * hy9
    *bz = a1 * hz1 + a2 * hz2 + a3 * hz3 + a4 * hz4 + a5 * hz5 + a6 * hz6 + 
	    a7 * hz7 + a8 * hz8 + a9 * hz9
#  MAKE THE TERMS IN THE 2ND SUM ("PARALLEL" SYMMETRY): #

#       I=1 #

# Computing 2nd power #
    d__1 = q1
# Computing 2nd power #
    d__2 = s1
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q1)
    syq = sin(*y / q1)
    czs = cos(z2 / s1)
    szs = sin(z2 / s1)
    exqs = exp(sqqs * x2)
    fx1 = -sqqs * exqs * cyq * czs * sps
    hy1 = exqs / q1 * syq * czs * sps
    fz1 = exqs * cyq / s1 * szs * sps
    hx1 = fx1 * ct2 + fz1 * st2
    hz1 = -fx1 * st2 + fz1 * ct2
# Computing 2nd power #
    d__1 = q1
# Computing 2nd power #
    d__2 = s2
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q1)
    syq = sin(*y / q1)
    czs = cos(z2 / s2)
    szs = sin(z2 / s2)
    exqs = exp(sqqs * x2)
    fx2 = -sqqs * exqs * cyq * czs * sps
    hy2 = exqs / q1 * syq * czs * sps
    fz2 = exqs * cyq / s2 * szs * sps
    hx2 = fx2 * ct2 + fz2 * st2
    hz2 = -fx2 * st2 + fz2 * ct2
# Computing 2nd power #
    d__1 = q1
# Computing 2nd power #
    d__2 = s3
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q1)
    syq = sin(*y / q1)
    czs = cos(z2 / s3)
    szs = sin(z2 / s3)
    exqs = exp(sqqs * x2)
    fx3 = -sqqs * exqs * cyq * czs * sps
    hy3 = exqs / q1 * syq * czs * sps
    fz3 = exqs * cyq / s3 * szs * sps
    hx3 = fx3 * ct2 + fz3 * st2
    hz3 = -fx3 * st2 + fz3 * ct2

#       I=2 #

# Computing 2nd power #
    d__1 = q2
# Computing 2nd power #
    d__2 = s1
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q2)
    syq = sin(*y / q2)
    czs = cos(z2 / s1)
    szs = sin(z2 / s1)
    exqs = exp(sqqs * x2)
    fx4 = -sqqs * exqs * cyq * czs * sps
    hy4 = exqs / q2 * syq * czs * sps
    fz4 = exqs * cyq / s1 * szs * sps
    hx4 = fx4 * ct2 + fz4 * st2
    hz4 = -fx4 * st2 + fz4 * ct2
# Computing 2nd power #
    d__1 = q2
# Computing 2nd power #
    d__2 = s2
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q2)
    syq = sin(*y / q2)
    czs = cos(z2 / s2)
    szs = sin(z2 / s2)
    exqs = exp(sqqs * x2)
    fx5 = -sqqs * exqs * cyq * czs * sps
    hy5 = exqs / q2 * syq * czs * sps
    fz5 = exqs * cyq / s2 * szs * sps
    hx5 = fx5 * ct2 + fz5 * st2
    hz5 = -fx5 * st2 + fz5 * ct2
# Computing 2nd power #
    d__1 = q2
# Computing 2nd power #
    d__2 = s3
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q2)
    syq = sin(*y / q2)
    czs = cos(z2 / s3)
    szs = sin(z2 / s3)
    exqs = exp(sqqs * x2)
    fx6 = -sqqs * exqs * cyq * czs * sps
    hy6 = exqs / q2 * syq * czs * sps
    fz6 = exqs * cyq / s3 * szs * sps
    hx6 = fx6 * ct2 + fz6 * st2
    hz6 = -fx6 * st2 + fz6 * ct2

#       I=3 #

# Computing 2nd power #
    d__1 = q3
# Computing 2nd power #
    d__2 = s1
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q3)
    syq = sin(*y / q3)
    czs = cos(z2 / s1)
    szs = sin(z2 / s1)
    exqs = exp(sqqs * x2)
    fx7 = -sqqs * exqs * cyq * czs * sps
    hy7 = exqs / q3 * syq * czs * sps
    fz7 = exqs * cyq / s1 * szs * sps
    hx7 = fx7 * ct2 + fz7 * st2
    hz7 = -fx7 * st2 + fz7 * ct2
# Computing 2nd power #
    d__1 = q3
# Computing 2nd power #
    d__2 = s2
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q3)
    syq = sin(*y / q3)
    czs = cos(z2 / s2)
    szs = sin(z2 / s2)
    exqs = exp(sqqs * x2)
    fx8 = -sqqs * exqs * cyq * czs * sps
    hy8 = exqs / q3 * syq * czs * sps
    fz8 = exqs * cyq / s2 * szs * sps
    hx8 = fx8 * ct2 + fz8 * st2
    hz8 = -fx8 * st2 + fz8 * ct2
# Computing 2nd power #
    d__1 = q3
# Computing 2nd power #
    d__2 = s3
    sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
    cyq = cos(*y / q3)
    syq = sin(*y / q3)
    czs = cos(z2 / s3)
    szs = sin(z2 / s3)
    exqs = exp(sqqs * x2)
    fx9 = -sqqs * exqs * cyq * czs * sps
    hy9 = exqs / q3 * syq * czs * sps
    fz9 = exqs * cyq / s3 * szs * sps
    hx9 = fx9 * ct2 + fz9 * st2
    hz9 = -fx9 * st2 + fz9 * ct2
    a1 = a[18] + a[19] * s2ps
    a2 = a[20] + a[21] * s2ps
    a3 = a[22] + a[23] * s2ps
    a4 = a[24] + a[25] * s2ps
    a5 = a[26] + a[27] * s2ps
    a6 = a[28] + a[29] * s2ps
    a7 = a[30] + a[31] * s2ps
    a8 = a[32] + a[33] * s2ps
    a9 = a[34] + a[35] * s2ps
    *bx = *bx + a1 * hx1 + a2 * hx2 + a3 * hx3 + a4 * hx4 + a5 * hx5 + a6 * 
	    hx6 + a7 * hx7 + a8 * hx8 + a9 * hx9
    *by = *by + a1 * hy1 + a2 * hy2 + a3 * hy3 + a4 * hy4 + a5 * hy5 + a6 * 
	    hy6 + a7 * hy7 + a8 * hy8 + a9 * hy9
    *bz = *bz + a1 * hz1 + a2 * hz2 + a3 * hz3 + a4 * hz4 + a5 * hz5 + a6 * 
	    hz6 + a7 * hz7 + a8 * hz8 + a9 * hz9

    return 0
} # shlcar3x3_ #


# ############################################################################ #


# Subroutine # int deformed_(doublereal *ps, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bxs, doublereal *bys, doublereal *bzs, 
	doublereal *bxo, doublereal *byo, doublereal *bzo, doublereal *bxe, 
	doublereal *bye, doublereal *bze)
{
    # Initialized data #

    static doublereal rh2 = -5.2
    static integer ieps = 3

    # System generated locals #
    integer i__1, i__2
    doublereal d__1, d__2, d__3

    # Builtin functions #
    // double sin(doublereal), sqrt(doublereal), pow_di(doublereal *, integer *),
	     //pow_dd(doublereal *, doublereal *)

    # Local variables #
    static doublereal f
    static integer k, l
    static doublereal r__, r2, rh, zr, cps, rrh, xas, zas, sps, fac1, fac2, 
	    fac3, dfdr, dfdrh, facps, bxase[20]	# was [5][4] #, byase[20]	
	    # was [5][4] #, bzase[20]	# was [5][4] #, drhdr, cpsas, drhdz,
	     bxaso[20]	# was [5][4] #, byaso[20]	# was [5][4] #, 
	    bzaso[20]	# was [5][4] #, bxass[5], byass[5], bzass[5], spsas,
	     psasx, psasy, psasz
    extern # Subroutine # int warped_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *)
    static doublereal dxasdx, dxasdy, dxasdz, dzasdx, dzasdy, dzasdz


#    CALCULATES GSM COMPONENTS OF 104 UNIT-AMPLITUDE TAIL FIELD MODES, #
#    TAKING INTO ACCOUNT BOTH EFFECTS OF DIPOLE TILT: #
#    WARPING IN Y-Z (DONE BY THE S/R WARPED) AND BENDING IN X-Z (DONE BY THIS SUBROUTINE) #




    # Parameter adjustments #
    bze -= 6
    bye -= 6
    bxe -= 6
    bzo -= 6
    byo -= 6
    bxo -= 6
    --bzs
    --bys
    --bxs

    # Function Body #

#  RH0,RH1,RH2, AND IEPS CONTROL THE TILT-RELATED DEFORMATION OF THE TAIL FIELD #

    sps = sin(*ps)
# Computing 2nd power #
    d__1 = sps
    cps = sqrt(1. - d__1 * d__1)
# Computing 2nd power #
    d__1 = *x
# Computing 2nd power #
    d__2 = *y
# Computing 2nd power #
    d__3 = *z__
    r2 = d__1 * d__1 + d__2 * d__2 + d__3 * d__3
    r__ = sqrt(r2)
    zr = *z__ / r__
# Computing 2nd power #
    d__1 = zr
    rh = rh0_1.rh0 + rh2 * (d__1 * d__1)
    drhdr = -zr / r__ * 2. * rh2 * zr
    drhdz = rh2 * 2. * zr / r__

    rrh = r__ / rh
    d__1 = pow(rrh, ieps) + 1.
    d__2 = 1. / ieps
    f = 1. / pow(d__1, d__2)
    i__1 = ieps - 1
    i__2 = ieps + 1
    dfdr = -pow(rrh, i__1) * pow(f, i__2) / rh
    dfdrh = -rrh * dfdr

    spsas = sps * f
# Computing 2nd power #
    d__1 = spsas
    cpsas = sqrt(1. - d__1 * d__1)

    xas = *x * cpsas - *z__ * spsas
    zas = *x * spsas + *z__ * cpsas

    facps = sps / cpsas * (dfdr + dfdrh * drhdr) / r__
    psasx = facps * *x
    psasy = facps * *y
    psasz = facps * *z__ + sps / cpsas * dfdrh * drhdz

    dxasdx = cpsas - zas * psasx
    dxasdy = -zas * psasy
    dxasdz = -spsas - zas * psasz
    dzasdx = spsas + xas * psasx
    dzasdy = xas * psasy
    dzasdz = cpsas + xas * psasz
    fac1 = dxasdz * dzasdy - dxasdy * dzasdz
    fac2 = dxasdx * dzasdz - dxasdz * dzasdx
    fac3 = dzasdx * dxasdy - dxasdx * dzasdy

#     DEFORM: #

    warped_(ps, &xas, y, &zas, bxass, byass, bzass, bxaso, byaso, bzaso, 
	    bxase, byase, bzase)

# --- New tail structure ------------- #
    for (k = 1 k <= 5 ++k) {
	bxs[k] = bxass[k - 1] * dzasdz - bzass[k - 1] * dxasdz + byass[k - 1] 
		* fac1
	bys[k] = byass[k - 1] * fac2
	bzs[k] = bzass[k - 1] * dxasdx - bxass[k - 1] * dzasdx + byass[k - 1] 
		* fac3
# L11: #
    }
    for (k = 1 k <= 5 ++k) {
	for (l = 1 l <= 4 ++l) {
	    bxo[k + l * 5] = bxaso[k + l * 5 - 6] * dzasdz - bzaso[k + l * 5 
		    - 6] * dxasdz + byaso[k + l * 5 - 6] * fac1
	    byo[k + l * 5] = byaso[k + l * 5 - 6] * fac2
	    bzo[k + l * 5] = bzaso[k + l * 5 - 6] * dxasdx - bxaso[k + l * 5 
		    - 6] * dzasdx + byaso[k + l * 5 - 6] * fac3
	    bxe[k + l * 5] = bxase[k + l * 5 - 6] * dzasdz - bzase[k + l * 5 
		    - 6] * dxasdz + byase[k + l * 5 - 6] * fac1
	    bye[k + l * 5] = byase[k + l * 5 - 6] * fac2
	    bze[k + l * 5] = bzase[k + l * 5 - 6] * dxasdx - bxase[k + l * 5 
		    - 6] * dzasdx + byase[k + l * 5 - 6] * fac3
# L13: #
	}
# L12: #
    }
# ------------------------------------ #

    return 0
} # deformed_ #


# ------------------------------------------------------------------ #


# Subroutine # int warped_(doublereal *ps, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bxs, doublereal *bys, doublereal *bzs, 
	doublereal *bxo, doublereal *byo, doublereal *bzo, doublereal *bxe, 
	doublereal *bye, doublereal *bze)
{
    # System generated locals #
    doublereal d__1, d__2, d__3

    # Builtin functions #
    // double sin(doublereal), sqrt(doublereal), atan2(doublereal, doublereal), 
	    //cos(doublereal)

    # Local variables #
    extern # Subroutine # int unwarped_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *)
    static doublereal f
    static integer k, l
    static doublereal cf, sf, xl, phi, rho, yas, zas, sps, rho2, cphi, dfdx, 
	    dgdx, sphi, rr4l4, dxldx, dfdphi, bx_ase__[20]	# was [5][4] 
	    #, by_ase__[20]	# was [5][4] #, bz_ase__[20]	# was [5][4] 
	    #, bphi_s__, dfdrho, bx_aso__[20]	# was [5][4] #, by_aso__[20]
	    	# was [5][4] #, bz_aso__[20]	# was [5][4] #, brho_s__, 
	    bx_ass__[5], by_ass__[5], bz_ass__[5], bphi_as__, brho_as__


#   CALCULATES GSM COMPONENTS OF THE WARPED FIELD FOR TWO TAIL UNIT MODES. #
#   THE WARPING DEFORMATION IS IMPOSED ON THE UNWARPED FIELD, COMPUTED #
#   BY THE S/R "UNWARPED".  THE WARPING PARAMETERS WERE TAKEN FROM THE #
#   RESULTS OF GEOTAIL OBSERVATIONS (TSYGANENKO ET AL. [1998]). #
#   NB # 6, P.106, OCT 12, 2000. #




    # Parameter adjustments #
    bze -= 6
    bye -= 6
    bxe -= 6
    bzo -= 6
    byo -= 6
    bxo -= 6
    --bzs
    --bys
    --bxs

    # Function Body #
    dgdx = 0.
    xl = 20.
    dxldx = 0.
    sps = sin(*ps)
# Computing 2nd power #
    d__1 = *y
# Computing 2nd power #
    d__2 = *z__
    rho2 = d__1 * d__1 + d__2 * d__2
    rho = sqrt(rho2)
    if (*y == 0. && *z__ == 0.) {
	phi = 0.
	cphi = 1.
	sphi = 0.
    } else {
	phi = atan2(*z__, *y)
	cphi = *y / rho
	sphi = *z__ / rho
    }
# Computing 2nd power #
    d__1 = rho2
# Computing 4th power #
    d__2 = xl, d__2 *= d__2
    rr4l4 = rho / (d__1 * d__1 + d__2 * d__2)
    f = phi + g_1.g * rho2 * rr4l4 * cphi * sps + g_1.tw * (*x / 10.)
    dfdphi = 1. - g_1.g * rho2 * rr4l4 * sphi * sps
# Computing 2nd power #
    d__1 = rr4l4
# Computing 4th power #
    d__2 = xl, d__2 *= d__2
# Computing 2nd power #
    d__3 = rho2
    dfdrho = g_1.g * (d__1 * d__1) * (d__2 * d__2 * 3. - d__3 * d__3) * cphi *
	     sps
# Computing 3rd power #
    d__1 = xl
    dfdx = rr4l4 * cphi * sps * (dgdx * rho2 - g_1.g * rho * rr4l4 * 4. * (
	    d__1 * (d__1 * d__1)) * dxldx) + g_1.tw / 10.
#  THE LAST TERM DESCRIBES THE IMF-INDUCED TWIS #
    cf = cos(f)
    sf = sin(f)
    yas = rho * cf
    zas = rho * sf
    unwarped_(x, &yas, &zas, bx_ass__, by_ass__, bz_ass__, bx_aso__, by_aso__,
	     bz_aso__, bx_ase__, by_ase__, bz_ase__)

    for (k = 1 k <= 5 ++k) {
# ------------------------------------------- Deforming symmetric modules #
	brho_as__ = by_ass__[k - 1] * cf + bz_ass__[k - 1] * sf
	bphi_as__ = -by_ass__[k - 1] * sf + bz_ass__[k - 1] * cf
	brho_s__ = brho_as__ * dfdphi
	bphi_s__ = bphi_as__ - rho * (bx_ass__[k - 1] * dfdx + brho_as__ * 
		dfdrho)
	bxs[k] = bx_ass__[k - 1] * dfdphi
	bys[k] = brho_s__ * cphi - bphi_s__ * sphi
	bzs[k] = brho_s__ * sphi + bphi_s__ * cphi
# L11: #
    }
    for (k = 1 k <= 5 ++k) {
	for (l = 1 l <= 4 ++l) {
# -------------------------------------------- Deforming odd modules #
	    brho_as__ = by_aso__[k + l * 5 - 6] * cf + bz_aso__[k + l * 5 - 6]
		     * sf
	    bphi_as__ = -by_aso__[k + l * 5 - 6] * sf + bz_aso__[k + l * 5 - 
		    6] * cf
	    brho_s__ = brho_as__ * dfdphi
	    bphi_s__ = bphi_as__ - rho * (bx_aso__[k + l * 5 - 6] * dfdx + 
		    brho_as__ * dfdrho)
	    bxo[k + l * 5] = bx_aso__[k + l * 5 - 6] * dfdphi
	    byo[k + l * 5] = brho_s__ * cphi - bphi_s__ * sphi
	    bzo[k + l * 5] = brho_s__ * sphi + bphi_s__ * cphi
# ------------------------------------------- Deforming even modules #
	    brho_as__ = by_ase__[k + l * 5 - 6] * cf + bz_ase__[k + l * 5 - 6]
		     * sf
	    bphi_as__ = -by_ase__[k + l * 5 - 6] * sf + bz_ase__[k + l * 5 - 
		    6] * cf
	    brho_s__ = brho_as__ * dfdphi
	    bphi_s__ = bphi_as__ - rho * (bx_ase__[k + l * 5 - 6] * dfdx + 
		    brho_as__ * dfdrho)
	    bxe[k + l * 5] = bx_ase__[k + l * 5 - 6] * dfdphi
	    bye[k + l * 5] = brho_s__ * cphi - bphi_s__ * sphi
	    bze[k + l * 5] = brho_s__ * sphi + bphi_s__ * cphi
# L13: #
	}
# L12: #
    }
    return 0
} # warped_ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. #
# Subroutine # int unwarped_(doublereal *x, doublereal *y, doublereal *z__, 
	doublereal *bxs, doublereal *bys, doublereal *bzs, doublereal *bxo, 
	doublereal *byo, doublereal *bzo, doublereal *bxe, doublereal *bye, 
	doublereal *bze)
{
    # Builtin functions #
    // double sqrt(doublereal)

    # Local variables #
    static integer k, l, m
    extern # Subroutine # int tailsht_s__(integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *), tailsht_oe__(integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *), 
	    shtbnorm_e__(integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *), 
	    shtbnorm_o__(integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *), 
	    shtbnorm_s__(integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *)
    static doublereal ajm[6], rho, ajmd[6], bxsk, bysk, bzsk, rkmr, hxsk, 
	    hysk, hzsk, rnot, bxekl, byekl, bzekl, hxekl, hyekl, hzekl, bxokl,
	     byokl, bzokl, hxokl, hyokl, hzokl
    extern # Subroutine # int bessjj_(integer *, doublereal *, doublereal *)
	    

#    CALCULATES GSM COMPONENTS OF THE SHIELDED FIELD OF 45 TAIL MODES WITH UNIT #
#    AMPLITUDES,  WITHOUT ANY WARPING OR BENDING.  NONLINEAR PARAMETERS OF THE MODES #
#    ARE FORWARDED HERE VIA A COMMON BLOCK /TAIL/. #


    # Parameter adjustments #
    bze -= 6
    bye -= 6
    bxe -= 6
    bzo -= 6
    byo -= 6
    bxo -= 6
    --bzs
    --bys
    --bxs

    # Function Body #
    rnot = 20.f

# --- New tail structure ------------- #


#    Rho_0 - scale parameter along the tail axis #
    for (k = 1 k <= 5 ++k) {
	rho = sqrt(*x * *x + *y * *y)
	rkmr = (doublereal) k * rho / rnot
#     July 2017, G.K.Stephens, all the Bessel functions are now evaluated first, #
#     and passed into the subroutines #
	bessjj_(&c__5, &rkmr, ajm)
# !! get all n in one call #
	for (m = 1 m <= 5 ++m) {
	    ajmd[m] = ajm[m - 1] - m * ajm[m] / rkmr
# L3: #
	}
	ajmd[0] = -ajm[1]
	tailsht_s__(&k, x, y, z__, ajm, &bxsk, &bysk, &bzsk)
	shtbnorm_s__(&k, x, y, z__, &hxsk, &hysk, &hzsk)
	bxs[k] = bxsk + hxsk
	bys[k] = bysk + hysk
	bzs[k] = bzsk + hzsk
	for (l = 1 l <= 4 ++l) {
	    tailsht_oe__(&c__1, &k, &l, x, y, z__, ajm, ajmd, &bxokl, &byokl, 
		    &bzokl)
	    shtbnorm_o__(&k, &l, x, y, z__, &hxokl, &hyokl, &hzokl)
	    bxo[k + l * 5] = bxokl + hxokl
	    byo[k + l * 5] = byokl + hyokl
	    bzo[k + l * 5] = bzokl + hzokl
	    tailsht_oe__(&c__0, &k, &l, x, y, z__, ajm, ajmd, &bxekl, &byekl, 
		    &bzekl)
	    shtbnorm_e__(&k, &l, x, y, z__, &hxekl, &hyekl, &hzekl)
	    bxe[k + l * 5] = bxekl + hxekl
	    bye[k + l * 5] = byekl + hyekl
	    bze[k + l * 5] = bzekl + hzekl
# L13: #
	}
# L11: #
    }
    return 0
} # unwarped_ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. The Bessel function #
#     values are now precomputed and  passed into this rather than computed #
#     inside this routine. #
# Subroutine # int tailsht_s__(integer *m, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *ajm, doublereal *bx, doublereal *by, 
	doublereal *bz)
{
    # Builtin functions #
    // double sqrt(doublereal), exp(doublereal)

    # Local variables #
    static doublereal zd, rj0, rj1, dkm, rho, rkm, rex, dltk, rkmr, rnot, 
	    rkmz, csphi, snphi



# THE COMMON BLOCKS FORWARDS TAIL SHEET THICKNESS #
# ----------------------------------------------------------------------------------- #

    rnot = 20.f
#    This can be replaced by introducing them #
    dltk = 1.f
#    through the above common block #
    rho = sqrt(*x * *x + *y * *y)
    csphi = *x / rho
    snphi = *y / rho

    dkm = (*m - 1) * dltk + 1.
    rkm = dkm / rnot

    rkmz = rkm * *z__
    rkmr = rkm * rho
    zd = sqrt(*z__ * *z__ + tail_1.d__ * tail_1.d__)

    rj0 = ajm[0]
    rj1 = ajm[1]
#     July 2017, G.K.Stephens, Bessel functions are now passed in. #
#      RJ0=bessj0(RKMR) #
#      RJ1=bessj1(RKMR) #
    rex = exp(rkm * zd)

    *bx = rkmz * rj1 * csphi / zd / rex
    *by = rkmz * rj1 * snphi / zd / rex
    *bz = rkm * rj0 / rex

#    CALCULATION OF THE MAGNETOTAIL CURRENT CONTRIBUTION IS FINISHED #

    return 0
} # tailsht_s__ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. #
# Subroutine # int shtbnorm_s__(integer *k, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *fx, doublereal *fy, doublereal *fz)
{
    # System generated locals #
    doublereal d__1

    # Builtin functions #
    // double atan2(doublereal, doublereal), sqrt(doublereal), cosh(doublereal), 
	    //sinh(doublereal), cos(doublereal), sin(doublereal)

    # Local variables #
    static integer l, m, n
    static doublereal ak[5], hx, hy, hz, hx1, hx2, hy1, hy2, ajm[15], akn, 
	    cmp, phi, chz, rho, smp, shz, ajmd[15], aknr, dpdx, dpdy, rhoi, 
	    aknri
    extern # Subroutine # int bessjj_(integer *, doublereal *, doublereal *)
	    

# modified SHTBNORM_S #
    ak[0] = tss_1.tss[*k * 80 - 5]
    ak[1] = tss_1.tss[*k * 80 - 4]
    ak[2] = tss_1.tss[*k * 80 - 3]
    ak[3] = tss_1.tss[*k * 80 - 2]
    ak[4] = tss_1.tss[*k * 80 - 1]
    phi = atan2(*y, *x)
    rho = sqrt(*x * *x + *y * *y)
    if (rho < 1e-8) {
	rhoi = 1e8
    } else {
	rhoi = 1. / rho
    }
    dpdx = -(*y) * rhoi * rhoi
    dpdy = *x * rhoi * rhoi
    *fx = 0.
    *fy = 0.
    *fz = 0.
    for (n = 1 n <= 5 ++n) {
	akn = (d__1 = ak[n - 1], abs(d__1))
	aknr = akn * rho
	if (aknr < 1e-8) {
	    aknri = 1e8
	} else {
	    aknri = 1. / aknr
	}
	chz = cosh(*z__ * akn)
	shz = sinh(*z__ * akn)
	bessjj_(&c__14, &aknr, ajm)
# !! get all n in one call #
	for (m = 1 m <= 14 ++m) {
	    ajmd[m] = ajm[m - 1] - m * ajm[m] * aknri
# L3: #
	}
	ajmd[0] = -ajm[1]
	for (m = 0 m <= 14 ++m) {
	    cmp = cos(m * phi)
	    smp = sin(m * phi)
	    hx1 = m * dpdx * smp * shz * ajm[m]
	    hx2 = -akn * *x * rhoi * cmp * shz * ajmd[m]
	    hx = hx1 + hx2
	    hy1 = m * dpdy * smp * shz * ajm[m]
	    hy2 = -akn * *y * rhoi * cmp * shz * ajmd[m]
	    hy = hy1 + hy2
	    hz = -akn * cmp * chz * ajm[m]
	    l = n + m * 5
	    *fx += hx * tss_1.tss[l + *k * 80 - 81]
	    *fy += hy * tss_1.tss[l + *k * 80 - 81]
	    *fz += hz * tss_1.tss[l + *k * 80 - 81]
# L4: #
	}
# L2: #
    }
    return 0
} # shtbnorm_s__ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. The Bessel function #
#     values are now precomputed and  passed into this rather than computed #
#     inside this routine. #
# Subroutine # int tailsht_oe__(integer *ievo, integer *mk, integer *m, 
	doublereal *x, doublereal *y, doublereal *z__, doublereal *ajm, 
	doublereal *ajmd, doublereal *bx, doublereal *by, doublereal *bz)
{
    # Builtin functions #
    //double sqrt(doublereal), atan2(doublereal, doublereal), cos(doublereal), 
	    //sin(doublereal), exp(doublereal)

    # Local variables #
    static doublereal zd, dkm, phi, bro, rho, rkm, rex, bphi, dltk, rkmr, 
	    rnot, rkmz, csphi, snphi, csmphi, snmphi



# THE COMMON BLOCKS FORWARDS TAIL SHEET THICKNES #
# ----------------------------------------------------------------------------------- #

    rnot = 20.f
#    Rho_0 - scale parameter along the tail axis #
    dltk = 1.f

# ----------------------------------------------------------------------------------- #
#    step in Km #
    rho = sqrt(*x * *x + *y * *y)

    csphi = *x / rho
    snphi = *y / rho

    phi = atan2(*y, *x)
    csmphi = cos(*m * phi)
    snmphi = sin(*m * phi)

    dkm = (*mk - 1) * dltk + 1.
    rkm = dkm / rnot

    rkmz = rkm * *z__
    rkmr = rkm * rho

    zd = sqrt(*z__ * *z__ + tail_2.d0 * tail_2.d0)

    rex = exp(rkm * zd)

#     July 2017, G.K.Stephens, Jm is now passed in, not computed internally #
# ---- calculating Jm and its derivatives ------ #

#                   if(m.gt.2) then #
#                   AJM=bessj(m,RKMR) #
#                   AJM1=bessj(m-1,RKMR) #
#                   AJMD=AJM1-m*AJM/RKMR #
#                   else #
#                  -------------------- #
#                   if(m.eq.2) then #
#                   AJM=bessj(2,RKMR) #
#                   AJM1=bessj1(RKMR) #
#                   AJMD=AJM1-m*AJM/RKMR #
#                   else #
#                  -------------------- #
#                   AJM=bessj1(RKMR) #
#                   AJM1=bessj0(RKMR) #
#                   AJMD=AJM1-AJM/RKMR #
#                  -------------------- #
#                   end if #
#                  -------------------- #
#                   endif #
# ----------------------------------------- #

    if (*ievo == 0) {
# ----------------------------------------- #
# calculating symmetric modes #
# ----------------------------------------- #

	bro = -(*m) * snmphi * *z__ * ajmd[*m] / zd / rex
	bphi = -(*m) * *m * csmphi * *z__ * ajm[*m] / rkmr / zd / rex
	*bz = *m * snmphi * ajm[*m] / rex

# ----------------------------------------- #
    } else {
# ----------------------------------------- #
# calculating asymmetric modes #
# ----------------------------------------- #

	bro = *m * csmphi * *z__ * ajmd[*m] / zd / rex
	bphi = -(*m) * *m * snmphi * *z__ * ajm[*m] / rkmr / zd / rex
	*bz = -(*m) * csmphi * ajm[*m] / rex

# ----------------------------------------- #
    }

# --- transformation from cylindrical ccordinates to GSM --- #

    *bx = bro * csphi - bphi * snphi
    *by = bro * snphi + bphi * csphi

#    CALCULATION OF THE MAGNETOTAIL CURRENT CONTRIBUTION IS FINISHED #

    return 0
} # tailsht_oe__ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. #
# Subroutine # int shtbnorm_o__(integer *k, integer *l, doublereal *x, 
	doublereal *y, doublereal *z__, doublereal *fx, doublereal *fy, 
	doublereal *fz)
{
    # System generated locals #
    doublereal d__1

    # Builtin functions #
    // double atan2(doublereal, doublereal), sqrt(doublereal), cosh(doublereal), 
	    //sinh(doublereal), cos(doublereal), sin(doublereal)

    # Local variables #
    static integer m, n, l1
    static doublereal ak[5], hx, hy, hz, hx1, hx2, hy1, hy2, ajm[15], akn, 
	    cmp, phi, chz, rho, smp, shz, ajmd[15], aknr, dpdx, dpdy, rhoi, 
	    aknri
    extern # Subroutine # int bessjj_(integer *, doublereal *, doublereal *)
	    

# modified SHTBNORM_O #
    ak[0] = tso_1.tso[(*k + *l * 5) * 80 - 405]
    ak[1] = tso_1.tso[(*k + *l * 5) * 80 - 404]
    ak[2] = tso_1.tso[(*k + *l * 5) * 80 - 403]
    ak[3] = tso_1.tso[(*k + *l * 5) * 80 - 402]
    ak[4] = tso_1.tso[(*k + *l * 5) * 80 - 401]
    phi = atan2(*y, *x)
    rho = sqrt(*x * *x + *y * *y)
    if (rho < 1e-8) {
	rhoi = 1e8
    } else {
	rhoi = 1. / rho
    }
    dpdx = -(*y) * rhoi * rhoi
    dpdy = *x * rhoi * rhoi
    *fx = 0.
    *fy = 0.
    *fz = 0.
    for (n = 1 n <= 5 ++n) {
	akn = (d__1 = ak[n - 1], abs(d__1))
	aknr = akn * rho
	if (aknr < 1e-8) {
	    aknri = 1e8
	} else {
	    aknri = 1. / aknr
	}
	chz = cosh(*z__ * akn)
	shz = sinh(*z__ * akn)
	bessjj_(&c__14, &aknr, ajm)
# !! get all n in one call #
	for (m = 1 m <= 14 ++m) {
	    ajmd[m] = ajm[m - 1] - m * ajm[m] * aknri
# L3: #
	}
	ajmd[0] = -ajm[1]
	for (m = 0 m <= 14 ++m) {
	    cmp = cos(m * phi)
	    smp = sin(m * phi)
	    hx1 = m * dpdx * smp * shz * ajm[m]
	    hx2 = -akn * *x * rhoi * cmp * shz * ajmd[m]
	    hx = hx1 + hx2
	    hy1 = m * dpdy * smp * shz * ajm[m]
	    hy2 = -akn * *y * rhoi * cmp * shz * ajmd[m]
	    hy = hy1 + hy2
	    hz = -akn * cmp * chz * ajm[m]
	    l1 = n + m * 5
	    *fx += hx * tso_1.tso[l1 + (*k + *l * 5) * 80 - 481]
	    *fy += hy * tso_1.tso[l1 + (*k + *l * 5) * 80 - 481]
	    *fz += hz * tso_1.tso[l1 + (*k + *l * 5) * 80 - 481]
# L4: #
	}
# L2: #
    }
    return 0
} # shtbnorm_o__ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. #
# Subroutine # int shtbnorm_e__(integer *k, integer *l, doublereal *x, 
	doublereal *y, doublereal *z__, doublereal *fx, doublereal *fy, 
	doublereal *fz)
{
    # System generated locals #
    doublereal d__1

    # Builtin functions #
    //double atan2(doublereal, doublereal), sqrt(doublereal), cosh(doublereal), 
	    //sinh(doublereal), cos(doublereal), sin(doublereal)

    # Local variables #
    static integer m, n, l1
    static doublereal ak[5], hx, hy, hz, hx1, hx2, hy1, hy2, ajm[15], akn, 
	    cmp, phi, chz, rho, smp, shz, ajmd[15], aknr, dpdx, dpdy, rhoi, 
	    aknri
    extern # Subroutine # int bessjj_(integer *, doublereal *, doublereal *)
	    

# modified SHTBNORM_E #
    ak[0] = tse_1.tse[(*k + *l * 5) * 80 - 405]
    ak[1] = tse_1.tse[(*k + *l * 5) * 80 - 404]
    ak[2] = tse_1.tse[(*k + *l * 5) * 80 - 403]
    ak[3] = tse_1.tse[(*k + *l * 5) * 80 - 402]
    ak[4] = tse_1.tse[(*k + *l * 5) * 80 - 401]
    phi = atan2(*y, *x)
    rho = sqrt(*x * *x + *y * *y)
    if (rho < 1e-8) {
	rhoi = 1e8
    } else {
	rhoi = 1. / rho
    }
    dpdx = -(*y) * rhoi * rhoi
    dpdy = *x * rhoi * rhoi
    *fx = 0.
    *fy = 0.
    *fz = 0.
    for (n = 1 n <= 5 ++n) {
	akn = (d__1 = ak[n - 1], abs(d__1))
	aknr = akn * rho
	if (aknr < 1e-8) {
	    aknri = 1e8
	} else {
	    aknri = 1. / aknr
	}
	chz = cosh(*z__ * akn)
	shz = sinh(*z__ * akn)
	bessjj_(&c__14, &aknr, ajm)
# !! get all n in one call #
	for (m = 1 m <= 14 ++m) {
	    ajmd[m] = ajm[m - 1] - m * ajm[m] * aknri
# L3: #
	}
	ajmd[0] = -ajm[1]
	for (m = 0 m <= 14 ++m) {
	    cmp = cos(m * phi)
	    smp = sin(m * phi)
	    hx1 = -m * dpdx * cmp * shz * ajm[m]
	    hx2 = -akn * *x * rhoi * smp * shz * ajmd[m]
	    hx = hx1 + hx2
	    hy1 = -m * dpdy * cmp * shz * ajm[m]
	    hy2 = -akn * *y * rhoi * smp * shz * ajmd[m]
	    hy = hy1 + hy2
	    hz = -akn * smp * chz * ajm[m]
	    l1 = n + m * 5
	    *fx += hx * tse_1.tse[l1 + (*k + *l * 5) * 80 - 481]
	    *fy += hy * tse_1.tse[l1 + (*k + *l * 5) * 80 - 481]
	    *fz += hz * tse_1.tse[l1 + (*k + *l * 5) * 80 - 481]
# L4: #
	}
# L2: #
    }
    return 0
} # shtbnorm_e__ #

# ========================================================================== #
#     July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's #
#     improvements to the Bessel function evaluation. #
# Subroutine # int bessjj_(integer *n, doublereal *x, doublereal *bessj)
{
    # System generated locals #
    integer i__1

    # Builtin functions #
    //double sqrt(doublereal)

    # Local variables #
    static integer i__, j, m
    static doublereal bj, ax, bjm, bjp, tox, bnorm
    static logical iseven
    static doublereal evnsum

# bessJ holds J0 to Jn #
    ax = abs(*x)
    tox = 2. / ax
#     start at some large m, larger than the desired n, multiply by 2 to ensure #
#     m starts at an even number #
    m = (*n + (integer) sqrt((doublereal) (*n * 40))) / 2 << 1
    evnsum = 0.
# keeps track of the sum of the even Js (J0+J2+J4+...) #
    iseven = false
#     we set the value of Jm to some arbitrary value, here Jm=1, after the loop #
#     is done, the values will be normalized using the sum #
    bjp = 0.
    bj = 1.
#     initialize to zero #
    i__1 = *n
    for (i__ = 0 i__ <= i__1 ++i__) {
	bessj[i__] = 0.f
    }
    for (j = m j >= 1 --j) {
#     the next value int the recursion relation J_n-1 = (2*n/x)*Jn - J_n+1 #
	bjm = j * tox * bj - bjp
	bjp = bj
# decrement so shift J_n+1 ot Jn #
	bj = bjm
#     if the value gets too large, shift the decimal of everything by 10 places #
# decrement so shift J_n ot J_n-1 #
	if (abs(bj) > 1e10) {
	    bj *= 1e-10
	    bjp *= 1e-10
	    evnsum *= 1e-10
	    i__1 = *n
	    for (i__ = j + 1 i__ <= i__1 ++i__) {
		bessj[i__] *= 1e-10
	    }
	}
	if (iseven) {
	    evnsum += bj
	}
# only sum over the even Jns #
	iseven = ! iseven
	if (j <= *n) {
	    bessj[j] = bjp
	}
# Jj(x) #
# L12: #
    }
#     sum is currently the sum of all the evens #
#     use Miller's algorithm for Bessel functions which uses the identity: #
#     1.0 = 2.0*sum(J_evens) - J0, thus the quantity (2.0*sum(J_evens) - J0) #
#     is used as a normalization factor #
    bnorm = evnsum * 2. - bj
#     normalize the Bessel functions #
    i__1 = *n
    for (i__ = 1 i__ <= i__1 ++i__) {
	bessj[i__] /= bnorm
    }
    bessj[0] = bj / bnorm
#     Apply Jn(-x) = (-1)^n * Jn(x) #
# J0(x) #
    if (*x < 0.) {
	i__1 = *n
	for (i__ = 1 i__ <= i__1 i__ += 2) {
	    bessj[i__] = -bessj[i__]
	}
    }
    return 0
} # bessjj_ #

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Subroutine # int birk_tot__(doublereal *ps, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bx11, doublereal *by11, doublereal *bz11,
	 doublereal *bx12, doublereal *by12, doublereal *bz12, doublereal *
	bx21, doublereal *by21, doublereal *bz21, doublereal *bx22, 
	doublereal *by22, doublereal *bz22)
{
    # Initialized data #

    static doublereal sh11[86] = { 46488.84663,-15541.95244,-23210.09824,
	    -32625.03856,-109894.4551,-71415.32808,58168.94612,55564.87578,
	    -22890.60626,-6056.763968,5091.3681,239.7001538,-13899.49253,
	    4648.016991,6971.310672,9699.351891,32633.34599,21028.48811,
	    -17395.9619,-16461.11037,7447.621471,2528.844345,-1934.094784,
	    -588.3108359,-32588.88216,10894.11453,16238.25044,22925.60557,
	    77251.11274,50375.97787,-40763.78048,-39088.6066,15546.53559,
	    3559.617561,-3187.730438,309.1487975,88.22153914,-243.0721938,
	    -63.63543051,191.1109142,69.94451996,-187.9539415,-49.89923833,
	    104.0902848,-120.2459738,253.5572433,89.25456949,-205.6516252,
	    -44.93654156,124.7026309,32.53005523,-98.85321751,-36.51904756,
	    98.8824169,24.88493459,-55.04058524,61.14493565,-128.4224895,
	    -45.3502346,105.0548704,-43.66748755,119.3284161,31.38442798,
	    -92.87946767,-33.52716686,89.98992001,25.87341323,-48.86305045,
	    59.69362881,-126.5353789,-44.39474251,101.5196856,59.41537992,
	    41.18892281,80.861012,3.066809418,7.893523804,30.56212082,
	    10.36861082,8.222335945,19.97575641,2.050148531,4.992657093,
	    2.300564232,.2256245602,-.05841594319 }
    static doublereal sh12[86] = { 210260.4816,-1443587.401,-1468919.281,
	    281939.2993,-1131124.839,729331.7943,2573541.307,304616.7457,
	    468887.5847,181554.7517,-1300722.65,-257012.8601,645888.8041,
	    -2048126.412,-2529093.041,571093.7972,-2115508.353,1122035.951,
	    4489168.802,75234.22743,823905.6909,147926.6121,-2276322.876,
	    -155528.5992,-858076.2979,3474422.388,3986279.931,-834613.9747,
	    3250625.781,-1818680.377,-7040468.986,-414359.6073,-1295117.666,
	    -346320.6487,3565527.409,430091.9496,-.1565573462,7.377619826,
	    .4115646037,-6.14607888,3.808028815,-.5232034932,1.454841807,
	    -12.32274869,-4.466974237,-2.941184626,-.6172620658,12.6461349,
	    1.494922012,-21.35489898,-1.65225696,16.81799898,-1.404079922,
	    -24.09369677,-10.99900839,45.9423782,2.248579894,31.91234041,
	    7.575026816,-45.80833339,-1.507664976,14.60016998,1.348516288,
	    -11.05980247,-5.402866968,31.69094514,12.28261196,-37.55354174,
	    4.155626879,-33.70159657,-8.437907434,36.22672602,145.0262164,
	    70.73187036,85.51110098,21.47490989,24.34554406,31.34405345,
	    4.655207476,5.747889264,7.802304187,1.844169801,4.86725455,
	    2.941393119,.1379899178,.06607020029 }
    static doublereal sh21[86] = { 162294.6224,503885.1125,-27057.67122,
	    -531450.1339,84747.05678,-237142.1712,84133.6149,259530.0402,
	    69196.0516,-189093.5264,-19278.55134,195724.5034,-263082.6367,
	    -818899.6923,43061.10073,863506.6932,-139707.9428,389984.885,
	    -135167.5555,-426286.9206,-109504.0387,295258.3531,30415.07087,
	    -305502.9405,100785.34,315010.9567,-15999.50673,-332052.2548,
	    54964.34639,-152808.375,51024.67566,166720.0603,40389.67945,
	    -106257.7272,-11126.14442,109876.2047,2.978695024,558.6019011,
	    2.685592939,-338.000473,-81.9972409,-444.1102659,89.44617716,
	    212.0849592,-32.58562625,-982.7336105,-35.10860935,567.8931751,
	    -1.917212423,-260.2023543,-1.023821735,157.5533477,23.00200055,
	    232.0603673,-36.79100036,-111.9110936,18.05429984,447.0481,
	    15.10187415,-258.7297813,-1.032340149,-298.6402478,-1.676201415,
	    180.5856487,64.52313024,209.0160857,-53.8557401,-98.5216429,
	    14.35891214,536.7666279,20.09318806,-309.734953,58.54144539,
	    67.4522685,97.92374406,4.75244976,10.46824379,32.9185611,
	    12.05124381,9.962933904,15.91258637,1.804233877,6.578149088,
	    2.515223491,.1930034238,-.02261109942 }
    static doublereal sh22[86] = { -131287.8986,-631927.6885,-318797.4173,
	    616785.8782,-50027.36189,863099.9833,47680.2024,-1053367.944,
	    -501120.3811,-174400.9476,222328.6873,333551.7374,-389338.7841,
	    -1995527.467,-982971.3024,1960434.268,297239.7137,2676525.168,
	    -147113.4775,-3358059.979,-2106979.191,-462827.1322,1017607.96,
	    1039018.475,520266.9296,2627427.473,1301981.763,-2577171.706,
	    -238071.9956,-3539781.111,94628.1642,4411304.724,2598205.733,
	    637504.9351,-1234794.298,-1372562.403,-2.646186796,-31.10055575,
	    2.295799273,19.20203279,30.01931202,-302.102855,-14.78310655,
	    162.1561899,.4943938056,176.8089129,-.244492168,-100.6148929,
	    9.172262228,137.430344,-8.451613443,-84.20684224,-167.3354083,
	    1321.830393,76.89928813,-705.7586223,18.28186732,-770.1665162,
	    -9.084224422,436.3368157,-6.374255638,-107.2730177,6.080451222,
	    65.53843753,143.2872994,-1028.009017,-64.2273933,547.8536586,
	    -20.58928632,597.3893669,10.17964133,-337.7800252,159.3532209,
	    76.34445954,84.74398828,12.76722651,27.63870691,32.69873634,
	    5.145153451,6.310949163,6.996159733,1.971629939,4.436299219,
	    2.904964304,.1486276863,.06859991529 }

    extern # Subroutine # int birk_shl__(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *)
    static doublereal fx11, fy11, fz11, hx11, hy11, hz11, fx12, fy12, fz12, 
	    hx12, hy12, hz12, fx21, fy21, fz21, hx21, hy21, hz21, fx22, fy22, 
	    fz22, hx22, hy22, hz22, x_sc__
    extern # Subroutine # int birk_1n2__(integer *, integer *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *)


#  INPUT PARAMETERS, SPECIFIED #
# PARAMETERS, CONTROL DAY #
# ====   LEAST SQUARES FITTING ONLY: #
#       BX11=0.D0 #
#       BY11=0.D0 #
#       BZ11=0.D0 #
#       BX12=0.D0 #
#       BY12=0.D0 #
#       BZ12=0.D0 #
#       BX21=0.D0 #
#       BY21=0.D0 #
#       BZ21=0.D0 #
#       BX22=0.D0 #
#       BY22=0.D0 #
#       BZ22=0.D0 #
# =================================== #
    dphi_b_rho0__1.xkappa = birkpar_1.xkappa1
#  FORWARDED IN BIRK_1N2 #
    x_sc__ = birkpar_1.xkappa1 - 1.1
#  FORWARDED IN BIRK_SHL #
    birk_1n2__(&c__1, &c__1, ps, x, y, z__, &fx11, &fy11, &fz11)
#  REGION 1, #
    birk_shl__(sh11, ps, &x_sc__, x, y, z__, &hx11, &hy11, &hz11)
    *bx11 = fx11 + hx11
    *by11 = fy11 + hy11
    *bz11 = fz11 + hz11
    birk_1n2__(&c__1, &c__2, ps, x, y, z__, &fx12, &fy12, &fz12)
#  REGION 1, #
    birk_shl__(sh12, ps, &x_sc__, x, y, z__, &hx12, &hy12, &hz12)
    *bx12 = fx12 + hx12
    *by12 = fy12 + hy12
    *bz12 = fz12 + hz12
    dphi_b_rho0__1.xkappa = birkpar_1.xkappa2
#  FORWARDED IN BIRK_1N2 #
    x_sc__ = birkpar_1.xkappa2 - 1.
#  FORWARDED IN BIRK_SHL #
    birk_1n2__(&c__2, &c__1, ps, x, y, z__, &fx21, &fy21, &fz21)
#  REGION 2, #
    birk_shl__(sh21, ps, &x_sc__, x, y, z__, &hx21, &hy21, &hz21)
    *bx21 = fx21 + hx21
    *by21 = fy21 + hy21
    *bz21 = fz21 + hz21
    birk_1n2__(&c__2, &c__2, ps, x, y, z__, &fx22, &fy22, &fz22)
#  REGION 2, #
    birk_shl__(sh22, ps, &x_sc__, x, y, z__, &hx22, &hy22, &hz22)
    *bx22 = fx22 + hx22
    *by22 = fy22 + hy22
    *bz22 = fz22 + hz22
    return 0
} # birk_tot__ #


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


# Subroutine # int birk_1n2__(integer *numb, integer *mode, doublereal *ps, 
	doublereal *x, doublereal *y, doublereal *z__, doublereal *bx, 
	doublereal *by, doublereal *bz)
{
    # Initialized data #

    static doublereal beta = .9
    static doublereal rh = 10.
    static doublereal eps = 3.
    static doublereal a11[31] = { .161806835,-.1797957553,2.999642482,
	    -.9322708978,-.681105976,.2099057262,-8.358815746,-14.8603355,
	    .3838362986,-16.30945494,4.537022847,2.685836007,27.97833029,
	    6.330871059,1.876532361,18.95619213,.96515281,.4217195118,
	    -.0895777002,-1.823555887,.7457045438,-.5785916524,-1.010200918,
	    .01112389357,.09572927448,-.3599292276,8.713700514,.9763932955,
	    3.834602998,2.492118385,.7113544659 }
    static doublereal a12[31] = { .705802694,-.2845938535,5.715471266,
	    -2.47282088,-.7738802408,.347829393,-11.37653694,-38.64768867,
	    .6932927651,-212.4017288,4.944204937,3.071270411,33.05882281,
	    7.387533799,2.366769108,79.22572682,.6154290178,.5592050551,
	    -.1796585105,-1.65493221,.7309108776,-.4926292779,-1.130266095,
	    -.009613974555,.1484586169,-.2215347198,7.883592948,.02768251655,
	    2.950280953,1.212634762,.5567714182 }
    static doublereal a21[31] = { .1278764024,-.2320034273,1.805623266,
	    -32.3724144,-.9931490648,.317508563,-2.492465814,-16.21600096,
	    .2695393416,-6.752691265,3.971794901,14.54477563,41.10158386,
	    7.91288973,1.258297372,9.583547721,1.014141963,.5104134759,
	    -.1790430468,-1.756358428,.7561986717,-.6775248254,-.0401401642,
	    .01446794851,.1200521731,-.2203584559,4.50896385,.8221623576,
	    1.77993373,1.102649543,.886788002 }
    static doublereal a22[31] = { .4036015198,-.3302974212,2.82773093,
	    -45.4440583,-1.611103927,.4927112073,-.003258457559,-49.59014949,
	    .3796217108,-233.7884098,4.31266698,18.05051709,28.95320323,
	    11.09948019,.7471649558,67.10246193,.5667096597,.6468519751,
	    -.1560665317,-1.460805289,.7719653528,-.6658988668,2.515179349e-6,
	    .02426021891,.1195003324,-.2625739255,4.377172556,.2421190547,
	    2.503482679,1.071587299,.724799743 }

    # System generated locals #
    doublereal d__1, d__2, d__3, d__4, d__5

    # Builtin functions #
    //double sqrt(doublereal), atan2(doublereal, doublereal), sin(doublereal), 
	    //cos(doublereal), pow_dd(doublereal *, doublereal *)

    # Local variables #
    static doublereal dphisphi, dphisrho
    extern # Subroutine # int twocones_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *)
    static doublereal xs, zs, phi, rho, rsc, bxs, xsc, ysc, zsc, bzs, rho2, 
	    r1rh, by_s__, byas, phis, brack, cphic, sphic, psias, bphi_s__, 
	    bphias, cphics, brho_s__, brhoas, sphics, dphisdy


#  CALCULATES COMPONENTS  OF REGION 1/2 FIELD IN SPHERICAL COORDS.  DERIVED FROM THE S/R DIPDEF2C (WHICH #
#    DOES THE SAME JOB, BUT INPUT/OUTPUT THERE WAS IN SPHERICAL COORDS, WHILE HERE WE USE CARTESIAN ONES) #

#   INPUT:  NUMB=1 (2) FOR REGION 1 (2) CURRENTS #
#           MODE=1 YIELDS SIMPLE SINUSOIDAL MLT VARIATION, WITH MAXIMUM CURRENT AT DAWN/DUSK MERIDIAN #
#     WHILE MODE=2 YIELDS THE SECOND HARMONIC. #


#   SEE N #
#  (1) DPHI:   HALF-DIFFERENCE (IN RADIANS) BETWEEN DAY AND NIGHT LATITUDE OF FAC OVAL AT IONOSPHERIC ALTITUD
E #
#              TYPICAL VALUE: 0.06 #
#  (2) B:      AN ASYMMETRY FACTOR AT HIGH-ALTITUDES  FOR B=0, THE ONLY ASYMMETRY IS THAT FROM DPHI #
#              TYPICAL VALUES: 0.35-0.70 #
#  (3) RHO_0:  A FIXED PARAMETER, DEFINING THE DISTANCE RHO, AT WHICH THE LATITUDE SHIFT GRADUALLY SATURATES 
AND #
#              STOPS INCREASING #
#              ITS VALUE WAS ASSUMED FIXED, EQUAL TO 7.0. #
#  (4) XKAPPA: AN OVERALL SCALING FACTOR, WHICH CAN BE USED FOR CHANGING THE SIZE OF THE F.A.C. OVAL #

# THESE PARAMETERS CONTRO #
# parameters of the tilt-depend #
    dphi_b_rho0__1.b = .5f
    dphi_b_rho0__1.rho_0__ = 7.f
    modenum_1.m = *mode
    if (*numb == 1) {
	dphi_b_rho0__1.dphi = .055
	dtheta_1.dtheta = .06
    }
    if (*numb == 2) {
	dphi_b_rho0__1.dphi = .03
	dtheta_1.dtheta = .09
    }
    xsc = *x * dphi_b_rho0__1.xkappa
    ysc = *y * dphi_b_rho0__1.xkappa
    zsc = *z__ * dphi_b_rho0__1.xkappa
# Computing 2nd power #
    d__1 = xsc
# Computing 2nd power #
    d__2 = zsc
    rho = sqrt(d__1 * d__1 + d__2 * d__2)
# Computing 2nd power #
    d__1 = xsc
# Computing 2nd power #
    d__2 = ysc
# Computing 2nd power #
    d__3 = zsc
    rsc = sqrt(d__1 * d__1 + d__2 * d__2 + d__3 * d__3)

# Computing 2nd power #
    d__1 = dphi_b_rho0__1.rho_0__
    rho2 = d__1 * d__1
    if (xsc == 0. && zsc == 0.) {
	phi = 0.
    } else {
	phi = atan2(-zsc, xsc)
#  FROM CARTESIAN TO CYLINDRICAL (RHO,PHI #
    }
    sphic = sin(phi)
    cphic = cos(phi)
#  "C" means "CYLINDRICAL", TO DISTINGUISH FROM S #
# Computing 2nd power #
    d__1 = rho
# Computing 2nd power #
    d__2 = rho
    brack = dphi_b_rho0__1.dphi + dphi_b_rho0__1.b * rho2 / (rho2 + 1.) * (
	    d__1 * d__1 - 1.) / (rho2 + d__2 * d__2)
    r1rh = (rsc - 1.) / rh
    d__1 = pow(r1rh, eps) + 1.
    d__2 = 1. / eps
    psias = beta * *ps / pow(d__1, d__2)
    phis = phi - brack * sin(phi) - psias
    dphisphi = 1. - brack * cos(phi)
# Computing 2nd power #
    d__2 = rho
# Computing 2nd power #
    d__1 = rho2 + d__2 * d__2
    d__3 = eps - 1.
    d__4 = pow(r1rh, eps) + 1.
    d__5 = 1. / eps + 1.
    dphisrho = dphi_b_rho0__1.b * -2. * rho2 * rho / (d__1 * d__1) * sin(phi) 
	    + beta * *ps * pow(r1rh, d__3) * rho / (rh * rsc * pow(
	    d__4, d__5))
    d__1 = eps - 1.
    d__2 = pow(r1rh, eps) + 1.
    d__3 = 1. / eps + 1.
    dphisdy = beta * *ps * pow(r1rh, d__1) * ysc / (rh * rsc * pow(
	    d__2, d__3))
    sphics = sin(phis)
    cphics = cos(phis)
    xs = rho * cphics
    zs = -rho * sphics
    if (*numb == 1) {
	if (*mode == 1) {
	    twocones_(a11, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
	if (*mode == 2) {
	    twocones_(a12, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
    } else {
	if (*mode == 1) {
	    twocones_(a21, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
	if (*mode == 2) {
	    twocones_(a22, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
    }
    brhoas = bxs * cphics - bzs * sphics
    bphias = -bxs * sphics - bzs * cphics
    brho_s__ = brhoas * dphisphi * dphi_b_rho0__1.xkappa
    bphi_s__ = (bphias - rho * (byas * dphisdy + brhoas * dphisrho)) * 
	    dphi_b_rho0__1.xkappa
    by_s__ = byas * dphisphi * dphi_b_rho0__1.xkappa
    *bx = brho_s__ * cphic - bphi_s__ * sphic
    *by = by_s__
    *bz = -brho_s__ * sphic - bphi_s__ * cphic
    return 0
} # birk_1n2__ #


# ========================================================================= #

# Subroutine # int twocones_(doublereal *a, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bx, doublereal *by, doublereal *bz)
{
    # System generated locals #
    doublereal d__1, d__2

    # Local variables #
    extern # Subroutine # int one_cone__(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *)
    static doublereal bxn, byn, bzn, bxs, bys, bzs


#   ADDS FIELDS FROM TWO CONES (NORTHERN AND SOUTHERN), WITH A PROPER SYMMETRY OF THE CURRENT AND FIELD, #
#     CORRESPONDING TO THE REGION 1 BIRKELAND CURRENTS. (SEE NB #6, P.58). #

    # Parameter adjustments #
    --a

    # Function Body #
    one_cone__(&a[1], x, y, z__, &bxn, &byn, &bzn)
    d__1 = -(*y)
    d__2 = -(*z__)
    one_cone__(&a[1], x, &d__1, &d__2, &bxs, &bys, &bzs)
    *bx = bxn - bxs
    *by = byn + bys
    *bz = bzn + bzs
    return 0
} # twocones_ #


# ------------------------------------------------------------------------- #

# Subroutine # int one_cone__(doublereal *a, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bx, doublereal *by, doublereal *bz)
{
    # Initialized data #

    static doublereal dr = 1e-6
    static doublereal dt = 1e-6

    # System generated locals #
    doublereal d__1, d__2

    # Builtin functions #
    //double sqrt(doublereal), atan2(doublereal, doublereal), sin(doublereal)

    # Local variables #
    static doublereal c__, r__, s, be, cf, br, sf, rs, phi
    extern doublereal r_s__(doublereal *, doublereal *, doublereal *)
    static doublereal rho, rsr, rho2, bphi, phis, bfast, theta, btast, drsdr, 
	    drsdt, dtsdr, dtsdt, stsst, theta0, btheta, thetas
    extern # Subroutine # int fialcos_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *, doublereal *,
	     doublereal *)
    extern doublereal theta_s__(doublereal *, doublereal *, doublereal *)


#  RETURNS FIELD COMPONENTS FOR A DEFORMED CONICAL CURRENT SYSTEM, FITTED TO A BIOSAVART FIELD #
#    BY SIM_14.FOR.  HERE ONLY THE NORTHERN CONE IS TAKEN INTO ACCOUNT. #

    # Parameter adjustments #
    --a

    # Function Body #
#   JUST FOR NUMERICAL DIFFERENTIATION #
    theta0 = a[31]
# Computing 2nd power #
    d__1 = *x
# Computing 2nd power #
    d__2 = *y
    rho2 = d__1 * d__1 + d__2 * d__2
    rho = sqrt(rho2)
# Computing 2nd power #
    d__1 = *z__
    r__ = sqrt(rho2 + d__1 * d__1)
    theta = atan2(rho, *z__)
    phi = atan2(*y, *x)

#   MAKE THE DEFORMATION OF COORDINATES: #

    rs = r_s__(&a[1], &r__, &theta)
    thetas = theta_s__(&a[1], &r__, &theta)
    phis = phi

#   CALCULATE FIELD COMPONENTS AT THE NEW POSITION (ASTERISKED): #

    fialcos_(&rs, &thetas, &phis, &btast, &bfast, &modenum_1.m, &theta0, &
	    dtheta_1.dtheta)

#   NOW TRANSFORM B{R,T,F}_AST BY THE DEFORMATION TENSOR: #

#      FIRST OF ALL, FIND THE DERIVATIVES: #


    d__1 = r__ + dr
    d__2 = r__ - dr
    drsdr = (r_s__(&a[1], &d__1, &theta) - r_s__(&a[1], &d__2, &theta)) / (dr 
	    * 2.)
    d__1 = theta + dt
    d__2 = theta - dt
    drsdt = (r_s__(&a[1], &r__, &d__1) - r_s__(&a[1], &r__, &d__2)) / (dt * 
	    2.)
    d__1 = r__ + dr
    d__2 = r__ - dr
    dtsdr = (theta_s__(&a[1], &d__1, &theta) - theta_s__(&a[1], &d__2, &theta)
	    ) / (dr * 2.)
    d__1 = theta + dt
    d__2 = theta - dt
    dtsdt = (theta_s__(&a[1], &r__, &d__1) - theta_s__(&a[1], &r__, &d__2)) / 
	    (dt * 2.)
    stsst = sin(thetas) / sin(theta)
    rsr = rs / r__
    br = -rsr / r__ * stsst * btast * drsdt
#   NB#6, P.43 #
    btheta = rsr * stsst * btast * drsdr
#               ( #
    bphi = rsr * bfast * (drsdr * dtsdt - drsdt * dtsdr)
    s = rho / r__
    c__ = *z__ / r__
    sf = *y / rho
    cf = *x / rho
    be = br * s + btheta * c__
    *bx = a[1] * (be * cf - bphi * sf)
    *by = a[1] * (be * sf + bphi * cf)
    *bz = a[1] * (br * c__ - btheta * s)
    return 0
} # one_cone__ #


# ===================================================================================== #
doublereal r_s__(doublereal *a, doublereal *r__, doublereal *theta)
{
    # System generated locals #
    doublereal ret_val, d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9, 
	    d__10, d__11, d__12, d__13

    # Builtin functions #
    //double sqrt(doublereal), cos(doublereal)


    # Parameter adjustments #
    --a

    # Function Body #
# Computing 2nd power #
    d__1 = *r__
# Computing 2nd power #
    d__2 = a[11]
# Computing 2nd power #
    d__3 = *r__
# Computing 2nd power #
    d__4 = a[12]
# Computing 2nd power #
    d__5 = *r__
# Computing 2nd power #
    d__6 = a[13]
# Computing 2nd power #
    d__7 = *r__
# Computing 2nd power #
    d__8 = a[14]
# Computing 2nd power #
    d__9 = *r__
# Computing 2nd power #
    d__10 = a[15]
# Computing 2nd power #
    d__12 = *r__
# Computing 2nd power #
    d__13 = a[16]
# Computing 2nd power #
    d__11 = d__12 * d__12 + d__13 * d__13
    ret_val = *r__ + a[2] / *r__ + a[3] * *r__ / sqrt(d__1 * d__1 + d__2 * 
	    d__2) + a[4] * *r__ / (d__3 * d__3 + d__4 * d__4) + (a[5] + a[6] /
	     *r__ + a[7] * *r__ / sqrt(d__5 * d__5 + d__6 * d__6) + a[8] * *
	    r__ / (d__7 * d__7 + d__8 * d__8)) * cos(*theta) + (a[9] * *r__ / 
	    sqrt(d__9 * d__9 + d__10 * d__10) + a[10] * *r__ / (d__11 * d__11)
	    ) * cos(*theta * 2.)

    return ret_val
} # r_s__ #


# ----------------------------------------------------------------------------- #

doublereal theta_s__(doublereal *a, doublereal *r__, doublereal *theta)
{
    # System generated locals #
    doublereal ret_val, d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9

    # Builtin functions #
    //double sin(doublereal), sqrt(doublereal)


    # Parameter adjustments #
    --a

    # Function Body #
# Computing 2nd power #
    d__1 = *r__
# Computing 2nd power #
    d__2 = *r__
# Computing 2nd power #
    d__3 = a[27]
# Computing 2nd power #
    d__4 = *r__
# Computing 2nd power #
    d__5 = a[28]
# Computing 2nd power #
    d__6 = *r__
# Computing 2nd power #
    d__7 = a[29]
# Computing 2nd power #
    d__8 = *r__
# Computing 2nd power #
    d__9 = a[30]
    ret_val = *theta + (a[17] + a[18] / *r__ + a[19] / (d__1 * d__1) + a[20] *
	     *r__ / sqrt(d__2 * d__2 + d__3 * d__3)) * sin(*theta) + (a[21] + 
	    a[22] * *r__ / sqrt(d__4 * d__4 + d__5 * d__5) + a[23] * *r__ / (
	    d__6 * d__6 + d__7 * d__7)) * sin(*theta * 2.) + (a[24] + a[25] / 
	    *r__ + a[26] * *r__ / (d__8 * d__8 + d__9 * d__9)) * sin(*theta * 
	    3.)

    return ret_val
} # theta_s__ #


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# Subroutine # int fialcos_(doublereal *r__, doublereal *theta, doublereal *
	phi, doublereal *btheta, doublereal *bphi, integer *n, doublereal *
	theta0, doublereal *dt)
{
    # System generated locals #
    integer i__1

    # Builtin functions #
    //double sin(doublereal), cos(doublereal), tan(doublereal)

    # Local variables #
    static integer m
    static doublereal t, fc, tg, ro, tm, fc1, ctg, tg21, bpn[10], btn[10], 
	    tgm, tgp, dtt, tgm2, dtt0, tgp2, ccos[10], ssin[10], cosm1, tgm2m,
	     sinm1, tgp2m, cosfi, tgm2m1, sinfi, coste, sinte, tetanm, tetanp


#  CONICAL MODEL OF BIRKELAND CURRENT FIELD BASED ON THE OLD S/R FIALCO (OF 1990-91) #
#  SEE THE OLD NOTEBOOK 1985-86-88, NOTE OF MARCH 5, BUT HERE BOTH INPUT AND OUTPUT ARE IN SPHERICAL CDS. #
#  BTN, AND BPN ARE THE ARRAYS OF BTHETA AND BPHI (BTN(i), BPN(i) CORRESPOND TO i-th MODE). #
#   ONLY FIRST  N  MODE AMPLITUDES ARE COMPUTED (N<=10). #
#    THETA0 IS THE ANGULAR HALF-WIDTH OF THE CONE, DT IS THE ANGULAR H.-W. OF THE CURRENT LAYER #
#   NOTE:  BR=0  (BECAUSE ONLY RADIAL CURRENTS ARE PRESENT IN THIS MODEL) #

    sinte = sin(*theta)
    ro = *r__ * sinte
    coste = cos(*theta)
    sinfi = sin(*phi)
    cosfi = cos(*phi)
    tg = sinte / (coste + 1.)
#        TAN(THETA/2) #
    ctg = sinte / (1. - coste)


#        CTG(THETA/2) #
    tetanp = *theta0 + *dt
    tetanm = *theta0 - *dt
    if (*theta < tetanm) {
	goto L1
    }
    tgp = tan(tetanp * .5)
    tgm = tan(tetanm * .5)
    tgm2 = tgm * tgm
    tgp2 = tgp * tgp
L1:
    cosm1 = 1.
    sinm1 = 0.
    tm = 1.
    tgm2m = 1.
    tgp2m = 1.
    i__1 = *n
    for (m = 1 m <= i__1 ++m) {
	tm *= tg
	ccos[m - 1] = cosm1 * cosfi - sinm1 * sinfi
	ssin[m - 1] = sinm1 * cosfi + cosm1 * sinfi
	cosm1 = ccos[m - 1]
	sinm1 = ssin[m - 1]
	if (*theta < tetanm) {
	    t = tm
	    dtt = m * .5 * tm * (tg + ctg)
	    dtt0 = 0.
	} else if (*theta < tetanp) {
	    tgm2m *= tgm2
	    fc = 1. / (tgp - tgm)
	    fc1 = 1. / ((m << 1) + 1)
	    tgm2m1 = tgm2m * tgm
	    tg21 = tg * tg + 1.
	    t = fc * (tm * (tgp - tg) + fc1 * (tm * tg - tgm2m1 / tm))
	    dtt = m * .5 * fc * tg21 * (tm / tg * (tgp - tg) - fc1 * (tm - 
		    tgm2m1 / (tm * tg)))
	    dtt0 = fc * .5 * ((tgp + tgm) * (tm * tg - fc1 * (tm * tg - 
		    tgm2m1 / tm)) + tm * (1. - tgp * tgm) - (tgm2 + 1.) * 
		    tgm2m / tm)
	} else {
	    tgp2m *= tgp2
	    tgm2m *= tgm2
	    fc = 1. / (tgp - tgm)
	    fc1 = 1. / ((m << 1) + 1)
	    t = fc * fc1 * (tgp2m * tgp - tgm2m * tgm) / tm
	    dtt = -t * m * .5 * (tg + ctg)
	}
	btn[m - 1] = m * t * ccos[m - 1] / ro
# L2: #
	bpn[m - 1] = -dtt * ssin[m - 1] / *r__
    }
    *btheta = btn[*n - 1] * 800.f
    *bphi = bpn[*n - 1] * 800.f
    return 0
} # fialcos_ #


# ------------------------------------------------------------------------- #


# Subroutine # int birk_shl__(doublereal *a, doublereal *ps, doublereal *
	x_sc__, doublereal *x, doublereal *y, doublereal *z__, doublereal *bx,
	 doublereal *by, doublereal *bz)
{
    # System generated locals #
    doublereal d__1, d__2

    # Builtin functions #
    //double cos(doublereal), sin(doublereal), sqrt(doublereal), exp(doublereal)
	    //

    # Local variables #
    static integer i__, k, l, m, n
    static doublereal p, q, r__, s, x1, x2, z1, z2
    static integer nn
    static doublereal fx, gx, gy, gz, fy, fz, hx, hy, hz, ct1, ct2, st1, st2, 
	    cps, epr, eqs, hxr, hzr, sps, pst1, s3ps, pst2, cypi, cyqi, czrk, 
	    czsk, sypi, syqi, sqpr, sqqs, szrk, szsk



    # Parameter adjustments #
    --a

    # Function Body #
    cps = cos(*ps)
    sps = sin(*ps)
    s3ps = cps * 2.

    pst1 = *ps * a[85]
    pst2 = *ps * a[86]
    st1 = sin(pst1)
    ct1 = cos(pst1)
    st2 = sin(pst2)
    ct2 = cos(pst2)
    x1 = *x * ct1 - *z__ * st1
    z1 = *x * st1 + *z__ * ct1
    x2 = *x * ct2 - *z__ * st2
    z2 = *x * st2 + *z__ * ct2

    l = 0
    gx = 0.
    gy = 0.
    gz = 0.

    for (m = 1 m <= 2 ++m) {
#                          AND M=2 IS FOR THE SECOND SUM ("PARALL." SYMMETRY) #
#    M=1 IS FOR THE 1ST SUM ("PERP." SYMMETRY) #
	for (i__ = 1 i__ <= 3 ++i__) {
	    p = a[i__ + 72]
	    q = a[i__ + 78]
	    cypi = cos(*y / p)
	    cyqi = cos(*y / q)
	    sypi = sin(*y / p)
	    syqi = sin(*y / q)

	    for (k = 1 k <= 3 ++k) {
		r__ = a[k + 75]
		s = a[k + 81]
		szrk = sin(z1 / r__)
		czsk = cos(z2 / s)
		czrk = cos(z1 / r__)
		szsk = sin(z2 / s)
# Computing 2nd power #
		d__1 = p
# Computing 2nd power #
		d__2 = r__
		sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
# Computing 2nd power #
		d__1 = q
# Computing 2nd power #
		d__2 = s
		sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
		epr = exp(x1 * sqpr)
		eqs = exp(x2 * sqqs)

		for (n = 1 n <= 2 ++n) {
#                                AND N=2 IS FOR THE SECOND ONE #
# N=1 IS FOR THE FIRST PART OF EACH COEFFI #
		    for (nn = 1 nn <= 2 ++nn) {
#                                         TO TAKE INTO ACCOUNT THE SCALE FACTOR DEPENDENCE #
#   NN = 1,2 FURTHER SPLITS THE COEFFICI #
			if (m == 1) {
			    fx = -sqpr * epr * cypi * szrk
			    fy = epr * sypi * szrk / p
			    fz = -epr * cypi * czrk / r__
			    if (n == 1) {
				if (nn == 1) {
				    hx = fx
				    hy = fy
				    hz = fz
				} else {
				    hx = fx * *x_sc__
				    hy = fy * *x_sc__
				    hz = fz * *x_sc__
				}
			    } else {
				if (nn == 1) {
				    hx = fx * cps
				    hy = fy * cps
				    hz = fz * cps
				} else {
				    hx = fx * cps * *x_sc__
				    hy = fy * cps * *x_sc__
				    hz = fz * cps * *x_sc__
				}
			    }
			} else {
#   M.EQ.2 #
			    fx = -sps * sqqs * eqs * cyqi * czsk
			    fy = sps / q * eqs * syqi * czsk
			    fz = sps / s * eqs * cyqi * szsk
			    if (n == 1) {
				if (nn == 1) {
				    hx = fx
				    hy = fy
				    hz = fz
				} else {
				    hx = fx * *x_sc__
				    hy = fy * *x_sc__
				    hz = fz * *x_sc__
				}
			    } else {
				if (nn == 1) {
				    hx = fx * s3ps
				    hy = fy * s3ps
				    hz = fz * s3ps
				} else {
				    hx = fx * s3ps * *x_sc__
				    hy = fy * s3ps * *x_sc__
				    hz = fz * s3ps * *x_sc__
				}
			    }
			}
			++l
			if (m == 1) {
			    hxr = hx * ct1 + hz * st1
			    hzr = -hx * st1 + hz * ct1
			} else {
			    hxr = hx * ct2 + hz * st2
			    hzr = -hx * st2 + hz * ct2
			}
			gx += hxr * a[l]
			gy += hy * a[l]
# L5: #
			gz += hzr * a[l]
		    }
# L4: #
		}
# L3: #
	    }
# L2: #
	}
# L1: #
    }
    *bx = gx
    *by = gy
    *bz = gz
    return 0
} # birk_shl__ #


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& #

# Subroutine # int birtotsy_(doublereal *ps, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bx11, doublereal *by11, doublereal *bz11,
	 doublereal *bx12, doublereal *by12, doublereal *bz12, doublereal *
	bx21, doublereal *by21, doublereal *bz21, doublereal *bx22, 
	doublereal *by22, doublereal *bz22)
{
    # Initialized data #

    static doublereal sh11[86] = { 4956703.683,-26922641.21,-11383659.85,
	    29604361.65,-38919785.97,70230899.72,34993479.24,-90409215.02,
	    30448713.69,-48360257.19,-35556751.23,57136283.6,-8013815.613,
	    30784907.86,13501620.5,-35121638.52,50297295.45,-84200377.18,
	    -46946852.58,107526898.8,-39003263.47,59465850.17,47264335.1,
	    -68892388.73,3375901.533,-9181255.754,-4494667.217,10812618.51,
	    -17351920.97,27016083.,18150032.11,-33186882.96,13340198.63,
	    -19779685.3,-17891788.15,21625767.23,16135.32442,133094.0241,
	    -13845.61859,-79159.98442,432.1215298,-85438.10368,1735.386707,
	    41891.71284,18158.14923,-105465.8135,-11685.73823,62297.34252,
	    -10811.08476,-87631.38186,9217.499261,52079.94529,-68.29127454,
	    56023.02269,-1246.029857,-27436.42793,-11972.61726,69607.08725,
	    7702.743803,-41114.3681,12.08269108,-21.30967022,-9.100782462,
	    18.26855933,-7.000685929,26.22390883,6.392164144,-21.99351743,
	    2.294204157,-16.10023369,-1.34431475,9.34212123,148.5493329,
	    99.79912328,70.78093196,35.23177574,47.45346891,58.44877918,
	    139.8135237,91.96485261,6.983488815,9.055554871,19.80484284,
	    2.860045019,.08213262337,-7.962186676e-6 }
    static doublereal sh12[86] = { -1210748.72,-52324903.95,-14158413.33,
	    19426123.6,6808641.947,-5138390.983,-1118600.499,-4675055.459,
	    2059671.506,-1373488.052,-114704.4353,-1435920.472,1438451.655,
	    61199067.17,16549301.39,-22802423.47,-7814550.995,5986478.728,
	    1299443.19,5352371.724,-2994351.52,1898553.337,203158.3658,
	    2270182.134,-618083.3112,-25950806.16,-7013783.326,9698792.575,
	    3253693.134,-2528478.464,-546323.4095,-2217735.237,1495336.589,
	    -914647.4222,-114374.1054,-1200441.634,-507068.47,1163189.975,
	    998411.8381,-861919.3631,5252210.872,-11668550.16,-4113899.385,
	    6972900.95,-2546104.076,7704014.31,2273077.192,-5134603.198,
	    256205.7901,-589970.8086,-503821.017,437612.8956,-2648640.128,
	    5887640.735,2074286.234,-3519291.144,1283847.104,-3885817.147,
	    -1145936.942,2589753.651,-408.7788403,1234.054185,739.8541716,
	    -965.8068853,3691.383679,-8628.635819,-2855.844091,5268.500178,
	    -1774.372703,5515.010707,1556.089289,-3665.43466,204.8672197,
	    110.7748799,87.36036207,5.52249133,31.0636427,73.57632579,
	    281.533136,140.3461448,17.07537768,6.729732641,4.100970449,
	    2.780422877,.08742978101,-1.028562327e-5 }
    static doublereal sh21[86] = { -67763516.61,-49565522.84,10123356.08,
	    51805446.1,-51607711.68,164360662.1,-4662006.024,-191297217.6,
	    -7204547.103,30372354.93,-750371.9365,-36564457.17,61114395.65,
	    45702536.5,-9228894.939,-47893708.68,47290934.33,-149155112.,
	    4226520.638,173588334.5,7998505.443,-33150962.72,832493.2094,
	    39892545.84,-11303915.16,-8901327.398,1751557.11,9382865.82,
	    -9054707.868,27918664.5,-788741.7146,-32481294.42,-2264443.753,
	    9022346.503,-233526.0185,-10856269.53,-244450.885,1908295.272,
	    185445.1967,-1074202.863,41827.75224,-241553.7626,-20199.1258,
	    123235.6084,199501.4614,-1936498.464,-178857.4074,1044724.507,
	    121044.9917,-946479.9247,-91808.28803,532742.7569,-20742.28628,
	    120633.2193,10018.49534,-61599.11035,-98709.58977,959095.177,
	    88500.43489,-517471.5287,-81.56122911,816.2472344,55.3071171,
	    -454.5368824,25.7469381,-202.500735,-7.369350794,104.9429812,
	    58.14049362,-685.5919355,-51.71345683,374.0125033,247.9296982,
	    159.2471769,102.3151816,15.81062488,34.99767599,133.0832773,
	    219.6475201,107.9582783,10.00264684,7.718306072,25.22866153,
	    5.013583103,.08407754233,-9.613356793e-6 }
    static doublereal sh22[86] = { -43404887.31,8896854.538,-8077731.036,
	    -10247813.65,6346729.086,-9416801.212,-1921670.268,7805483.928,
	    2299301.127,4856980.17,-1253936.462,-4695042.69,54305735.91,
	    -11158768.1,10051771.85,12837129.47,-6380785.836,12387093.5,
	    1687850.192,-10492039.47,-5777044.862,-6916507.424,2855974.911,
	    7027302.49,-26176628.93,5387959.61,-4827069.106,-6193036.589,
	    2511954.143,-6205105.083,-553187.2984,5341386.847,3823736.361,
	    3669209.068,-1841641.7,-3842906.796,281561.722,-5013124.63,
	    379824.5943,2436137.901,-76337.55394,548518.2676,42134.28632,
	    -281711.3841,-365514.8666,-2583093.138,-232355.8377,1104026.712,
	    -131536.3445,2320169.882,-174967.6603,-1127251.881,35539.82827,
	    -256132.9284,-19620.06116,131598.7965,169033.6708,1194443.5,
	    107320.3699,-510672.0036,1211.177843,-17278.19863,1140.037733,
	    8347.612951,-303.8408243,2405.771304,174.0634046,-1248.72295,
	    -1231.229565,-8666.932647,-754.0488385,3736.878824,227.2102611,
	    115.9154291,94.3436483,3.625357304,64.03192907,109.0743468,
	    241.4844439,107.7583478,22.36222385,6.282634037,27.79399216,
	    2.270602235,.08708605901,-1.256706895e-5 }

    extern # Subroutine # int birsh_sy__(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *)
    static doublereal fx11, fy11, fz11, hx11, hy11, hz11, fx12, fy12, fz12, 
	    hx12, hy12, hz12, fx21, fy21, fz21, hx21, hy21, hz21, fx22, fy22, 
	    fz22, hx22, hy22, hz22, x_sc__
    extern # Subroutine # int bir1n2sy_(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *)


#   THIS S/R IS ALMOST IDENTICAL TO BIRK_TOT, BUT IT IS FOR THE SYMMETRIC MODE, IN WHICH #
#     J_parallel IS AN EVEN FUNCTION OF Ygsm. #


#      IOPBS -  BIRKELAND FIELD MODE FLAG: #
#         IOPBS=0 - ALL COMPONENTS #
#         IOPBS=1 - REGION 1, MODES 1 & 2 (SYMMETRIC !) #
#         IOPBS=2 - REGION 2, MODES 1 & 2 (SYMMETRIC !) #

#                                            (JOINT WITH  BIRK_TOT  FOR THE ANTISYMMETRICAL MODE) #

#  INPUT PARAMETERS, SPECIFIED #

# PARAMETERS, CONTROL DAY #




    dphi_b_rho0__1.xkappa = birkpar_1.xkappa1
#  FORWARDED IN BIR1N2SY #
    x_sc__ = birkpar_1.xkappa1 - 1.1
#  FORWARDED IN BIRSH_SY #
    bir1n2sy_(&c__1, &c__1, ps, x, y, z__, &fx11, &fy11, &fz11)
#  REGION 1, #
    birsh_sy__(sh11, ps, &x_sc__, x, y, z__, &hx11, &hy11, &hz11)
    *bx11 = fx11 + hx11
    *by11 = fy11 + hy11
    *bz11 = fz11 + hz11
    bir1n2sy_(&c__1, &c__2, ps, x, y, z__, &fx12, &fy12, &fz12)
#  REGION 1, #
    birsh_sy__(sh12, ps, &x_sc__, x, y, z__, &hx12, &hy12, &hz12)
    *bx12 = fx12 + hx12
    *by12 = fy12 + hy12
    *bz12 = fz12 + hz12
    dphi_b_rho0__1.xkappa = birkpar_1.xkappa2
#  FORWARDED IN BIR1N2SY #
    x_sc__ = birkpar_1.xkappa2 - 1.
#  FORWARDED IN BIRSH_SY #
    bir1n2sy_(&c__2, &c__1, ps, x, y, z__, &fx21, &fy21, &fz21)
#  REGION 2, #
    birsh_sy__(sh21, ps, &x_sc__, x, y, z__, &hx21, &hy21, &hz21)
    *bx21 = fx21 + hx21
    *by21 = fy21 + hy21
    *bz21 = fz21 + hz21
    bir1n2sy_(&c__2, &c__2, ps, x, y, z__, &fx22, &fy22, &fz22)
#  REGION 2, #
    birsh_sy__(sh22, ps, &x_sc__, x, y, z__, &hx22, &hy22, &hz22)
    *bx22 = fx22 + hx22
    *by22 = fy22 + hy22
    *bz22 = fz22 + hz22
    return 0
} # birtotsy_ #


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Subroutine # int bir1n2sy_(integer *numb, integer *mode, doublereal *ps, 
	doublereal *x, doublereal *y, doublereal *z__, doublereal *bx, 
	doublereal *by, doublereal *bz)
{
    # Initialized data #

    static doublereal beta = .9
    static doublereal rh = 10.
    static doublereal eps = 3.
    static doublereal a11[31] = { .161806835,-.1797957553,2.999642482,
	    -.9322708978,-.681105976,.2099057262,-8.358815746,-14.8603355,
	    .3838362986,-16.30945494,4.537022847,2.685836007,27.97833029,
	    6.330871059,1.876532361,18.95619213,.96515281,.4217195118,
	    -.0895777002,-1.823555887,.7457045438,-.5785916524,-1.010200918,
	    .01112389357,.09572927448,-.3599292276,8.713700514,.9763932955,
	    3.834602998,2.492118385,.7113544659 }
    static doublereal a12[31] = { .705802694,-.2845938535,5.715471266,
	    -2.47282088,-.7738802408,.347829393,-11.37653694,-38.64768867,
	    .6932927651,-212.4017288,4.944204937,3.071270411,33.05882281,
	    7.387533799,2.366769108,79.22572682,.6154290178,.5592050551,
	    -.1796585105,-1.65493221,.7309108776,-.4926292779,-1.130266095,
	    -.009613974555,.1484586169,-.2215347198,7.883592948,.02768251655,
	    2.950280953,1.212634762,.5567714182 }
    static doublereal a21[31] = { .1278764024,-.2320034273,1.805623266,
	    -32.3724144,-.9931490648,.317508563,-2.492465814,-16.21600096,
	    .2695393416,-6.752691265,3.971794901,14.54477563,41.10158386,
	    7.91288973,1.258297372,9.583547721,1.014141963,.5104134759,
	    -.1790430468,-1.756358428,.7561986717,-.6775248254,-.0401401642,
	    .01446794851,.1200521731,-.2203584559,4.50896385,.8221623576,
	    1.77993373,1.102649543,.886788002 }
    static doublereal a22[31] = { .4036015198,-.3302974212,2.82773093,
	    -45.4440583,-1.611103927,.4927112073,-.003258457559,-49.59014949,
	    .3796217108,-233.7884098,4.31266698,18.05051709,28.95320323,
	    11.09948019,.7471649558,67.10246193,.5667096597,.6468519751,
	    -.1560665317,-1.460805289,.7719653528,-.6658988668,2.515179349e-6,
	    .02426021891,.1195003324,-.2625739255,4.377172556,.2421190547,
	    2.503482679,1.071587299,.724799743 }

    # System generated locals #
    doublereal d__1, d__2, d__3, d__4, d__5

    # Builtin functions #
    //double sqrt(doublereal), atan2(doublereal, doublereal), sin(doublereal), 
	    //cos(doublereal), pow_dd(doublereal *, doublereal *)

    # Local variables #
    static doublereal dphisphi, dphisrho
    extern # Subroutine # int twoconss_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *)
    static doublereal xs, zs, phi, rho, rsc, bxs, xsc, ysc, zsc, bzs, rho2, 
	    r1rh, by_s__, byas, phis, brack, cphic, sphic, psias, bphi_s__, 
	    bphias, cphics, brho_s__, brhoas, sphics, dphisdy


#   THIS CODE IS VERY SIMILAR TO BIRK_1N2, BUT IT IS FOR THE "SYMMETRICAL" MODE, IN WHICH J_parallel #
#     IS A SYMMETRIC (EVEN) FUNCTION OF Ygsm #

#  CALCULATES COMPONENTS  OF REGION 1/2 FIELD IN SPHERICAL COORDS.  DERIVED FROM THE S/R DIPDEF2C (WHICH #
#    DOES THE SAME JOB, BUT INPUT/OUTPUT THERE WAS IN SPHERICAL COORDS, WHILE HERE WE USE CARTESIAN ONES) #

#   INPUT:  NUMB=1 (2) FOR REGION 1 (2) CURRENTS #
#           MODE=1 YIELDS SIMPLE SINUSOIDAL MLT VARIATION, WITH MAXIMUM CURRENT AT DAWN/DUSK MERIDIAN #
#     WHILE MODE=2 YIELDS THE SECOND HARMONIC. #


#   SEE N #
#  (1) DPHI:   HALF-DIFFERENCE (IN RADIANS) BETWEEN DAY AND NIGHT LATITUDE OF FAC OVAL AT IONOSPHERIC ALTITUD
E #
#              TYPICAL VALUE: 0.06 #
#  (2) B:      AN ASYMMETRY FACTOR AT HIGH-ALTITUDES  FOR B=0, THE ONLY ASYMMETRY IS THAT FROM DPHI #
#              TYPICAL VALUES: 0.35-0.70 #
#  (3) RHO_0:  A FIXED PARAMETER, DEFINING THE DISTANCE RHO, AT WHICH THE LATITUDE SHIFT GRADUALLY SATURATES 
AND #
#              STOPS INCREASING #
#              ITS VALUE WAS ASSUMED FIXED, EQUAL TO 7.0. #
#  (4) XKAPPA: AN OVERALL SCALING FACTOR, WHICH CAN BE USED FOR CHANGING THE SIZE OF THE F.A.C. OVAL #

# THESE PARAMETERS CONTRO #
# parameters of the tilt-depend #
    dphi_b_rho0__1.b = .5f
    dphi_b_rho0__1.rho_0__ = 7.f
    modenum_1.m = *mode
    if (*numb == 1) {
	dphi_b_rho0__1.dphi = .055
	dtheta_1.dtheta = .06
    }
    if (*numb == 2) {
	dphi_b_rho0__1.dphi = .03
	dtheta_1.dtheta = .09
    }
    xsc = *x * dphi_b_rho0__1.xkappa
    ysc = *y * dphi_b_rho0__1.xkappa
    zsc = *z__ * dphi_b_rho0__1.xkappa
# Computing 2nd power #
    d__1 = xsc
# Computing 2nd power #
    d__2 = zsc
    rho = sqrt(d__1 * d__1 + d__2 * d__2)
# Computing 2nd power #
    d__1 = xsc
# Computing 2nd power #
    d__2 = ysc
# Computing 2nd power #
    d__3 = zsc
    rsc = sqrt(d__1 * d__1 + d__2 * d__2 + d__3 * d__3)

# Computing 2nd power #
    d__1 = dphi_b_rho0__1.rho_0__
    rho2 = d__1 * d__1
    if (xsc == 0. && zsc == 0.) {
	phi = 0.
    } else {
	phi = atan2(-zsc, xsc)
#  FROM CARTESIAN TO CYLINDRICAL (RHO,PHI #
    }
    sphic = sin(phi)
    cphic = cos(phi)
#  "C" means "CYLINDRICAL", TO DISTINGUISH FROM S #
# Computing 2nd power #
    d__1 = rho
# Computing 2nd power #
    d__2 = rho
    brack = dphi_b_rho0__1.dphi + dphi_b_rho0__1.b * rho2 / (rho2 + 1.) * (
	    d__1 * d__1 - 1.) / (rho2 + d__2 * d__2)
    r1rh = (rsc - 1.) / rh
    d__1 = pow(r1rh, eps) + 1.
    d__2 = 1. / eps
    psias = beta * *ps / pow(d__1, d__2)
    phis = phi - brack * sin(phi) - psias
    dphisphi = 1. - brack * cos(phi)
# Computing 2nd power #
    d__2 = rho
# Computing 2nd power #
    d__1 = rho2 + d__2 * d__2
    d__3 = eps - 1.
    d__4 = pow(r1rh, eps) + 1.
    d__5 = 1. / eps + 1.
    dphisrho = dphi_b_rho0__1.b * -2. * rho2 * rho / (d__1 * d__1) * sin(phi) 
	    + beta * *ps * pow(r1rh, d__3) * rho / (rh * rsc * pow(
	    d__4, d__5))
    d__1 = eps - 1.
    d__2 = pow(r1rh, eps) + 1.
    d__3 = 1. / eps + 1.
    dphisdy = beta * *ps * pow(r1rh, d__1) * ysc / (rh * rsc * pow(
	    d__2, d__3))
    sphics = sin(phis)
    cphics = cos(phis)
    xs = rho * cphics
    zs = -rho * sphics
    if (*numb == 1) {
	if (*mode == 1) {
	    twoconss_(a11, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
	if (*mode == 2) {
	    twoconss_(a12, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
    } else {
	if (*mode == 1) {
	    twoconss_(a21, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
	if (*mode == 2) {
	    twoconss_(a22, &xs, &ysc, &zs, &bxs, &byas, &bzs)
	}
    }
    brhoas = bxs * cphics - bzs * sphics
    bphias = -bxs * sphics - bzs * cphics
    brho_s__ = brhoas * dphisphi * dphi_b_rho0__1.xkappa
    bphi_s__ = (bphias - rho * (byas * dphisdy + brhoas * dphisrho)) * 
	    dphi_b_rho0__1.xkappa
    by_s__ = byas * dphisphi * dphi_b_rho0__1.xkappa
    *bx = brho_s__ * cphic - bphi_s__ * sphic
    *by = by_s__
    *bz = -brho_s__ * sphic - bphi_s__ * cphic
    return 0
} # bir1n2sy_ #


# ========================================================================= #

# Subroutine # int twoconss_(doublereal *a, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bx, doublereal *by, doublereal *bz)
{
    # Initialized data #

    static doublereal hsqr2 = .707106781

    # System generated locals #
    doublereal d__1, d__2

    # Local variables #
    extern # Subroutine # int one_cone__(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *)
    static doublereal bxn, byn, bzn, xas, yas, bxs, bys, bzs, bxas, byas


#   DIFFERS FROM TWOCONES:  THIS S/R IS FOR THE "SYMMETRIC" MODE OF BIRKELAND CURRENTS IN THAT #
#                           HERE THE FIELD IS ROTATED BY 90 DEGS FOR M=1 AND BY 45 DEGS FOR M=2 #

#   ADDS FIELDS FROM TWO CONES (NORTHERN AND SOUTHERN), WITH A PROPER SYMMETRY OF THE CURRENT AND FIELD, #
#     CORRESPONDING TO THE REGION 1 BIRKELAND CURRENTS. (SEE NB #6, P.58). #

    # Parameter adjustments #
    --a

    # Function Body #
    if (modenum_1.m == 1) {
#   ROTATION BY 90 DEGS #
	xas = *y
	yas = -(*x)
    } else {
#   ROTATION BY 45 DEGS #
	xas = (*x + *y) * hsqr2
	yas = (*y - *x) * hsqr2
    }
    one_cone__(&a[1], &xas, &yas, z__, &bxn, &byn, &bzn)
    d__1 = -yas
    d__2 = -(*z__)
    one_cone__(&a[1], &xas, &d__1, &d__2, &bxs, &bys, &bzs)
    bxas = bxn - bxs
    byas = byn + bys
    *bz = bzn + bzs
    if (modenum_1.m == 1) {
#   ROTATION BY 90 DEGS #
	*bx = -byas
	*by = bxas
    } else {
	*bx = (bxas - byas) * hsqr2
	*by = (bxas + byas) * hsqr2
    }
    return 0
} # twoconss_ #


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

# Subroutine # int birsh_sy__(doublereal *a, doublereal *ps, doublereal *
	x_sc__, doublereal *x, doublereal *y, doublereal *z__, doublereal *bx,
	 doublereal *by, doublereal *bz)
{
    # System generated locals #
    doublereal d__1, d__2

    # Builtin functions #
    //double cos(doublereal), sin(doublereal), sqrt(doublereal), exp(doublereal)
	    //

    # Local variables #
    static integer i__, k, l, m, n
    static doublereal p, q, r__, s, x1, x2, z1, z2
    static integer nn
    static doublereal fx, gx, gy, gz, fy, fz, hx, hy, hz, ct1, ct2, st1, st2, 
	    cps, epr, eqs, hxr, hzr, sps, pst1, s3ps, pst2, cypi, cyqi, czrk, 
	    czsk, sypi, syqi, sqpr, sqqs, szrk, szsk


#   THIS S/R IS QUITE SIMILAR TO BIRK_SHL, BUT IT IS FOR THE SYMMETRIC MODE OF BIRKELAND CURRENT FIELD #
#     AND FOR THAT REASON THE FIELD COMPONENTS HAVE A DIFFERENT KIND OF SYMMETRY WITH RESPECT TO Y_gsm #


    # Parameter adjustments #
    --a

    # Function Body #
    cps = cos(*ps)
    sps = sin(*ps)
    s3ps = cps * 2.

    pst1 = *ps * a[85]
    pst2 = *ps * a[86]
    st1 = sin(pst1)
    ct1 = cos(pst1)
    st2 = sin(pst2)
    ct2 = cos(pst2)
    x1 = *x * ct1 - *z__ * st1
    z1 = *x * st1 + *z__ * ct1
    x2 = *x * ct2 - *z__ * st2
    z2 = *x * st2 + *z__ * ct2

    l = 0
    gx = 0.
    gy = 0.
    gz = 0.

    for (m = 1 m <= 2 ++m) {
#                          AND M=2 IS FOR THE SECOND SUM ("PARALL." SYMMETRY) #
#    M=1 IS FOR THE 1ST SUM ("PERP." SYMMETRY) #
	for (i__ = 1 i__ <= 3 ++i__) {
	    p = a[i__ + 72]
	    q = a[i__ + 78]
	    cypi = cos(*y / p)
	    cyqi = cos(*y / q)
	    sypi = sin(*y / p)
	    syqi = sin(*y / q)

	    for (k = 1 k <= 3 ++k) {
		r__ = a[k + 75]
		s = a[k + 81]
		szrk = sin(z1 / r__)
		czsk = cos(z2 / s)
		czrk = cos(z1 / r__)
		szsk = sin(z2 / s)
# Computing 2nd power #
		d__1 = p
# Computing 2nd power #
		d__2 = r__
		sqpr = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
# Computing 2nd power #
		d__1 = q
# Computing 2nd power #
		d__2 = s
		sqqs = sqrt(1. / (d__1 * d__1) + 1. / (d__2 * d__2))
		epr = exp(x1 * sqpr)
		eqs = exp(x2 * sqqs)

		for (n = 1 n <= 2 ++n) {
#                                AND N=2 IS FOR THE SECOND ONE #
# N=1 IS FOR THE FIRST PART OF EACH COEFFI #
		    for (nn = 1 nn <= 2 ++nn) {
#                                         TO TAKE INTO ACCOUNT THE SCALE FACTOR DEPENDENCE #
#   NN = 1,2 FURTHER SPLITS THE COEFFICI #
			if (m == 1) {
			    fx = sqpr * epr * sypi * szrk
			    fy = epr * cypi * szrk / p
			    fz = epr * sypi * czrk / r__
			    if (n == 1) {
				if (nn == 1) {
				    hx = fx
				    hy = fy
				    hz = fz
				} else {
				    hx = fx * *x_sc__
				    hy = fy * *x_sc__
				    hz = fz * *x_sc__
				}
			    } else {
				if (nn == 1) {
				    hx = fx * cps
				    hy = fy * cps
				    hz = fz * cps
				} else {
				    hx = fx * cps * *x_sc__
				    hy = fy * cps * *x_sc__
				    hz = fz * cps * *x_sc__
				}
			    }
			} else {
#   M.EQ.2 #
			    fx = sps * sqqs * eqs * syqi * czsk
			    fy = sps / q * eqs * cyqi * czsk
			    fz = -sps / s * eqs * syqi * szsk
			    if (n == 1) {
				if (nn == 1) {
				    hx = fx
				    hy = fy
				    hz = fz
				} else {
				    hx = fx * *x_sc__
				    hy = fy * *x_sc__
				    hz = fz * *x_sc__
				}
			    } else {
				if (nn == 1) {
				    hx = fx * s3ps
				    hy = fy * s3ps
				    hz = fz * s3ps
				} else {
				    hx = fx * s3ps * *x_sc__
				    hy = fy * s3ps * *x_sc__
				    hz = fz * s3ps * *x_sc__
				}
			    }
			}
			++l
			if (m == 1) {
			    hxr = hx * ct1 + hz * st1
			    hzr = -hx * st1 + hz * ct1
			} else {
			    hxr = hx * ct2 + hz * st2
			    hzr = -hx * st2 + hz * ct2
			}
			gx += hxr * a[l]
			gy += hy * a[l]
# L5: #
			gz += hzr * a[l]
		    }
# L4: #
		}
# L3: #
	    }
# L2: #
	}
# L1: #
    }
    *bx = gx
    *by = gy
    *bz = gz
    return 0
} # birsh_sy__ #


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& #

# Subroutine # int dipole_(doublereal *ps, doublereal *x, doublereal *y, 
	doublereal *z__, doublereal *bx, doublereal *by, doublereal *bz)
{
    # Initialized data #

    static integer m = 0
    static doublereal psi = 5.

    # System generated locals #
    doublereal d__1, d__2

    # Builtin functions #
    //double sin(doublereal), cos(doublereal), sqrt(doublereal)

    # Local variables #
    static doublereal p, q, t, u, v, cps, sps


#      A DOUBLE PRECISION ROUTINE #

#  CALCULATES GSM COMPONENTS OF A GEODIPOLE FIELD WITH THE DIPOLE MOMENT #
#  CORRESPONDING TO THE EPOCH OF 2000. #

# ----INPUT PARAMETERS: #
#     PS - GEODIPOLE TILT ANGLE IN RADIANS, #
#     X,Y,Z - GSM COORDINATES IN RE (1 RE = 6371.2 km) #

# ----OUTPUT PARAMETERS: #
#     BX,BY,BZ - FIELD COMPONENTS IN GSM SYSTEM, IN NANOTESLA. #

#  LAST MODIFICATION: JAN. 5, 2001. THE VALUE OF THE DIPOLE MOMENT WAS UPDATED TO 2000. #
#    AND A "SAVE" STATEMENT HAS BEEN ADDED, TO AVOID POTENTIAL PROBLEMS WITH SOME #
#    FORTRAN COMPILERS #

#  WRITTEN BY: N. A. TSYGANENKO #

    if (m == 1 && (d__1 = *ps - psi, abs(d__1)) < 1e-5) {
	goto L1
    }
    sps = sin(*ps)
    cps = cos(*ps)
    psi = *ps
    m = 1
L1:
# Computing 2nd power #
    d__1 = *x
    p = d__1 * d__1
# Computing 2nd power #
    d__1 = *z__
    u = d__1 * d__1
    v = *z__ * 3. * *x
# Computing 2nd power #
    d__1 = *y
    t = d__1 * d__1
# Computing 5th power #
    d__1 = sqrt(p + t + u), d__2 = d__1, d__1 *= d__1
    q = 30115. / (d__2 * (d__1 * d__1))
    *bx = q * ((t + u - p * 2.) * sps - v * cps)
    *by = *y * -3. * q * (*x * sps + *z__ * cps)
    *bz = q * ((p + t - u * 2.) * cps - v * sps)
    return 0
}

'''

def t96_mgnp_d(xn_pd, vel, xgsm, ygsm, zgsm):
    '''
    DOUBLE-PRECISION VERSION !!!!!!!!   HENCE THE SUFFIX "D" IN THE NAME 

    FOR ANY POINT OF SPACE WITH GIVEN COORDINATES (XGSM,YGSM,ZGSM), THIS SUBROUTINE DEFINES 
    THE POSITION OF A POINT (XMGNP,YMGNP,ZMGNP) AT THE T96 MODEL MAGNETOPAUSE, HAVING THE 
    SAME VALUE OF THE ELLIPSOIDAL TAU-COORDINATE, AND THE DISTANCE BETWEEN THEM.  THIS IS 
    NOT THE SHORTEST DISTANCE D_MIN TO THE BOUNDARY, BUT DIST ASYMPTOTICALLY TENDS TO D_MIN, 
    AS THE OBSERVATION POINT GETS CLOSER TO THE MAGNETOPAUSE. 

    INPUT: XN_PD - EITHER SOLAR WIND PROTON NUMBER DENSITY (PER C.C.) (IF VEL>0)
                    OR THE SOLAR WIND RAM PRESSURE IN NANOPASCALS   (IF VEL<0) 
         VEL - EITHER SOLAR WIND VELOCITY (KM/SEC) 
                  OR ANY NEGATIVE NUMBER, WHICH INDICATES THAT XN_PD STANDS
                     FOR THE SOLAR WIND PRESSURE, RATHER THAN FOR THE DENSITY 

         XGSM,YGSM,ZGSM - COORDINATES OF THE OBSERVATION POINT IN EARTH RADII

    OUTPUT: XMGNP,YMGNP,ZMGNP - GSM POSITION OF THE BOUNDARY POINT, HAVING THE SAME 
          VALUE OF TAU-COORDINATE AS THE OBSERVATION POINT (XGSM,YGSM,ZGSM)
          DIST -  THE DISTANCE BETWEEN THE TWO POINTS, IN RE,
          ID -    POSITION FLAG ID=+1 (-1) MEANS THAT THE POINT (XGSM,YGSM,ZGSM)
          LIES INSIDE (OUTSIDE) THE MODEL MAGNETOPAUSE, RESPECTIVELY. 

    THE PRESSURE-DEPENDENT MAGNETOPAUSE IS THAT USED IN THE T96_01 MODEL 
    (TSYGANENKO, JGR, V.100, P.5599, 1995 ESA SP-389, P.181, OCT. 1996) 

    AUTHOR:  N.A. TSYGANENKO 
    DATE:    AUG.1, 1995, REVISED APRIL 3, 2003.


    DEFINE SOLAR WIND DYNAMIC PRESSURE (NANOPASCALS, ASSUMING 4% OF ALPHA-PARTICLES), 
    IF NOT EXPLICITLY SPECIFIED IN THE INPUT: 
    '''
    if vel < 0.:
        pd = xn_pd
    else:
    # Computing 2nd power
        d1 = vel
        pd = xn_pd * 1.94e-6 * (d1**2)

    #  RATIO OF PD TO THE AVERAGE PRESSURE, ASSUMED EQUAL TO 2 nPa:
    rat = pd / 2.
    rat16 = rat**.14
    # (THE POWER INDEX 0.14 IN THE SCALING FACTOR IS THE BEST-FIT VALUE OBTAINED FROM DATA
    #    AND USED IN THE T96_01 VERSION)

    # VALUES OF THE MAGNETOPAUSE PARAMETERS FOR  PD = 2 nPa:

    a0 = 34.586
    s00 = 1.196
    x00 = 3.4397

    # VALUES OF THE MAGNETOPAUSE PARAMETERS, SCALED BY THE ACTUAL PRESSURE:

    a = a0 / rat16
    s0 = s00
    x0 = x00 / rat16
    xm = x0 - a

    # (XM IS THE X-COORDINATE OF THE "SEAM" BETWEEN THE ELLIPSOID AND THE CYLINDER)

    #     (FOR DETAILS ON THE ELLIPSOIDAL COORDINATES, SEE THE PAPER:
    #      N.A.TSYGANENKO, SOLUTION OF CHAPMAN-FERRARO PROBLEM FOR AN
    #      ELLIPSOIDAL MAGNETOPAUSE, PLANET.SPACE SCI., V.37, P.1037, 1989).

    if ygsm != 0. or zgsm != 0.:
        phi = atan2(ygsm, zgsm)
    else:
        phi = 0.

    # Computing 2nd power
    d1 = ygsm
    # Computing 2nd power
    d2 = zgsm
    rho = sqrt(d1**2 + d2**2)

    if xgsm < xm:
        xmgnp = xgsm
        # Computing 2nd power
        d1 = s0
        rhomgnp = a * sqrt(d1**2 - 1.)
        ymgnp = rhomgnp * sin(phi)
        zmgnp = rhomgnp * cos(phi)
        # Computing 2nd power
        d1 = xgsm - xmgnp
        # Computing 2nd power
        d2 = ygsm - ymgnp
        # Computing 2nd power
        d3 = zgsm - zmgnp
        dist = sqrt(d1**2 + d2**2 + d3**2)

        if rhomgnp > rho:
            id = 1

        if rhomgnp <= rho:
            id = -1

        return xmgnp, ymgnp, zmgnp, dist, id

    xksi = (xgsm - x0) / a + 1.
    xdzt = rho / a
    # Computing 2nd power
    d1 = xksi + 1.
    # Computing 2nd power
    d2 = xdzt
    sq1 = sqrt(d1**2 + d2**2)
    # Computing 2nd power
    d1 = 1. - xksi
    # Computing 2nd power 
    d2 = xdzt
    sq2 = sqrt(d1**2 + d2**2)
    sigma = (sq1 + sq2) * .5
    tau = (sq1 - sq2) * .5

    # NOW CALCULATE (X,Y,Z) FOR THE CLOSEST POINT AT THE MAGNETOPAUSE

    xmgnp = x0 - a * (1. - s0 * tau)
    # Computing 2nd power
    d1 = s0
    # Computing 2nd power
    d2 = tau
    arg = (d1**2 - 1.) * (1. - d2**2)

    if (arg < 0.):
        arg = 0.

    rhomgnp = a * sqrt(arg)
    ymgnp = rhomgnp * sin(phi)
    zmgnp = rhomgnp * cos(phi)

    #  NOW CALCULATE THE DISTANCE BETWEEN THE POINTS {XGSM,YGSM,ZGSM} AND {XMGNP,YMGNP,ZMGNP}:
    #   (IN GENERAL, THIS IS NOT THE SHORTEST DISTANCE D_MIN, BUT DIST ASYMPTOTICALLY TENDS 
    #    TO D_MIN, AS WE ARE GETTING CLOSER TO THE MAGNETOPAUSE):

    # Computing 2nd power #
    d1 = xgsm - xmgnp
    # Computing 2nd power #
    d2 = ygsm - ymgnp
    # Computing 2nd power #
    d3 = zgsm - zmgnp
    dist = sqrt(d1**2 + d2**2 + d3**2)

    if sigma > s0:
        id = -1

    #  ID=-1 MEANS THAT THE POINT LIES OUTSID
    if sigma <= s0:
        id = 1

    #                                           THE MAGNETOSPHERE
    #  ID=+1 MEANS THAT THE POINT LIES INSIDE
    return xmgnp, ymgnp, zmgnp, dist, id

