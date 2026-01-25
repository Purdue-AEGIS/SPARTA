import numpy as np
import scipy.sparse as sps
from scipy.sparse import diags
from oodesign import Load
import const
from const import NodeSide, StudyType, phase_angle_map
from models.model import Model, ValType
import utils
from utils import phasor_to_timedomain, get_start_end_idx
from pprint import pformat

"""
1) class LoadModel represent any load connected to the node.
2) Each instance of these class could have different no. of phases based on the line object passed to it.
3) Load object passed to the init function here comes from the object oriented representation of the network.
4) class LoadModel has following functions:
    a)get_M_powerflow
    b)get_u_powerflow
    c)get_fy_powerflow        
5)  Presently, only StarConstantPower model is implemented.
TODO: M, K, fy, u for dynamic simulation
TODO: Other load models to be implemented.
"""


class LoadModel(Model):
    def __init__(self, load_obj: Load):
        # map all required items from Load object to the Load model object
        self.obj = load_obj

        # the number of phases
        self.n_ph = self.obj.n_ph

        self.num_term = 1

        # the node at which the load is connected
        # self.at_node = self.obj.node

        # nominal voltage, active and reactive power consumed by the load and pf computation
        self.nominal_voltage = self.obj.nominal_voltage
        self.P = self.obj.active_power
        self.Q = self.obj.reactive_power

        # NOTE: each subclass should calculate and update its own number of eqns
        self.num_eqns = None

    def get_id(self):
        return self.obj.id

    def get_basetype(self):
        return "load"


####################################################################################################################################################
#  Star Constant Impedance Load Model
class StarConstantImpedanceLModel(LoadModel):
    def __init__(self, load_obj: Load):
        super().__init__(load_obj)

        # book keeping of the equations
        # fmt: off
        self.num_eqns_real = 0
        self.num_eqns_complex= (self.n_ph # first set  :1)-I + i = 0
                        + self.n_ph # second set :2)V - v = 0
                        + self.n_ph # third set  :3)Li - lamda = 0
                        + self.n_ph # fourth set :4)-v + Ri + jw * lamda = 0
                        )
        # fmt: on
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex
        self.num_eqns_dynamic = self.num_eqns

        # book keeping of the variables

        # fmt: off
        # y = [w, V, I, v, i, lamda]
        self.num_vars_real = (1 ) # w
        self.num_vars_complex = (
            self.n_ph   # V
            + self.n_ph # I
            + self.n_ph # v
            + self.n_ph # i
            + self.n_ph # lamda
        )

        self.num_vars = self.num_vars_real + self.num_vars_complex
        self.num_vars_dynamic = self.num_vars
        # fmt: on

        # create dictionaries to store the offset of each variable in the y vector
        # fmt:off
        # 1. dictionary of var offset for real variables
        self.var_offset_real = {"w": 0}
        # 2. dictionary of var offset for complex variables
        self.var_offset_complex = {
            "V" :0,
            "I" :self.n_ph,
            "v" :self.n_ph + self.n_ph,
            "i" :self.n_ph + self.n_ph + self.n_ph,
            "lamda" :self.n_ph + self.n_ph + self.n_ph + self.n_ph
        }
        # 3. dictionary of var offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
            "lamda": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }

        # to check if the total number of variables is correct with respect to the offset
        assert len(self.var_offset_real.keys()) + len(self.var_offset_complex.keys()) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["lamda"] + self.n_ph

        self.var_offset_dynamic = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
            "lamda": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }

        # constant computation

        # create L and R matrices
        # get the nominal voltage from the object
        # get the S from active and reactive power from the load object
        # get the impedance Z = V^2/S*
        # TODO: all this should actually be a part of the adapter. At this stage we should only have L and R
        V = np.array(list(self.nominal_voltage.values()))
        S = np.array(list(self.P.values())) + 1j * np.array(list(self.Q.values()))
        self.Z = V**2 / S.conjugate()
        self.L = sps.diags(list(self.Z.imag)) / const.w_nominal
        self.R = sps.diags(list(self.Z.real))

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [V, I, w, v, i, lamda]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

        # initialize S
        # get the active and reactive power from the load object
        P = np.array(list(self.P.values())).reshape(-1, 1)  # convert to W
        Q = np.array(list(self.Q.values())).reshape(-1, 1)  # convert to VAR
        S = P + 1j * Q
        i_phasors = (S / v_phasors).conjugate()
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        y_0[idx_i_start:idx_i_end, 0] = i_phasors

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        return y_0

    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ):
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn

    def initial_yp_dynamic(self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom):
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        idx_v_start = self.var_offset_dynamic["v"]
        idx_v_end = idx_v_start + self.n_ph
        v = y0_dyn_comp[idx_v_start:idx_v_end]

        idx_i_start = self.var_offset_dynamic["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y0_dyn_comp[idx_i_start:idx_i_end]

        idx_p_lamda_start = self.var_offset_dynamic["lamda"]
        idx_p_lamda_end = idx_p_lamda_start + self.n_ph

        yp0_dyn[idx_p_lamda_start:idx_p_lamda_end] = v - i @ self.R

        return yp0_dyn

    def initial_guess_dynamic_zero(self, y_comp, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # powerflow vars: [w, V, I, v, i, lamda]
        # dyn vars: [w, V, I, v, i, lamda]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        # y0_dyn[0] = y_comp[0].real

        # # init "v" and "i":
        # v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        # v_phasors = np.array(list(v_phasors_dict.values()))
        # dyn_idx_v_start, dyn_idx_v_end = get_start_end_idx(self.var_offset_dynamic, "v", self.n_ph)
        # y0_dyn[dyn_idx_v_start:dyn_idx_v_end] = [np.sqrt(2) * phasor_to_timedomain(val) for val in v_phasors]

        # # initialize S
        # # get the active and reactive power from the load object
        # P = np.array(list(self.P.values()))  # convert to W
        # Q = np.array(list(self.Q.values()))  # convert to VAR
        # S = P + 1j * Q
        # i_phasors = (S / v_phasors).conjugate()
        # dyn_idx_i_start, dyn_idx_i_end = get_start_end_idx(self.var_offset_dynamic, "i", self.n_ph)
        # y0_dyn[dyn_idx_i_start:dyn_idx_i_end] = [np.sqrt(2) * phasor_to_timedomain(val) for val in i_phasors]

        return y0_dyn

    def initial_guess_dynamic(self, y_comp, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # powerflow vars: [w, V, I, v, i, lamda]
        # dyn vars: [w, V, I, v, i, lamda]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        y0_dyn[0] = y_comp[0].real
        # rest:
        y0_dyn[1:] = [np.sqrt(2) * phasor_to_timedomain(val) for val in y_comp[1:]]
        return y0_dyn

    def get_local_idx_dynamic(
        self, var: str, ph: str | None = None, side: NodeSide | None = None
    ):
        # since this load has the same variable in both powerflow and dynamic, just reusing get_local_idx() here
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self, stage=None) -> np.ndarray:
        """
         1)This function creates the M matrix for powerflow
         2) First we create the identity and coefficient matrices required for each eqn
        and then place them in the matrix using sps.bmat.
         3)Id_*: identity matrix
         4)Z_*: matrix of zeros

        """

        Id_ph = sps.identity(self.n_ph, format="coo")

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)

        if stage is None or stage == "stage1":
            # fmt: off
            M = sps.bmat(
                [   #w       V     I        v       i       lamda
                    [Z_w1,  None,  -Id_ph,  None,   Id_ph,  None ], # 1)-I + i = 0
                    [Z_w2,  Id_ph, None,    -Id_ph, None,   None ], # 2)V - v =0
                    [Z_w3,  None,  None,    None,   self.L,-Id_ph], # 3)Li - lamda =0
                    [Z_w4,  None,  None,    -Id_ph, self.R, None], # 4)-v + Ri + jw * lamda =0
                ]
            )
            # fmt: on

        elif stage == "stage2":
            # fmt: off
            M = sps.bmat(
                [   #w       V     I        v       i            lamda
                    [Z_w1,  None,  -Id_ph,  None,   Id_ph,       None ], # 1)-I + i = 0
                    [Z_w2,  Id_ph, None,    -Id_ph, None,        None ], # 2)V - v =0
                    [Z_w3,  None,  None,    None,   self.L,     -Id_ph], # 3)Li - lamda =0
                    [Z_w4,  None,  None,    -Id_ph, self.R/2, None], # 4)-v + Ri + jw * lamda =0
                ]
            )
            # fmt: on

        else:
            raise ValueError(f"unknown stage: {stage}")

        return M


    def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        u = sps.lil_matrix((self.num_eqns, 1), dtype=complex)
        return u.real, u[self.num_eqns_real :].imag

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Eqn 4 is non-linear because of the product of w and i
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im
        # create an empty matrix for fy
        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 4
        # start stop index eqn 4
        idx_eq4_start = (
            self.n_ph  # first set  :1)-I + Ai = 0
            + self.n_ph  # second set :2)A'V - v = 0
            + self.n_ph  # third set  :3)Li - lamda = 0
        )

        idx_eq4_end = idx_eq4_start + self.n_ph

        # start stop index of w and lamda in y vector
        idx_w = self.var_offset["w"]
        idx_lamda_start = self.var_offset["lamda"]
        idx_lamda_end = idx_lamda_start + self.n_ph

        fy[idx_eq4_start:idx_eq4_end, 0] = (
            1j * y[idx_w, 0] * y[idx_lamda_start:idx_lamda_end]
        )

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        # check that the variables received are of same shape as in the model
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        # get the index of w
        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]

        # get the index of lamda
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # eq4_re i.e. the real part of the pd of eqn 4
        eq4_re_start_row = self.n_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset

            # get the index of lamda in the y vector
            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = -lamda_im

            # lamda_re : derivative is zero

            # lamda_im
            pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

        # eq4_im i.e the imaginary part of the pd of eqn 4
        eq4_im_start_row = self.num_eqns + self.n_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_im_start_row + offset

            # get the index of lamda in the y vector
            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = lamda_re

            # lamda_re
            pd_fy_split[row, lamda_re_col_offset] = w
            # lamda_im : derivative is zero

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        """
        1. This function is to be called from the Optimal Powerflow.
        2. This function finds the Jacobian for the new non-linear vector formed by the multiplication of
           the lagrange multiplier and the transposed Jacobian of the non-linear vector of the powerflow eqns
        3. g(y) = pd_f(y).transpose() * lagm.
        4. This function returns the partial derivative of g(y) with respect to y for this component to be stacked
            appropriately in the overall Jacobian.
        """
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        idx_eq4 = (
            self.n_ph  # first set  :1)-I + i = 0
            + self.n_ph  # second set :2)V - v = 0
            + self.n_ph  # third set  :3)Li - lamda = 0
        )

        w_offset = self.var_offset["w"]
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.num_vars + self.var_offset_complex["lamda"]

        # eq4_re: (f(y) : -w * lamda_im)
        lagm = lagm_real[idx_eq4, 0]
        for offset in range(self.n_ph):
            row = w_offset
            col = lamda_im_start_offset + offset
            pd_gy_split[row, col] += -1 * lagm

        for offset in range(self.n_ph):
            row = lamda_im_start_offset + offset
            col = w_offset
            pd_gy_split[row, col] += -1 * lagm

        # eq4_im: (f(y) : w * lamda_re)
        lagm = lagm_imag[idx_eq4, 0]
        for offset in range(self.n_ph):
            row = w_offset
            col = lamda_re_start_offset + offset
            pd_gy_split[row, col] += 1 * lagm

        for offset in range(self.n_ph):
            row = lamda_re_start_offset + offset
            col = w_offset
            pd_gy_split[row, col] += 1 * lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    #######################################dynamic simulation functions###########################################################
    """
        1) The dynamic equations only require a K matrix for the coefficients of the
           dynamic state variables.
        2) The M matrix remains the same.
        3) The non-linear vector changes
        4) The input vector also changes
        """

    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        """
        1) This function creates the K matrix for dynamic equations
        2) First we create the identity and coefficient matrices required for each eqn
            and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros
        """
        Id_ph = sps.identity(self.n_ph, format="coo")
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)

        # fmt: off
        K = sps.bmat([
                #w       V     I     v       i      lamda
                [Z_w,  None,  None,  None,   Z_ph,  None ], # 1)-I + i = 0
                [Z_w,  Z_ph,  None,  None,   None,  None ], # 2)V - v =0
                [Z_w,  None,  Z_ph,  None,   None,  None], # 3)Li - lamda =0
                [Z_w,  None,  None,  Z_ph,   None,  Id_ph], # 4)-v + Ri + d(lamda)/dt =0

        ])
        # fmt: on

        assert K.shape[0] == self.num_eqns_dynamic
        assert K.shape[1] == self.num_vars_dynamic

        return K

    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        return self.get_M_powerflow_inner(stage)

    def get_fy_dynamic(self, t, y, yp, stage=None) -> sps.coo_array:
        fy = np.zeros(self.num_eqns_dynamic, dtype=float)
        # fy = sps.lil_matrix((self.num_eqns_dynamic, 1), dtype=float)
        return fy

    def get_u_dynamic(self, t: float, y: np.ndarray, stage=None) -> np.ndarray:
        u_dynamic = np.zeros(self.num_eqns_dynamic, dtype=float)
        return u_dynamic


###########################################################################################################################
#  Star Constant Current Load Model


class StarConstantCurrentLModel(LoadModel):
    def __init__(self, load_obj: Load):
        super().__init__(load_obj)

        # create a constant matrix for power factor
        self.pf_const = np.array(list(self.obj.power_factor.values())).reshape(-1, 1)
        self.pf_const_flat = self.pf_const.flatten()

        self.theta = np.arccos(self.pf_const).flatten()
        self.ipeak = np.sqrt(2) * self.obj.iconst.flatten()
        # self.ipeak = self.obj.iconst.flatten()
        self.irms = self.obj.iconst.flatten()
        # self.irms = self.obj.iconst.flatten() / np.sqrt(2)

        self.phi_init = None

        # calculate the number of equations in this load model

        # fmt:off
        self.num_eqns_real = (
            self.n_ph       # 1) pf - pf_const = 0
            + self.n_ph     # 2) iiconj - iconst^2 = 0
            + self.n_ph     # 3) Sreal / |S| - pf = 0
            + self.n_ph     # 4) iiconj - ii* = 0
        )
        self.num_eqns_complex = (
            self.n_ph       # 5) I - i = 0
            + self.n_ph     # 6) V - v =0
            + self.n_ph     # 7) -s + vi* =0
        )
        # fmt:on
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping for the variables
        # calculate the number of variable in this load model
        # fmt:off

        # y = [w, pf, iiconj, V, I, v, i, S]

        self.num_vars_real = (1 + self.n_ph + self.n_ph) # w, pf(per phase), iiconj(per phase)
        self.num_vars_complex = (
            self.n_ph   # V
            + self.n_ph # I
            + self.n_ph # v
            + self.n_ph # i
            + self.n_ph # S
        )

        self.num_vars = self.num_vars_real + self.num_vars_complex
        # fmt:on

        # create a dictionary to store the offset of each variable in the y vector
        # fmt:off
        # 1.offset for real variables
        self.var_offset_real = {
            "w": 0,
            "pf":1,
            "iiconj":1 + self.n_ph,
        }
        # 2.offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ph,
            "v": self.n_ph + self.n_ph,
            "i": self.n_ph + self.n_ph + self.n_ph,
            "S": self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # 3.offset for all variables
        self.var_offset = {
            "w": 0,
            "pf":1,
            "iiconj":1 + self.n_ph,
            "V": 1 + self.n_ph + self.n_ph,
            "I": 1 + self.n_ph + self.n_ph + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            "S": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # fmt:on

        # to check if the total number of variables is correct with respect to the offset
        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["S"] + self.n_ph

        # dynamic:
        # fmt: off
        self.num_eqns_dynamic = (
            self.n_ph   
            + self.n_ph
            + self.n_ph
            + self.n_ph
            + self.n_ph
            # + self.n_ph
            # + self.n_ph
            # + self.n_ph
        )

        # [w V I v i vp phi_v]
        self.num_vars_dynamic = (
            1
            + self.n_ph
            + self.n_ph
            + self.n_ph
            + self.n_ph
            + self.n_ph
            + self.n_ph
            # + self.n_ph
            # + self.n_ph
            # + self.n_ph
            # + self.n_ph
        )

        self.var_offset_dynamic = {
            "w": 0,
            "V": 1 ,
            "I": 1 + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
            "vp": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            "phi_v": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            # "theta": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            # "delta": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            # "P": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
            # "Q": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # assert self.var_offset_dynamic["Q"] + self.n_ph == self.num_vars_dynamic
        assert self.var_offset_dynamic["phi_v"] + self.n_ph == self.num_vars_dynamic
        # fmt: on

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    # TODO: For this model the constant current should come from the adapter
    # TODO: Also the power factor to be maintained should come from the adapter

    def get_M_powerflow_inner(self) -> sps.coo_array:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
           and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        Id_ph = sps.identity(self.n_ph, format="coo")
        Z_ph = sps.lil_matrix((Id_ph.shape[0], Id_ph.shape[0]), dtype=float)

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)

        # fmt: off
        M = sps.bmat([
            #[w       pf    iiconj   V       I       v         i        S]
            [Z_w5,  Id_ph,  None,    None,   None,    None,   None,   None],   #1 pf - pf_const = 0 #1
            [Z_w5,  None,  Id_ph,   None,   None,    None,    None,  None],    #2 iiconj - iconst^2 = 0 #2
            [Z_w4, -Id_ph,  None,    None,   None,   None,    None,   None ],  #3 Sreal/|S| - pf = 0 (Sreal/S non-linear) #6
            [Z_w5,  None,  Id_ph,    None,   None,    None,   None,   None ],  #4 iiconj - ii* = 0 (ii* non-linear) #7
            [Z_w1,  None,  None,    None,   -Id_ph,  None,    Id_ph,  None ],  #5 -I + Ai = 0 #3
            [Z_w2,  None,  None,    Id_ph,  None,    -Id_ph,  None,   None ],  #6 A'V - v =0 #4
            [Z_w3,  None,  None,    None,   None,    None,    None,  -Id_ph],  #7 -S + vi* =0 (vi* non-linear) #5
        ])
        # fmt: on

        return M

    def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        u = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        idx_eq1_start = 0
        idx_eq1_end = idx_eq1_start + self.n_ph
        u[idx_eq1_start:idx_eq1_end, 0] = -self.pf_const
        print(f">> pf_const: {self.pf_const}")

        idx_eq2_start = self.n_ph
        idx_eq2_end = idx_eq2_start + self.n_ph
        u[idx_eq2_start:idx_eq2_end, 0] = -(self.obj.iconst**2)
        print(f">> iconst**2: {self.obj.iconst**2}")

        return u.real, u[self.num_eqns_real :].imag

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Equation 3, 4 and 7 have non-linearity in them
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this load model.
        """
        # #[w       pf    iiconj   V       I       v         i        S]
        # [Z_w5,  Id_ph,  None,    None,   None,    None,   None,   None],   #1  pf - pf_const = 0 #1
        # [Z_w5,  None,  Id_ph,   None,   None,    None,    None,  None],    #2  iiconj - iconst^2 = 0 #2
        # [Z_w4, -Id_ph,  None,    None,   None,   None,    None,   None ],  #3  Sreal/|S| - pf = 0 (Sreal/S non-linear) #6
        # [Z_w5,  None,  Id_ph,    None,   None,    None,   None,   None ],  #4  iiconj - ii* = 0 (ii* non-linear) #7
        # [Z_w1,  None,  None,    None,   -Id_ph,  None,    Id_ph,  None ],  #5 -I + Ai = 0 #3
        # [Z_w2,  None,  None,    Id_ph,  None,    -Id_ph,  None,   None ],  #6) A'V - v =0 #4
        # [Z_w3,  None,  None,    None,   None,    None,    None,  -Id_ph],  #7 -S + vi* =0 (vi* non-linear) #5

        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # y = [w, pf, iiconj, V, I, v, i, S]

        # fy update for eqn #3 -----
        # start stop index of S in y vector
        idx_S_start = self.var_offset["S"]
        idx_S_end = idx_S_start + self.n_ph
        S = y[idx_S_start:idx_S_end, 0]

        # start stop index of eqn #3
        idx_eq3_start = self.n_ph + self.n_ph
        idx_eq3_end = idx_eq3_start + self.n_ph
        fy[idx_eq3_start:idx_eq3_end, 0] = S.real / np.abs(S)

        # fy update for eqn #4 -----
        # start stop index of i in y vector
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start:idx_i_end, 0]

        # start stop index of eqn 4
        idx_eq4_start = self.n_ph + self.n_ph + self.n_ph
        idx_eq4_end = idx_eq4_start + self.n_ph
        fy[idx_eq4_start:idx_eq4_end, 0] = -(i.multiply(i.conjugate()))

        # fy update for eqn #7 -----
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        v = y[idx_v_start:idx_v_end, 0]

        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start:idx_i_end, 0]

        idx_eq7_start = (
            self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph
        )
        idx_eq7_end = idx_eq7_start + self.n_ph
        fy[idx_eq7_start:idx_eq7_end, 0] = v.multiply(i.conjugate())

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        # #[w       pf    iiconj   V       I       v         i        S]
        # [Z_w5,  Id_ph,  None,    None,   None,    None,   None,   None],   #1 6) pf - pf_const = 0 #1
        # [Z_w5,  None,  Id_ph,   None,   None,    None,    None,  None],    #2 7) iiconj - iconst^2 = 0 #2
        # [Z_w4, -Id_ph,  None,    None,   None,   None,    None,   None ],  #3 4) Sreal/|S| - pf = 0 (Sreal/S non-linear) #6
        # [Z_w5,  None,  Id_ph,    None,   None,    None,   None,   None ],  #4 5) iiconj - ii* = 0 (ii* non-linear) #7
        # [Z_w1,  None,  None,    None,   -Id_ph,  None,    Id_ph,  None ],  #5 1)-I + Ai = 0 #3
        # [Z_w2,  None,  None,    Id_ph,  None,    -Id_ph,  None,   None ],  #6 2) A'V - v =0 #4
        # [Z_w3,  None,  None,    None,   None,    None,    None,  -Id_ph],  #7 3)-S + vi* =0 (vi* non-linear) #5

        # check that the variables received are of same shape as in the model
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # pd_fy update ofr eqn #3
        # eq3_re i.e. the real part of the pd of eqn 3
        eq3_re_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset

            # get the index of S in the y vector
            S_re_col_offset = self.var_offset["S"] + offset
            S_im_col_offset = self.var_offset_complex["S"] + offset
            S_re = y_real[S_re_col_offset, 0]
            S_im = y_imag[S_im_col_offset, 0]

            # S_re
            pd_fy_split[row, S_re_col_offset] = S_im**2 / (
                (S_re**2 + S_im**2) ** (3 / 2)
            )
            # S_im
            pd_fy_split[row, self.num_vars + S_im_col_offset] = (
                -S_re * S_im / ((S_re**2 + S_im**2) ** (3 / 2))
            )

        # eq3_im there is no imaginary part of of eqn 3 so no updates here

        # pd_fy update ofr eqn #4
        # eq4_re
        eq4_re_start_row = self.n_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset

            # get the index of i in the y vector
            i_re_col_offset = self.var_offset["i"] + offset
            i_im_col_offset = self.var_offset_complex["i"] + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # i_re
            pd_fy_split[row, i_re_col_offset] = -2 * i_re
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = -2 * i_im

        # eq4_im: there is no imaginary parto of eqn 4 so no updates here

        # pd_fy update ofr eqn #7
        # eq7_re
        eq7_re_start_row = (
            self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph
        )
        for offset in range(self.n_ph):
            row = eq7_re_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = i_re
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_im
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_re
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = v_im

        # eq7_im
        eq7_im_start_row = (
            self.num_eqns + self.n_ph + self.n_ph
        )  # num_eqns + num_complex eqns ONLY
        for offset in range(self.n_ph):
            row = eq7_im_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = -i_im
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_re
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_im
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = -v_re

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [V, I, w, v, i, S, pf, iiconj]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        print(f">> self.nominal_voltage: {self.nominal_voltage}")
        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        # intitialize S
        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q

        idx_S_start = self.var_offset["S"]
        idx_S_end = idx_S_start + self.n_ph
        y_0[idx_S_start:idx_S_end, 0] = S

        # # initialize i
        # phases = [ph for ph in self.get_phases() if ph != "N"]
        # i_dict = {}
        # for ph in phases:
        #     i_dict[ph] = self.obj.iconst[0][0]
        # i_phasors_dict = utils.get_vector_phasors(i_dict)
        # i_phasors =  np.array(list(i_phasors_dict.values())).reshape(-1, 1)
        # idx_i_start = self.var_offset["i"]
        # idx_i_end = idx_i_start + self.n_ph
        # y_0[idx_i_start:idx_i_end, 0] = i_phasors

        print(f">> S: {S}")
        print(f">> v_ph: {v_phasors}")
        i_phasors = (S.reshape(-1, 1) / v_phasors).conjugate()
        print(f">> i_phasors: {i_phasors}")
        # input("continue?")
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        y_0[idx_i_start:idx_i_end, 0] = i_phasors

        print(f">> i_phasors: {i_phasors}")
        # input("continue?")

        # initialize pf
        idx_pf_start = self.var_offset["pf"]
        idx_pf_end = idx_pf_start + self.n_ph
        y_0[idx_pf_start:idx_pf_end, 0] = np.array(
            list(self.obj.power_factor.values())
        ).reshape(-1, 1)

        # iiconj
        idx_iiconj_start = self.var_offset["iiconj"]
        idx_iiconj_end = idx_iiconj_start + self.n_ph
        y_0[idx_iiconj_start:idx_iiconj_end, 0] = self.obj.iconst**2

        return y_0

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        # eq3, eq4, eq7

        # eq3_re: (orig: Sreal/sqrt(Sreal^2 + Simag^2))
        eq3_re_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset
            lagm = lagm_real[row, 0]

            S_re_row = self.var_offset["S"] + offset
            S_re_col = S_re_row
            S_im_row = self.num_vars + self.var_offset_complex["S"] + offset
            S_im_col = S_im_row

            idx_S_re = self.var_offset["S"] + offset
            idx_S_im = self.var_offset_complex["S"] + offset
            S_re = y_real[idx_S_re, 0]
            S_im = y_imag[idx_S_im, 0]

            # S_re:
            val = -3 * S_re * S_im**2 / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_re_row, S_re_col] += val * lagm

            val = (2 * S_re**2 * S_im - S_im**3) / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_re_row, S_im_col] += val * lagm

            # S_im:
            val = (2 * S_re**2 * S_im - S_im**3) / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_im_row, S_re_col] += val * lagm

            val = (2 * S_im**2 * S_re - S_re**3) / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_im_row, S_im_col] += val * lagm

        # eq3_im: NA

        # eq4_re: (orig: -ii*)
        eq4_re_start_row = self.n_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset
            lagm = lagm_real[row, 0]

            i_re_row = self.var_offset["i"] + offset
            i_re_col = i_re_row
            i_im_row = self.num_vars + self.var_offset_complex["i"] + offset
            i_im_col = i_im_row

            # i_re:
            pd_gy_split[i_re_row, i_re_col] += -2 * lagm

            # i_im:
            pd_gy_split[i_im_row, i_im_col] += -2 * lagm

        # eq4_re: NA

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq7_re: (orig: v_re * i_re + v_im * i_im)
        eq7_re_start_row = 6 * self.n_ph
        for offset in range(self.n_ph):
            row = eq7_re_start_row + offset
            lagm = lagm_real[row, 0]

            # v_re:
            v_re_row = v_re_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[v_re_row, i_re_col] += 1 * lagm

            # i_re:
            i_re_row = i_re_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[i_re_row, v_re_col] += 1 * lagm

            # v_im:
            v_im_row = v_im_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[self.num_vars + v_im_row, self.num_vars + i_im_col] += 1 * lagm

            # i_im:
            i_im_row = i_im_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[self.num_vars + i_im_row, self.num_vars + v_im_col] += 1 * lagm

        # eq7_im: (orig: -v_re * i_im + v_im * i_re)
        eq7_im_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq7_im_start_row + offset
            lagm = lagm_imag[row, 0]

            # v_re:
            v_re_row = v_re_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[v_re_row, self.num_vars + i_im_col] += -1 * lagm

            # i_im:
            i_im_row = i_im_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[self.num_vars + i_im_row, v_re_col] += -1 * lagm

            # v_im:
            v_im_row = v_im_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[self.num_vars + v_im_row, i_re_col] += 1 * lagm

            # i_re:
            i_re_col = i_re_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[i_re_col, self.num_vars + v_im_col] += 1 * lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    #################################### dynamic simulation functions #################################
    """
    1) This function is used to get the K matrix for dynamic simulation
    2) Since there are no state variables in this model, the K matrix is a zero matrix

    """

    def get_local_idx_dynamic(
        self, var: str, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset

    def get_M_dynamic(self) -> sps.coo_array:
        Id_ph = sps.identity(self.n_ph, format="coo")
        Z_ph = sps.lil_matrix((Id_ph.shape[0], Id_ph.shape[0]), dtype=float)

        # zero vectors and matrices
        Z_w = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_b = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)

        # fmt: off
        # M = sps.bmat([
        #     # [w     V       I       v         i   , vp  ,   theta , delta , P     , Q     ]
        #     [Z_w,  None,   None,    None,   -Id_ph, None,   None  , None  , None  , None  ],  #1 -i + ipeak * cos(delta - theta) = 0
        #     [Z_w, None,   -Id_ph,   None,    Id_ph, None,   None  , None  , None  , None  ],  #2 -I + Ai = 0
        #     [Z_w, Id_ph,  None,    -Id_ph,    None, None,   None  , None  , None  , None  ],  #3 A'V - v = 0
        #     [Z_w, None,     None,   None,     None, None,   Z_b,    None,   None,   None  ],  #4 cos(theta) - pf = 0
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   Id_ph,  None  ],  #5 P - (1/sqrt(2)) * vp * irms * cos(theta) = 0
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   None,   Id_ph ],  #6 Q - (1/sqrt(2)) * vp * irms * sin(theta) = 0
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   None,   None  ],  #7 P / (sqrt(P**2 + Q**2)) - pf = 0
        #     [Z_w, None,     None,   Id_ph,    None, Z_b,    None,   Z_b,    None,   None  ],  #8 v - vp * cos(delta)= 0
        # ])
        # fmt: on

        # fmt:off
        M = sps.bmat([
            #[w     V       I       v         i   ,     vp  ,   phi_v,]
            [Z_w,  None,   None,    None,   -Id_ph,     None,   None, ],  #1 -i + ipeak * cos(wt + phi_v - theta ) = 0      (fy)
            [Z_w, None,   -Id_ph,   None,    Id_ph,     None,   None, ],  #2 -I + Ai = 0
            [Z_w, Id_ph,   None,    -Id_ph,  None,      None,   None, ],  #3 A'V - v = 0          
            [Z_w, None,    None,    Id_ph,   None,      Z_b,    None, ],  #8 v - vp * cos(wt + phi_v)= 0                    (fy)
            [Z_w, None,    None,    None,    None,      None,   Id_ph,]   #9 phi_v - phi_init = 0                           (u)
        ])
        # fmt: on

        return M

    # create an empty K matrix of shape M in sparse format
    def get_K_dynamic(self) -> sps.coo_array:
        K = sps.lil_matrix((self.num_eqns_dynamic, self.num_vars_dynamic), dtype=float)
        return K

    def get_fy_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        # shall have values for eqns 1
        fy = np.zeros(self.num_eqns_dynamic, dtype=float)

        # M = sps.bmat([
        #     #[w     V       I       v         i   , vp  ,   theta , delta , P     , Q     ]
        #     [Z_w,  None,   None,    None,   -Id_ph, None,   None  , None  , None  , None  ],  #1 -i + ipeak * cos(delta - theta) = 0      <- fy
        #     [Z_w, None,   -Id_ph,   None,    Id_ph, None,   None  , None  , None  , None  ],  #2 -I + Ai = 0
        #     [Z_w, Id_ph,   None,    -Id_ph,    None, None,   None  , None  , None  , None  ],  #3 A'V - v = 0
        #     [Z_w, None,    None,   None,     None, None,   None,   None,   None,   None  ],  #4 cos(theta) - pf = 0                       <- fy
        #     [Z_w, None,    None,   None,     None, None,   None,   None,   Id_ph,  None  ],  #5 P - (1/sqrt(2)) * vp * irms * cos(theta) = 0    <- fy
        #     [Z_w, None,    None,   None,     None, None,   None,   None,   None,   Id_ph ],  #6 Q - (1/sqrt(2)) * vp * irms * sin(theta) = 0    <- fy
        #     [Z_w, None,    None,   None,     None, None,   None,   None,   None,   None  ],  #7 P / (sqrt(P**2 + Q**2)) - pf = 0          <- fy
        #     [Z_w, None,    None,   Id_ph,    None, None,   None,   None,   None,   None  ],  #8 v - vp * cos(delta)= 0                     <- fy

        # ])

        # M = sps.bmat([
        #     #[w     V       I       v         i   ,     vp  ,   phi_v,]
        #     [Z_w,  None,   None,    None,   -Id_ph,     None,   None, ],  #1 -i + ipeak * cos(wt + phi_v - theta) = 0      (fy)
        #     [Z_w, None,   -Id_ph,   None,    Id_ph,     None,   None, ],  #2 -I + Ai = 0
        #     [Z_w, Id_ph,   None,    -Id_ph,  None,      None,   None, ],  #3 A'V - v = 0
        #     [Z_w, None,    None,    Id_ph,   None,      Z_b,    None, ],  #8 v - vp * cos(wt + phi_v)= 0                    (fy)
        #     [Z_w, None,    None,    None,    None,      None,   Id_ph,]   #9 phi_v - phi_init = 0                           (u)
        # ])

        # vars:
        idx_w = self.var_offset_dynamic["w"]
        w = y[idx_w]

        idx_phiv_start = self.var_offset_dynamic["phi_v"]
        idx_phiv_end = idx_phiv_start + self.n_ph
        phi_v = y[idx_phiv_start:idx_phiv_end]

        idx_vp_start = self.var_offset_dynamic["vp"]
        idx_vp_end = idx_vp_start + self.n_ph
        vp = y[idx_vp_start:idx_vp_end]

        # eqn1:
        idx_eq1_start = 0
        fy[idx_eq1_start : idx_eq1_start + self.n_ph] = self.ipeak * np.cos(
            w * t + phi_v - self.theta
        )

        # eqn4:
        idx_eq4_start = 3 * self.n_ph
        fy[idx_eq4_start : idx_eq4_start + self.n_ph] = -vp * np.cos(w * t + phi_v)

        return fy

    def get_u_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        u = np.zeros(self.num_eqns_dynamic, dtype=float)

        # M = sps.bmat([
        #     #[w     V       I       v         i   , vp  ,   theta , delta , P     , Q     ]
        #     [Z_w,  None,   None,    None,   -Id_ph, None,   None  , None  , None  , None  ],  #1 -i + ipeak * cos(delta - theta) = 0
        #     [Z_w, None,   -Id_ph,   None,    Id_ph, None,   None  , None  , None  , None  ],  #2 -I + Ai = 0
        #     [Z_w, Id_ph,  None,    -Id_ph,    None, None,   None  , None  , None  , None  ],  #3 A'V - v = 0
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   None,   None  ],  #4 cos(theta) - pf = 0                       <- u
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   Id_ph,  None  ],  #5 P - (1/sqrt(2)) * vp * irms * cos(theta) = 0
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   None,   Id_ph ],  #6 Q - (1/sqrt(2)) * vp * irms * sin(theta) = 0
        #     [Z_w, None,     None,   None,     None, None,   None,   None,   None,   None  ],  #7 P / (sqrt(P**2 + Q**2)) - pf = 0          <- u
        #     [Z_w, None,     None,   Id_ph,    None, None,   None,   None,   None,   None  ],  #8 v - vp * cos(delta)= 0                     <- fy
        # ])

        # M = sps.bmat([
        #     #[w     V       I       v         i   ,     vp  ,   phi_v,]
        #     [Z_w,  None,   None,    None,   -Id_ph,     None,   None, ],  #1 -i + ipeak * cos(wt + phi_v - theta) = 0      (fy)
        #     [Z_w, None,   -Id_ph,   None,    Id_ph,     None,   None, ],  #2 -I + Ai = 0
        #     [Z_w, Id_ph,   None,    -Id_ph,  None,      None,   None, ],  #3 A'V - v = 0
        #     [Z_w, None,    None,    Id_ph,   None,      Z_b,    None, ],  #8 v - vp * cos(wt + phi_v)= 0                    (fy)
        #     [Z_w, None,    None,    None,    None,      None,   Id_ph,]   #9 phi_v - phi_init = 0                           (u)
        # ])

        # eqn5:
        idx_eq5_start = 4 * self.n_ph
        u[idx_eq5_start : idx_eq5_start + self.n_ph] = -self.phi_init

        return u

    def initial_guess_dynamic(self, y_comp) -> np.ndarray:
        assert len(y_comp) == self.num_vars  # this is the powerflow y vector

        # powerflow vars: [w, pf, iiconj, V, I, v, i, S]
        # dynamic vars: [w, V, I, v, i]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        # initialization from powerflow directly
        vars_counts_real = [
            ("w", 1),
        ]
        for var, count in vars_counts_real:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = self.var_offset[var] + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = self.var_offset_dynamic[var] + count
            y0_dyn[idx_var_dyn_start:idx_var_dyn_end] = y_comp[
                idx_var_pf_start:idx_var_pf_end
            ]

        vars_counts_complex = [
            ("V", self.n_ph),
            ("I", self.n_ph),
            ("v", self.n_ph),
            ("i", self.n_ph),
        ]

        for var, count in vars_counts_complex:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = self.var_offset[var] + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = self.var_offset_dynamic[var] + count
            y0_dyn[idx_var_dyn_start:idx_var_dyn_end] = np.sqrt(
                2
            ) * phasor_to_timedomain(y_comp[idx_var_pf_start:idx_var_pf_end])

        # init by calculation
        vars_counts_calc = [("vp", self.n_ph), ("phi_v", self.n_ph)]

        # vp
        idx_v_pf_start = self.var_offset["v"]
        idx_v_pf_end = self.var_offset["v"] + self.n_ph
        idx_vp_dyn_start = self.var_offset_dynamic["vp"]
        idx_vp_dyn_end = self.var_offset_dynamic["vp"] + self.n_ph
        y0_dyn[idx_vp_dyn_start:idx_vp_dyn_end] = np.sqrt(2) * np.abs(
            y_comp[idx_v_pf_start:idx_v_pf_end]
        )

        # phi_v
        idx_phiv_dyn_start = self.var_offset_dynamic["phi_v"]
        idx_phiv_dyn_end = self.var_offset_dynamic["phi_v"] + self.n_ph
        self.phi_init = np.angle(y_comp[idx_v_pf_start:idx_v_pf_end])
        y0_dyn[idx_phiv_dyn_start:idx_phiv_dyn_end] = self.phi_init

        return y0_dyn


class StarConstantPowerLModel(LoadModel):
    def __init__(self, load_obj: Load):
        super().__init__(load_obj)

        self.num_eqns_real = 0
        self.num_eqns_complex = (
            # 1           # first set (w) :1)eqn for w
            self.n_ph  # second set    : 2)-I + i = 0
            + self.n_ph  # third set     : 3)V -v =0
            + self.n_ph  # fourth set    : 4)-S + vi* =0
            + self.n_ph  # fifth set     : 5)-S + u[] =0
        )
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book-keeping for the variables
        # fmt: off
        # y = [w, V, I, v, i, S]
        # self.vars_real = ["w"]
        # self.vars_complex = ["V", "I", "v", "i", "q", "lamda"]

        self.num_vars_real = (1) # w
        self.num_vars_complex = (
            self.n_ph   # V
            + self.n_ph   # I
            + self.n_ph   # v
            + self.n_ph   # i
            + self.n_ph   # S
        )
        self.num_vars = self.num_vars_real + self.num_vars_complex
        # fmt: on

        # 1. var offset for real variables
        self.var_offset_real = {"w": 0}
        # 2. var offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ph,
            "v": self.n_ph + self.n_ph,
            "i": self.n_ph + self.n_ph + self.n_ph,
            "S": self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # 3. var offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
            "S": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }

        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["S"] + self.n_ph

        # dynamic:
        # fmt: off
        self.num_eqns_dynamic = (
            self.n_ph  # 1) -I + Ai = 0
            + self.n_ph  # 2) A'V - v = 0
            # + self.n_ph #3) 
            # + self.n_ph #4)
            + self.n_ph #5) 
            + self.n_ph #6)
            + self.n_ph #7)
            + self.n_ph #8)
        )

        # y = [w V I v i vp ip phi_v]
        self.num_vars_dynamic = (
            1  # w
            + self.n_ph  # V
            + self.n_ph  # I
            + self.n_ph  # v
            + self.n_ph  # i
            # + self.n_ph  # pv # derivative of v
            # + self.n_ph  # pi # derivative of i
            + self.n_ph  # v_amp v amplitude
            + self.n_ph  # i_amp i amplitude
            + self.n_ph  # S_mag magnitude of complex power / cos_theta_i          
           
        )

        # self.var_offset_dynamic = {
        #     "w": 0,
        #     "V": 1,
        #     "I": 1 + self.n_ph,
        #     "v": 1 + self.n_ph + self.n_ph,
        #     "i": 1 + self.n_ph + self.n_ph + self.n_ph,
        #     "p_v": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        #     "p_i": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        #     "v_amp": 1 + self.n_ph * 6,
        #     "i_amp": 1 + self.n_ph * 7,
        #     "S_mag": 1 + self.n_ph * 8
            
        # }

        self.var_offset_dynamic = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,            
            "v_amp": 1 + self.n_ph * 4,
            "i_amp": 1 + self.n_ph * 5,
            # "S_mag": 1 + self.n_ph * 6, 
            "cos_theta_i": 1 + self.n_ph * 6,           
        }

        # assert self.var_offset_dynamic["S_mag"] + self.n_ph == self.num_vars_dynamic
        assert self.var_offset_dynamic["cos_theta_i"] + self.n_ph == self.num_vars_dynamic

        self.p = np.array([self.P[ph] for ph in self.get_phases()])

        self.q = np.array([self.Q[ph] for ph in self.get_phases()])
        print(f"p: {self.p}, q: {self.q}")

        mag = np.sqrt(self.p**2 + self.q**2)
        self.pf = (self.p / mag)  # power factor 
        self.theta = np.arccos(self.pf)  # angle in radians

        # tmp bases created to check if per unit works
        self.P_base = 1e6
        self.V_base = 2400*np.sqrt(2) # Vpeak
        self.I_base = self.P_base/self.V_base

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [V, I, w, v, i, S]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        # initialize v
        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        print(f">> load nominal_voltage: {self.nominal_voltage}")
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

        # initialize S
        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q
        idx_S_start = self.var_offset["S"]
        idx_S_end = idx_S_start + self.n_ph
        y_0[idx_S_start:idx_S_end, 0] = S

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        return y_0

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self) -> np.ndarray:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
           and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        Id_ph = sps.identity(self.n_ph, format="coo")

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)

        # fmt: off
        M = sps.bmat(
            [   #w      V      I          v       i       S                
                [Z_w1,  None,  -Id_ph,    None,   Id_ph,  None ],  # 1) -I + i = 0
                [Z_w2,  -Id_ph,None,      Id_ph,  None,   None ],  # 2)  V - v =0
                [Z_w3,  None,  None,      None,   None,  -Id_ph],  # 3) -S + vi* =0
                [Z_w4,  None,  None,      None,   None,  -Id_ph],  # 4) -S + u[] =0
            ]
        )
        # fmt: on

        return M

    # return [comp_fy_re, comp_fy_im]
    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Eqn 4 is non-linear because of the product of v and i
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 4
        # start stop index of v in y vector
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        v = y[idx_v_start:idx_v_end, 0]
        # print(f"type(v): {type(v)}")
        # print(f"v.shape: {v.shape}")

        # start stop index of i in y vector
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start:idx_i_end, 0]
        # print(f"type(i): {type(i)}")

        # start stop index of eqn 4
        # fmt: off
        idx_eq3_start = (
            self.n_ph # second set 2)-I + i = 0
            + self.n_ph # third set  3)V - v =0
        )
        # fmt: on
        idx_eq3_end = idx_eq3_start + self.n_ph
        # print(f">> v: \n{v.toarray()}")
        # print(f">> i: \n{i.toarray()}")

        fy[idx_eq3_start:idx_eq3_end, 0] = v.multiply(i.conjugate())
        # input("continue?")

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq3_re: (orig: v_re * i_re + v_im * i_im)
        eq3_re_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset
            lagm = lagm_real[row, 0]

            # v_re:
            v_re_row = v_re_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[v_re_row, i_re_col] += 1 * lagm

            # i_re:
            i_re_row = i_re_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[i_re_row, v_re_col] += 1 * lagm

            # v_im:
            v_im_row = v_im_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[self.num_vars + v_im_row, self.num_vars + i_im_col] += 1 * lagm

            # i_im:
            i_im_row = i_im_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[self.num_vars + i_im_row, self.num_vars + v_im_col] += 1 * lagm

        # eq3_im: (orig: -v_re * i_im + v_im * i_re)
        eq3_im_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_im_start_row + offset
            lagm = lagm_imag[row, 0]

            # v_re:
            v_re_row = v_re_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[v_re_row, self.num_vars + i_im_col] += -1 * lagm

            # i_im:
            i_im_row = i_im_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[self.num_vars + i_im_row, v_re_col] += -1 * lagm

            # v_im:
            v_im_row = v_im_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[self.num_vars + v_im_row, i_re_col] += 1 * lagm

            # i_re:
            i_re_col = i_re_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[i_re_col, self.num_vars + v_im_col] += 1 * lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq3_re i.e the real part of eqn 3
        eq3_re_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = i_re
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_im
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_re
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = v_im

        # eq3_im i.e. the imaginary part of eqn 3
        eq3_im_start_row = self.num_eqns + eq3_re_start_row
        for offset in range(self.n_ph):
            row = eq3_im_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = -i_im
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_re
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_im
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = -v_re

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_u_powerflow(self) -> np.ndarray:
        """
        Load model u has constant power S consumed by the load
        """

        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype="float")

        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q

        # update u for eqn 5
        # start stop index of eqn 5 in u vector
        # fmt: off
        idx_eq4_re_start = (
            # 1           # first set  :1)eqn for w
            self.n_ph # second set :1)-I + i = 0
            + self.n_ph # third set  :2)V - v =0
            + self.n_ph # fourth set :3) -S + vi* =0
        )
        # fmt: on
        idx_eq4_re_end = idx_eq4_re_start + self.n_ph
        u[idx_eq4_re_start:idx_eq4_re_end, 0] = S.real

        idx_eq4_im_start = self.num_eqns + idx_eq4_re_start
        idx_eq4_im_end = idx_eq4_im_start + self.n_ph
        u[idx_eq4_im_start:idx_eq4_im_end, 0] = S.imag

        return u[: self.num_eqns], u[self.num_eqns :]

    ###############################dynamic simulation functions#################################
    """
    1) This block provides the matrices and vectors for dynamic simulation constant PQ
     

    """
    # fmt:off

    def get_local_idx_dynamic(self, var : str, ph: str | None, side: NodeSide | None) -> int:
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None
        
        side_offset = 0 # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset

    def initial_guess_dynamic(self, y_comp) -> np.ndarray:
        assert (
            len(y_comp) == self.num_vars
        )  # this is the powerflow y vector for this model

        # powerflow y vector is [w, V, I, v, i, S]
        # dynamic y vector is [w, V, I, v, i, vp, ip, delta]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        # initialization from powerflow
        vars_counts_real = [
            ("w", 1),
        ]
        for var, count in vars_counts_real:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = self.var_offset[var] + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = self.var_offset_dynamic[var] + count
            y0_dyn[idx_var_dyn_start:idx_var_dyn_end] = y_comp[
                idx_var_pf_start:idx_var_pf_end
            ]
        vars_counts_complex = [
            
            ("V", self.n_ph),
            ("I", self.n_ph),
            ("v", self.n_ph),
            ("i", self.n_ph),
        ]

        for var, count in vars_counts_complex:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count
            y0_dyn[idx_var_dyn_start:idx_var_dyn_end] = np.sqrt(2) * phasor_to_timedomain(
                y_comp[idx_var_pf_start:idx_var_pf_end]
            )

        # initialization by computation
        # [vp, ip, delta]

        # vp
        idx_v_pf_start = self.var_offset["v"]
        idx_v_pf_end = idx_v_pf_start + self.n_ph
        v = y_comp[idx_v_pf_start:idx_v_pf_end]
        idx_vp_dyn_start = self.var_offset_dynamic["vp"]
        idx_vp_dyn_end = idx_vp_dyn_start + self.n_ph
        y0_dyn[idx_vp_dyn_start:idx_vp_dyn_end] = np.sqrt(2) * np.abs(v)  # v_peak
        print(f">> v_init : {v}")
        
        # ip
        idx_i_pf_start = self.var_offset["i"]
        idx_i_pf_end = idx_i_pf_start + self.n_ph
        i = y_comp[idx_i_pf_start:idx_i_pf_end]
        idx_ip_dyn_start = self.var_offset_dynamic["ip"]
        idx_ip_dyn_end = idx_ip_dyn_start + self.n_ph
        y0_dyn[idx_ip_dyn_start:idx_ip_dyn_end] = np.sqrt(2) * np.abs(i)  # i_peak

        # # phi_v
        # idx_phiv_dyn_start = self.var_offset_dynamic["phi_v"]
        # idx_phiv_dyn_end = self.var_offset_dynamic["phi_v"] + self.n_ph
        # self.phi_init = np.angle(y_comp[idx_v_pf_start:idx_v_pf_end])
        # y0_dyn[idx_phiv_dyn_start:idx_phiv_dyn_end] = self.phi_init

        # # theta
        # idx_theta_dyn_start = self.var_offset_dynamic["theta"]
        # idx_theta_dyn_end = idx_theta_dyn_start + self.n_ph
        # y0_dyn[idx_theta_dyn_start:idx_theta_dyn_end] = np.arccos(
        #     self.pf
        # )  # pf = cos(theta)

        # # delta
        idx_delta_dyn_start = self.var_offset_dynamic["delta"]
        idx_delta_dyn_end = idx_delta_dyn_start + self.n_ph
        y0_dyn[idx_delta_dyn_start:idx_delta_dyn_end] = np.angle(v)

        return y0_dyn

    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        Id_ph = sps.identity(self.n_ph, format="coo")  # also A for eqn 1 and 2

        # zero vectors fand matrices:
        Z_w = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)

        # fmt: off
        # M = sps.bmat([
        #     #[w     V       I       v         i,       vp  ,   ip      theta , delta ]
        #     [Z_w,  None,    -Id_ph,  None,    Id_ph,  None,   None,   None,   None  ],  #1 -I + Ai = 0
        #     [Z_w,  Id_ph,   None,   -Id_ph,   None,   None,   None,   None,   None  ],  #2 A.T*V - v = 0
        #     [Z_w,  None,    None,   -Id_ph,   None,   Z_ph,   None,   None,   None  ],  #3 -v + vp*cos(delta) = 0 (fy)
        #     [Z_w,  None,    None,   None,     -Id_ph, None,   Z_ph,   None,   Z_ph, ],  #4 -i + ip*cos(delta - theta) = 0 (fy)
        #     [Z_w,  None,    None,   None,     None,   None,   None,   Z_ph,   None ],   #5 (vp*ip)/2*cos(theta) - P = 0 (fy) (u)
        #     [Z_w,  None,    None,   None,     None,   None,   None,   None,   None, ],  #6 (vp*ip)/2*sin(theta) - Q = 0 (fy) (u)
        #     [Z_w,  None,    None,   None,     None,   None,   None,   None,   None, ],  #7 cos(theta) - pf = 0 (fy) (u)
               
        # ])
        # fmt: on

        # M = sps.bmat([
        #     #[w     V       I       v         i,       vp  ,   ip,     delta]
        #     [Z_w,  None,    -Id_ph,  None,    Id_ph,  None,   None,    None ],  #1 -I + Ai = 0
        #     [Z_w,  Id_ph,   None,   -Id_ph,   None,   None,   None,    None ],  #2 A.T*V - v = 0
        #     [Z_w,  None,    None,   -Id_ph,   None,   Z_ph,   None,    Z_ph ],  #3 -v + vp*cos(delta) = 0 (fy)
        #     [Z_w,  None,    None,   None,     -Id_ph, None,   Z_ph,    None ],  #4 -i + ip*cos(delta - theta) = 0 (fy)
        #     [Z_w,  None,    None,   None,     None,   None,   None,    None ],  #5 (1/2)*(vp*ip)*cos(theta) - P = 0 (fy) (u)
        #     [Z_w,  None,    None,   None,     None,   None,   None,    None],   #6) (1/2)*(vp*ip)*sin(theta) - Q = 0
               
        # ])

        # # fmt:off
        # M = sps.bmat([
        #     #[w     V       I        v         i,     p_v,    p_i,     v_amp_pu  i_amp_pu, S_mag_pu,]
        #     [Z_w,  None,    -Id_ph,  None,    Id_ph,  None,   None,    None,     None,     None, ],  #1 -I + Ai = 0
        #     [Z_w,  Id_ph,   None,   -Id_ph,   None,   None,   None,    None,     None,     None, ],  #2 A.T*V - v = 0
        #     [Z_w,  None,    None,    None,    None,   -Id_ph, None,    None,     None,     None, ],  #3 -p_v + d(v)/dt = 0 (K)
        #     [Z_w,  None,    None,    None,    None,   None,   -Id_ph,  None,     None,     None, ],  #4 -p_i + d(i)/dt = 0 (K))
        #     [Z_w,  None,    None,    None,    None,   None,   None,   -Id_ph,    None,     None, ],  #5 -v_amp_pu + (np.sqrt(v^2 + (p_v/w)^2)/V_base =0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,     -Id_ph,   None, ],  #6 -i_amp_pu + (np.sqrt(i^2 + (p_i/w)^2)/I_base =0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,     None,    -Id_ph,],  #7 -S_mag_pu + (v_amp*i_amp)/2 = 0
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,     None,    -Id_ph,],  #8 -S_mag_pu + [u] = 0 (u)      
        # ])

        # # fmt:on

        # # fmt:off
        # M = sps.bmat([
        #     #[w     V       I        v         i,      v_amp     i_amp,    S_mag,]
        #     [Z_w,  None,    -Id_ph,  None,    Id_ph,   None,     None,     None, ],  #1 -I + Ai = 0
        #     [Z_w,  Id_ph,   None,   -Id_ph,   None,    None,     None,     None, ],  #2 A.T*V - v = 0           
        #     [Z_w,  None,    None,    None,    None,   -Id_ph,    None,     None, ],  #3 -v_amp_pu + (np.sqrt(v^2 + (p_v/w)^2)/V_base =0 (fy)
        #     [Z_w,  None,    None,    None,    None,    None,     -Id_ph,   None, ],  #4 -i_amp_pu + (np.sqrt(i^2 + (p_i/w)^2)/I_base =0 (fy)
        #     [Z_w,  None,    None,    None,    None,    None,     None,    -Id_ph,],  #5 -S_mag_pu + (v_amp*i_amp)/2 = 0
        #     [Z_w,  None,    None,    None,    None,    None,     None,    -Id_ph,],  #6 -S_mag_pu + [u] = 0 (u)      
        # ])
        # # fmt:on

        # fmt:off
        M = sps.bmat([
            #[w     V       I        v         i,      v_amp     i_amp,   cos_theta_i,]
            [Z_w,  None,    -Id_ph,  None,    Id_ph,   None,     None,    None,  ],  #1 -I + Ai = 0
            [Z_w,  Id_ph,   None,   -Id_ph,   None,    None,     None,    None,  ],  #2 A.T*V - v = 0           
            [Z_w,  None,    None,    None,    None,   -Id_ph,    None,    None,  ],  #3 -v_amp + (np.sqrt(v^2 + (v_p/w)^2) =0 (fy)
            [Z_w,  None,    None,    None,    None,    None,     Z_ph,    None,  ],  #4 -(v_amp*i_amp)/2 + [u] = 0 (fy) (u)
            [Z_w,  None,    None,    None,    None,    None,     None,    -Id_ph,],  #5 -cos_theta_i + (v/v_amp*cos(phi) + v'/w*v_amp*(sin(phi)) = 0 (fy)    
            [Z_w,  None,    None,    None,    -Id_ph,  None,     None,    None   ],  #6 -i + i_amp*cos_theta_i (fy)  
        ])
        # fmt:on
        
        return M
    # Note: assuming inductive load so current is lagging.


    # no differential equations in this model
    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        Id_ph = sps.identity(self.n_ph, format="coo")  # also A for eqn 1 and 2

        # zero vectors fand matrices:
        Z_w = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)

        # # fmt:off
        # K = sps.bmat([
        #     #[w     V       I        v         i,     p_v,    p_i,     v_amp  i_amp, S_mag,]
        #     [Z_w,  None,    Z_ph,    None,    Z_ph,   None,   None,    None,  None,  None, ],  #1 -I + Ai = 0
        #     [Z_w,  Z_ph,    None,    Z_ph,    None,   None,   None,    None,  None,  None, ],  #2  A.T*V - v = 0
        #     [Z_w,  None,    None,    Id_ph,   None,   Z_ph,   None,    None,  None,  None, ],  #3 -p_v + d(v)/dt = 0 (K)
        #     [Z_w,  None,    None,    None,    Id_ph,  None,   Z_ph,    None,  None,  None, ],  #4 -p_i + d(i)/dt = 0 (K))
        #     [Z_w,  None,    None,    None,    None,   None,   None,    Z_ph,  None,  None, ],  #5 -v_amp + np.sqrt(v^2 + (p_v/w)^2 =0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,  Z_ph,  None, ],  #6 -i_amp + np.sqrt(i^2 + (p_i/w)^2 =0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,  None,  Z_ph, ],  #7 -S_mag + (v_amp*i_amp)/2 = 0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,  None,  Z_ph, ],  #8 -S_mag + [u] = 0     (u)          
        # ])

        # # fmt:off
        # K = sps.bmat([
        #     #[w     V       I        v         i,      v_amp  i_amp, S_mag,]
        #     [Z_w,  None,    Z_ph,    None,    Z_ph,    None,  None,  None, ],  #1 -I + Ai = 0
        #     [Z_w,  Z_ph,    None,    Z_ph,    None,    None,  None,  None, ],  #2  A.T*V - v = 0            
        #     [Z_w,  None,    None,    None,    None,    Z_ph,  None,  None, ],  #3 -v_amp + np.sqrt(v^2 + (p_v/w)^2 =0 (fy)
        #     [Z_w,  None,    None,    None,    None,    None,  Z_ph,  None, ],  #4 -i_amp + np.sqrt(i^2 + (p_i/w)^2 =0 (fy)
        #     [Z_w,  None,    None,    None,    None,    None,  None,  Z_ph, ],  #5 -S_mag + (v_amp*i_amp)/2 = 0 (fy)
        #     [Z_w,  None,    None,    None,    None,    None,  None,  Z_ph, ],  #6 -S_mag + [u] = 0     (u)          
        # ])

        # # fmt:on

         # fmt:off
        K = sps.bmat([
            #[w     V       I        v         i,      v_amp  i_amp, cos_theta_i,]
            [Z_w,  None,    Z_ph,    None,    Z_ph,    None,  None,  None, ],  #1 -I + Ai = 0
            [Z_w,  Z_ph,    None,    Z_ph,    None,    None,  None,  None, ],  #2  A.T*V - v = 0            
            [Z_w,  None,    None,    None,    None,    Z_ph,  None,  None, ],  #3 -v_amp + np.sqrt(v^2 + (p_v/w)^2 =0 (fy)
            [Z_w,  None,    None,    None,    None,    None,  Z_ph,  None, ],  #4 -(v_amp*i_amp)/2 + [u] = 0 (fy) (u)
            [Z_w,  None,    None,    None,    None,    None,  None,  Z_ph, ],  #5 -cos_theta_i + (v/v_amp*cos(phi) - v'/w*v_amp*(sin(phi)) = 0 (fy) 
            [Z_w,  None,    None,    None,    None,    None,  None,  Z_ph, ],  #6 -i + i_amp*cos_theta_i (fy) 
        ])
        # fmt:on
        return K

    def get_fy_dynamic(self, t: float, y: np.ndarray, yp: np.ndarray, stage=None) -> np.ndarray:

        #  # fmt:off
        # M = sps.bmat([
        #     #[w     V       I        v         i,     p_v,    p_i,     v_amp  i_amp, S_mag,]
        #     [Z_w,  None,    -Id_ph,  None,    Id_ph,  None,   None,    None,   None,  None, ],  #1 -I + Ai = 0
        #     [Z_w,  Id_ph,   None,   -Id_ph,   None,   None,   None,    None,   None,  None, ],  #2 A.T*V - v = 0
        #     [Z_w,  None,    None,    None,    None,   -Id_ph, None,    None,   None,  None, ],  #3 -p_v + d(v)/dt = 0 (K)
        #     [Z_w,  None,    None,    None,    None,   None,   -Id_ph,  None,   None,  None, ],  #4 -p_i + d(i)/dt = 0 (K))
        #     [Z_w,  None,    None,    None,    None,   None,   None,   -Id_ph,  None,  None, ],  #5 -v_amp + np.sqrt(v^2 + (p_v/w)^2 =0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,  -Id_ph, None, ],  #6 -i_amp + np.sqrt(i^2 + (p_i/w)^2 =0 (fy)
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,  None,  -Id_ph,],  #7 -S_mag + (v_amp*i_amp)/2 = 0
        #     [Z_w,  None,    None,    None,    None,   None,   None,    None,  None,  -Id_ph,],  #8 -S_mag + [u] = 0               
        # ])

        # fmt:on
             
               
        assert y.shape[0] == self.num_vars_dynamic
        # assert y.shape[1] == 1

        fy = np.zeros(self.num_eqns_dynamic, dtype=float)

        idx_w = self.var_offset_dynamic["w"]        
        w = 2*np.pi*60

        cos_phi = self.p/(np.sqrt(self.p**2 + self.q**2))
        sin_phi = self.q/(np.sqrt(self.p**2 + self.q**2))

        idx_v_start = self.var_offset_dynamic["v"]
        idx_v_end = idx_v_start + self.n_ph
        v = y[idx_v_start : idx_v_end]
        print(f"value of v : {v}")  
        

        # idx_p_v_start = self.var_offset_dynamic["p_v"]
        # idx_p_v_end = idx_p_v_start + self.n_ph
        # p_v = y[idx_p_v_start : idx_p_v_end]
        # print(f"value of p_v : {p_v}")
        
        v_p = yp[idx_v_start : idx_v_end]
        print(f"value of v_p : {v_p}")

        idx_i_start = self.var_offset_dynamic["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start : idx_i_end]
        print(f"value of i : {i}")

        # idx_p_i_start = self.var_offset_dynamic["p_i"]
        # idx_p_i_end = idx_p_i_start + self.n_ph
        # p_i = y[idx_p_i_start : idx_p_i_end]
        # print(f"value of p_i : {p_i}")
        
        i_p = yp[idx_i_start : idx_i_end]
        print(f"value of i_p : {i_p}")

        idx_v_amp_start = self.var_offset_dynamic["v_amp"]
        idx_v_amp_end = idx_v_amp_start + self.n_ph
        v_amp = y[idx_v_amp_start : idx_v_amp_end]

        idx_i_amp_start = self.var_offset_dynamic["i_amp"]
        idx_i_amp_end = idx_i_amp_start + self.n_ph
        i_amp = y[idx_i_amp_start : idx_i_amp_end]

        idx_cos_theta_i_start = self.var_offset_dynamic["cos_theta_i"]
        idx_cos_theta_i_end = idx_cos_theta_i_start + self.n_ph
        cos_theta_i = y[idx_cos_theta_i_start : idx_cos_theta_i_end]

        print(f"cos_theta_i: {cos_theta_i}")

        # idx_eqn5_start = 4 * self.n_ph
        # idx_eqn5_end = idx_eqn5_start + self.n_ph
        # fy[idx_eqn5_start : idx_eqn5_end] = np.sqrt(v**2 + (p_v/w)**2)

        # idx_eqn6_start = 5 * self.n_ph
        # idx_eqn6_end = idx_eqn6_start + self.n_ph
        # fy[idx_eqn6_start : idx_eqn6_end] = np.sqrt(i**2 + (p_i/w)**2)

        idx_eqn3_start = 2 * self.n_ph
        idx_eqn3_end = idx_eqn3_start + self.n_ph
        fy[idx_eqn3_start : idx_eqn3_end] = np.sqrt(v**2 + (v_p/w)**2)

        # idx_eqn4_start = 3 * self.n_ph
        # idx_eqn4_end = idx_eqn4_start + self.n_ph
        # fy[idx_eqn4_start : idx_eqn4_end] = np.sqrt(i**2 + (i_p/w)**2)


        # idx_eqn7_start = 6 * self.n_ph
        # idx_eqn7_end = idx_eqn7_start + self.n_ph
        # print(f"shape of fy[eqn7] :{ fy[idx_eqn7_start : idx_eqn7_end].shape}")
        # print(f"v_amp : {v_amp}")
        # print(f"i_amp : {i_amp}")
        # fy[idx_eqn7_start : idx_eqn7_end] = (v_amp * i_amp)/2
        # input("continue?")

        # idx_eqn5_start = 4 * self.n_ph
        # idx_eqn5_end = idx_eqn5_start + self.n_ph
        # print(f"shape of fy[eqn5] :{ fy[idx_eqn5_start : idx_eqn5_end].shape}")
        # print(f"v_amp : {v_amp}")
        # print(f"i_amp : {i_amp}")
        # fy[idx_eqn5_start : idx_eqn5_end] = (v_amp * i_amp)/2
        # # input("continue?")

        idx_eqn4_start = 3 * self.n_ph
        idx_eqn4_end = idx_eqn4_start + self.n_ph
        print(f"shape of fy[eqn4] :{ fy[idx_eqn4_start : idx_eqn4_end].shape}")
        print(f"v_amp : {v_amp}")
        print(f"i_amp : {i_amp}")
    
        fy[idx_eqn4_start : idx_eqn4_end] = -(v_amp * i_amp)/2

        idx_eqn5_start = 4 * self.n_ph
        idx_eqn5_end = idx_eqn5_start + self.n_ph
        fy[idx_eqn5_start : idx_eqn5_end] = (v/v_amp)*cos_phi - (v_p / (w*v_amp))*sin_phi

        idx_eqn6_start = 5 * self.n_ph
        idx_eqn6_end = idx_eqn6_start + self.n_ph
        fy[idx_eqn6_start : idx_eqn6_end] = i_amp*cos_theta_i       

        return fy

    def get_u_dynamic(self, t: float, y: np.ndarray) -> np.ndarray:
        
        # eqn8 : -S_mag + [u] = 0

        u = np.zeros(self.num_eqns_dynamic, dtype=float)

        # # eqn 8 :
        # idx_eqn8_start = 7 * self.n_ph
        # idx_eqn8_end = idx_eqn8_start + self.n_ph
        # u[idx_eqn8_start : idx_eqn8_end] = np.sqrt(self.p**2 + self.q**2)

        # # eqn 6:
        # idx_eqn6_start = 5* self.n_ph
        # idx_eqn6_end = idx_eqn6_start + self.n_ph
        # u[idx_eqn6_start : idx_eqn6_end] = np.sqrt(self.p**2 + self.q**2)
        # print(f"u[idx_eqn6_start : idx_eqn6_end]: {u[idx_eqn6_start : idx_eqn6_end]}")

        idx_eqn4_start = 3* self.n_ph
        idx_eqn4_end = idx_eqn4_start + self.n_ph
        u[idx_eqn4_start : idx_eqn4_end] = np.sqrt(self.p**2 + self.q**2)
        print(f"u[idx_eqn4_start : idx_eqn4_end]: {u[idx_eqn4_start : idx_eqn4_end]}")
    
        return u
    
    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ):
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn
    
    def initial_yp_dynamic(self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom):
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn

    def initial_guess_dynamic_zero(self, y_comp, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # powerflow vars: [w, V, I, v, i, lamda]
        # dyn vars: [w, V, I, v, i, lamda]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        
        # w
        y0_dyn[0] = y_comp[0]. real

        # # S_mag_pu
        # idx_S_mag_start = self.var_offset_dynamic["S_mag"]
        # idx_S_mag_end = idx_S_mag_start + self.n_ph
        # y0_dyn[idx_S_mag_start : idx_S_mag_end] = np.sqrt(self.p**2 + self.q**2)

        # # init "v" and "i":
        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values()))
        # dyn_idx_v_start, dyn_idx_v_end = get_start_end_idx(self.var_offset_dynamic, "v", self.n_ph)
        # y0_dyn[dyn_idx_v_start:dyn_idx_v_end] = [np.sqrt(2) * phasor_to_timedomain(val) for val in v_phasors]


        # V_amp_pu
        idx_v_amp_start = self.var_offset_dynamic["v_amp"]
        idx_v_amp_end = idx_v_amp_start + self.n_ph
        y0_dyn[idx_v_amp_start : idx_v_amp_end] = [np.sqrt(2) * np.abs(val) for val in v_phasors]

        # v
        dyn_idx_v_start, dyn_idx_v_end = get_start_end_idx(self.var_offset_dynamic, "v", self.n_ph)
        y0_dyn[dyn_idx_v_start:dyn_idx_v_end] = [np.sqrt(2) * phasor_to_timedomain(val) for val in v_phasors]

        # # S = P + jQ
        # S = self.p + 1j * self.q
        # i_phasors = (S / v_phasors).conjugate()
        # # i_phasors = -i_phasors
        # print(f">> init i_phasors: {i_phasors}")
        # # input("continue?")

        # # I_amp_pu
        # idx_i_amp_start = self.var_offset_dynamic["i_amp"]
        # idx_i_amp_end = idx_i_amp_start + self.n_ph
        # y0_dyn[idx_i_amp_start : idx_i_amp_end] = [np.sqrt(2) * np.abs(val) for val in i_phasors]
        # print(f"init i_amp: {y0_dyn[idx_i_amp_start : idx_i_amp_end]}")
        # # input("continue?")

        # # i
        # dyn_idx_i_start, dyn_idx_i_end = get_start_end_idx(self.var_offset_dynamic, "i", self.n_ph)
        # y0_dyn[dyn_idx_i_start:dyn_idx_i_end] = [np.sqrt(2) * phasor_to_timedomain(val) for val in i_phasors]

        # # # cos_theta_i
        # # dyn_costhetai_start, dyn_costhetai_end = get_start_end_idx(self.var_offset_dynamic, "cos_theta_i", self.n_ph)
        # # y0_dyn[dyn_costhetai_start:dyn_costhetai_end] = np.cos(np.angle(i_phasors))
    
        return y0_dyn

    def initial_guess_dynamic(self, y_comp, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        # powerflow vars: [w, V, I, v, i, lamda]
        # dyn vars: [w, V, I, v, i, lamda]

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return y0_dyn


    ###############################Optimal Powerflow functions#################################
    def get_hy_powerflow(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array] | None:
        return None

    def get_pd_hy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array] | None:
        return None

    def get_pd_objective_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        re = sps.lil_matrix((self.num_vars, 1), dtype=float)
        im = sps.lil_matrix((self.num_vars_complex, 1), dtype=float)

        return re, im

    def get_pd_pd_objective_split(
        self,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        pd_pd_obj_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        rr = pd_pd_obj_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_pd_obj_split[0 : self.num_vars, self.num_vars :]
        ir = pd_pd_obj_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_pd_obj_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_pd_hy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        mu_real: sps.coo_array,
        mu_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array] | None:
        return None

    #################################################################################################
    # Delta connected load models
    #################################################################################################
    # 1. Delta Constant Impedance Load Model (LoadModel)


class DeltaConstantImpedanceLModel(LoadModel):
    def __init__(self, load_obj: Load):
        super().__init__(load_obj)

        # calculate the number of equations:

        self.n_ext_ph = 2 if self.n_ph == 1 else 3

        # book keeping of the equations
        # fmt:off
        self.num_eqns_real = 0
        self.num_eqns_complex= (self.n_ext_ph # first set :1) -I + Ai = 0
                       + self.n_ph # second set: 2) A'V - v =0
                       + self.n_ph # third set: 3) Li - lamda = 0
                       + self.n_ph) # fourth set : 4)-v + Ri + jw * lamda = 0
        # fmt: on
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping of variables
        # fmt: off
        # y = [ w, V, I, i, v, lamda]
        # self.vars_real = ["w"]
        # self.vars_complex = ["V", "I", "v", "i", "lamda"]

        self.num_vars_real = (1) # w
        self.num_vars_complex = (
            self.n_ext_ph   # V
            + self.n_ext_ph # I            
            + self.n_ph # v
            + self.n_ph # i
            + self.n_ph # lamda
        )
        self.num_vars = self.num_vars_real + self.num_vars_complex
        # fmt: on

        # create  dictionaries to store the offset of each variable in the y vector
        # fmt: off
        # 1. offset for real variables
        self.var_offset_real = {"w": 0}
        # 2. offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ext_ph,
            "v": self.n_ext_ph + self.n_ext_ph,
            "i": self.n_ext_ph + self.n_ext_ph + self.n_ph,
            "lamda": self.n_ext_ph + self.n_ext_ph + self.n_ph + self.n_ph,
        }
        # 3. offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ext_ph,            
            "v": 1 + self.n_ext_ph + self.n_ext_ph,
            "i": 1 + self.n_ext_ph + self.n_ext_ph + self.n_ph,
            "lamda": 1 + self.n_ext_ph + self.n_ext_ph + self.n_ph + self.n_ph,
        }
        # fmt: on

        # to check if the total number of variables is correct with respect to the offset
        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["lamda"] + self.n_ph

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # y = [w, V, I, v, i, lamda]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        V_phasors = utils.get_vector_phasors(self.nominal_voltage)
        V_line_line = utils.get_phase_phase_values(V_phasors)
        V_line_line_abs = {k: np.abs(v) for k, v in V_line_line.items()}
        Vll = {k: v for k, v in V_line_line_abs.items() if k in self.P}
        Vll = np.array(list(Vll.values()))

        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = Vll

        idx_V_start = self.var_offset["V"]
        idx_V_end = idx_V_start + self.n_ext_ph
        y_0[idx_V_start:idx_V_end, 0] = np.array(list(V_phasors.values()))

        # initialize S
        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q

        i_phasors = (S / Vll).conjugate()
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        y_0[idx_i_start:idx_i_end, 0] = i_phasors

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        return y_0

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self) -> np.ndarray:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
        and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros
        """

        Id_ext_ph = sps.identity(self.n_ext_ph, format="coo")
        Id_ph = sps.identity(self.n_ph, format="coo")

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_ext_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)

        # create L and R matrices
        # get the nominal voltage from the object
        # get the S from active and reactive power from the load object
        # get the impedance Z = V^2/S*
        # TODO: all this should actually be a part of the adapter. At this stage we should only have L and R

        V_phasors = utils.get_vector_phasors(self.nominal_voltage)
        V_line_line = utils.get_phase_phase_values(V_phasors)
        V_line_line_abs = {k: np.abs(v) for k, v in V_line_line.items()}
        Vll = {k: v for k, v in V_line_line_abs.items() if k in self.P}
        Vll = np.array(list(Vll.values()))
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q

        # calculate the constant impedance
        Z = Vll**2 / S.conjugate()
        L = sps.diags(list(Z.imag)) / const.w_nominal
        R = sps.diags(list(Z.real))

        print(f">> L: {pformat(L.toarray())}")
        print(f">> R: {pformat(R.toarray())}")

        # create A matrix for this load model considering that there could be 1-ph, 2-ph or 3-ph load

        # for single phase load
        if self.n_ph == 1:
            if self.P.keys() == {"CA"}:
                A_T = np.array([[-1, 1]])
            else:
                A_T = np.array([[1, -1]])

        # for 2-ph load
        elif self.n_ph == 2:
            if self.P.keys() == {"AB", "BC"}:
                # fmt: off
                A_T = np.array([
                    [1, -1, 0],
                    [0, 1, -1]
                ])
                # fmt: on
            elif self.P.keys() == {"BC", "CA"}:
                # fmt: off
                A_T = np.array([
                    [0, 1, -1],
                    [-1, 0, 1]
                ])
                # fmt: on
            else:
                # fmt: off
                A_T = np.array([
                    [1, -1, 0],
                    [-1, 0, 1]
                ])
                # fmt: on

        # for 3-ph load
        else:
            # fmt: off
            A_T = np.array([
                [1, -1, 0],
                [0, 1, -1],
                [-1, 0, 1],
            ])       
            # fmt: on

        A = A_T.T

        # fmt: off
        M = sps.bmat(
            [ #w      V       I           v       i       lamda
            [ Z_w1,   None,  -Id_ext_ph,  None,   A,      None ],  # 1)-I + Ai = 0
            [ Z_w2,   A.T,   None,        -Id_ph, None,   None ],  # 2)A'V - v =0
            [ Z_w3,   None,  None,        None,   L,     -Id_ph], # 3)Li - lamda =0
            [ Z_w4,   None,  None,        -Id_ph, R,       None], # 4)-v + Ri + jw * lamda =0
            ]
        )
        # fmt: on

        return M

    def get_u_powerflow(self) -> sps.coo_array:
        u = sps.lil_matrix((self.num_eqns, 1), dtype=complex)
        return u.real, u[self.num_eqns_real :].imag

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Eqn 4 is non-linear because of the product of v and i
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """

        # assemble y:
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im
        # create a zero matrix for fy
        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 4
        # start stop index eqn 4
        idx_eq4_start = (
            self.n_ext_ph  # first set  :1)-I + Ai = 0
            + self.n_ph  # second set :2)A'V - v = 0
            + self.n_ph  # third set  :3)Li - lamda = 0
        )

        idx_eq4_end = idx_eq4_start + self.n_ph

        # start stop index of w and lamda in y vector
        idx_w = self.var_offset["w"]
        idx_lamda_start = self.var_offset["lamda"]
        idx_lamda_end = idx_lamda_start + self.n_ph

        fy[idx_eq4_start:idx_eq4_end, 0] = (
            1j * y[idx_w, 0] * y[idx_lamda_start:idx_lamda_end]
        )

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        # check that the variables received are of same shape as in the model
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        # get the index of w
        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # eq4_re i.e. the real part of the pd of eqn 4
        eq4_re_start_row = self.n_ext_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset

            # get the index of lamda in the y vector
            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = -lamda_im
            # lamda_re: derivative is zero

            # lamda_im
            pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

        # eq4_im
        eq4_im_start_row = self.num_eqns + self.n_ext_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_im_start_row + offset

            # get the index of lamda in the y vector
            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = lamda_re

            # lamda_re
            pd_fy_split[row, lamda_re_col_offset] = w
            # lamda_im : derivative is zero

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_gy(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        """
        1. This function is to be called from the Optimal Powerflow
        2. This function finds the Jacobian for the new non-linear vector formed by the multiplication of
            the lagrange multiplier and the transposed Jacobian of the non-linear vector of the powerflow eqns.
        3. g(y) = pd_f(y).transpose() * lagm.
        4. This function returns the partial derivative of g(y) with respect to y for this component to be stacked
            appropriately in the overall Jacobian.
        """
        # check that the variables received are of same shape as in the model

        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.lil_matrix(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        )

        idx_eq4 = (
            self.n_ext_ph  # first set :1) -I + Ai = 0
            + self.n_ph  # second set: 2) A'V - v =0
            + self.n_ph  # third set: 3) Li - lamda = 0
        )

        w_offset = self.var_offset["w"]
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.num_vars + self.var_offset_complex["lamda"]

        # eq4_re: (f(y) : -w * lamda_im)
        lagm = lagm_real[idx_eq4, 0]
        for offset in range(self.n_ph):
            row = w_offset
            col = lamda_im_start_offset + offset
            pd_gy_split[row, col] = -1 * lagm

        for offset in range(self.n_ph):
            row = lamda_re_start_offset + offset
            col = w_offset
            pd_gy_split[row, col] = -1 * lagm

        # eq4_im: (f(y) : w * lmada_re)
        lagm = lagm_imag[idx_eq4, 0]
        for offset in range(self.n_ph):
            row = w_offset
            col = lamda_re_start_offset + offset
            pd_gy_split[row, col] = 1 * lagm

        for offset in range(self.n_ph):
            row = lamda_re_start_offset + offset
            col = lamda_im_start_offset + offset
            pd_gy_split[row, col] = 1 * lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ir, ri, ii)

    ######################################dynamic simulation functions#################################
    """
     1) The dynamic equations only require a K matrix for the coefficients of the
           dynamic state variables.
        2) The M matrix remains the same.
        3) The non-linear vector changes
        4) The input vector also changes
        """

    def get_K_dynamic(self) -> sps.coo_array:
        """
        1) This function creates the K matrix for dynamic equations
        2) First we create the identity and coefficient matrices required for each eqn
            and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros
        """

        Id_ph = sps.identity(self.n_ph, format="coo")
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        Z_w = sps.lil_matrix((1, 1), dtype=float)

        K = sps.bmat(
            [
                # w       V     I        v       i       lamda
                [Z_w, None, None, None, Z_ph, None],  # 1)-I + i = 0
                [Z_w, Z_ph, None, None, None, None],  # 2)V - v =0
                [Z_w, None, Z_ph, None, None, None],  # 3)Li - lamda =0
                [Z_w, None, None, Z_ph, None, Id_ph],  # 4)-v + Ri + d(lamda)/dt =0
            ]
        )
        # fmt: on

        return K

    def get_fy_dynamic():
        pass

    def get_u_dynamic():
        # this values should be initialized by the powerflow
        pass


class DeltaConstantCurrentLModel(LoadModel):
    def __init__(self, load_obj: Load):
        super().__init__(load_obj)

        # create a constant matrix for power factor to be maintained as a constant for all iterations
        self.pf_const = np.array(list(self.obj.power_factor.values())).reshape(-1, 1)

        # book keeping of eqns
        self.n_ext_ph = 2 if self.n_ph == 1 else 3
        # fmt:off
        self.num_eqns_real = (self.n_ph  # first set :1) pf - pf_const = 0
                            + self.n_ph  # second set:2) iiconj - i_const^2 = 0
                            + self.n_ph  #)third set: 3) Sreal/|S| - pf = 0
                            + self.n_ph) # fourth set: 4) iiconj - ii* =  0
        self.num_eqns_complex = (
            self.n_ext_ph # first set :1)-I + Ai = 0
            + self.n_ph # second set: 2) A'V - v =0
            + self.n_ph  # third set  :3)-s + vi* =0           
        )
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex
        # fmt: on

        # book keeping of variables
        # fmt: off
        # y = [ w, pf, iiconj, V, I, v, i, S]
        self.num_vars_real = (1             # w 
                              + self.n_ph # pf
                              + self.n_ph # iiconj)
        )
        self.num_vars_complex = (
            self.n_ext_ph   # V
            + self.n_ext_ph # I           
            + self.n_ph # v
            + self.n_ph # i
            + self.n_ph # S            
        )
        self.num_vars = self.num_vars_real + self.num_vars_complex


        # fmt: on

        # create dictionaries to store the offset of each variable in the y vector
        # fmt: off
        # 1. offset for real variables
        self.var_offset_real = {"w": 0,
                                "pf": 1,
                                "iiconj": 1 + self.n_ph}
        # 2. pffset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ext_ph,
            "v": self.n_ext_ph + self.n_ext_ph,
            "i": self.n_ext_ph + self.n_ext_ph + self.n_ph,
            "S": self.n_ext_ph + self.n_ext_ph + self.n_ph + self.n_ph,
        }
        # 3. offset for all variables
        self.var_offset = {
            "w": 0,
            "pf": 1,
            "iiconj": 1 + self.n_ph,
            "V": 1 + self.n_ph + self.n_ph,
            "I": 1 + self.n_ph + self.n_ph + self.n_ext_ph,
            "v": 1 + self.n_ph + self.n_ph + self.n_ext_ph + self.n_ext_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ext_ph + self.n_ext_ph + self.n_ph,
            "S": 1 + self.n_ph + self.n_ph + self.n_ext_ph + self.n_ext_ph + self.n_ph + self.n_ph,
        }

        # to check if the total number of variables is correct with respect to the offset
        assert len(self.var_offset_real.keys()) + len(self.var_offset_complex.keys()) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["S"] + self.n_ph

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        if val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self) -> np.ndarray:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
           and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        Id_ext_ph = sps.identity(self.n_ext_ph, format="coo")
        Id_ph = sps.identity(self.n_ph, format="coo")

        Z_ext_ph = sps.lil_matrix((Id_ext_ph.shape[0], Id_ext_ph.shape[0]), dtype=float)
        Z_ph = sps.lil_matrix((Id_ph.shape[0], Id_ph.shape[0]), dtype=float)

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((Id_ext_ph.shape[0], 1), dtype=float)
        Z_w6 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w7 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)

        # print(f">>> self.P.keys(): {self.P.keys()}")
        # input("continue?")

        # # for single phase load
        # if self.n_ph == 1:
        #     if self.P.keys() == {"CA"}:
        #         A = np.array([[-1], [1]])
        #     else:
        #         A = np.array([[1], [-1]])

        # for single phase load
        if self.n_ph == 1:
            if self.P.keys() == {"CA"}:
                A = np.array([[-1, 1]])
            else:
                A = np.array([[1, -1]])

        # for 2-ph load
        elif self.n_ph == 2:
            if self.P.keys() == {"AB", "BC"}:
                # fmt: off
                A = np.array([
                    [1, -1, 0],
                    [0, 1, -1]
                ])
                # fmt: on
            elif self.P.keys() == {"BC", "CA"}:
                # fmt: off
                A = np.array([
                    [0, 1, -1],
                    [-1, 0, 1]
                ])
                # fmt: on
            else:
                # fmt: off
                A = np.array([
                    [1, -1, 0],
                    [-1, 0, 1]
                ])
                # fmt: on

        # for 3-ph load
        else:
            # fmt: off
            A = np.array([
                [1, -1, 0],
                [0, 1, -1],
                [-1, 0, 1]
            ])       
            # fmt: on

        A = A.T

        # note: I am confused whether the constant current and power factor should be passed from the adapter
        # or should be calculated here. I am assuming that it should be passed from the adapter

        # fmt: off

        # print(f">>> A: {pformat(A)}")
        # input("continue?")


        M = sps.bmat([
            #[w     pf      iiconj   V       I             v         i        S]
            [Z_w1,  Id_ph,  None,    None,   None,        None,     None,   None],  #1 pf - pf_const = 0 #1
            [Z_w2,  None,   Id_ph,   None,   None,        None,     None,   None],  #2 iiconj - iconst^2 = 0 #2
            [Z_w3, -Id_ph,  None,    None,   None,        None,     None,   None],  #3 Sreal/|S| - pf = 0 (Sreal/S non-linear) #6
            [Z_w4,  None,  Id_ph,    None,   None,        None,     None,   None ], #4 iiconj - ii* = 0 (ii* non-linear) #7
            [Z_w5,  None,  None,    None,   -Id_ext_ph,   None,     A,      None ], #5 -I + Ai = 0 #3
            [Z_w6,  None,  None,    A.T,      None,       -Id_ph,   None,   None ], #6 A'V - v =0 #4
            [Z_w7,  None,  None,    None,    None,         None,     None,  -Id_ph], #7 -S + vi* =0 (vi* non-linear) #5
        ])
        # fmt: on

        return M

    def get_u_powerflow(self) -> sps.coo_array:
        u = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        idx_eq1_start = 0
        idx_eq1_end = idx_eq1_start + self.n_ph
        u[idx_eq1_start:idx_eq1_end, 0] = -self.pf_const
        print(f">> pf_const: {self.pf_const}")

        idx_eq2_start = self.n_ph
        idx_eq2_end = idx_eq2_start + self.n_ph
        u[idx_eq2_start:idx_eq2_end, 0] = -(self.obj.iconst**2)
        print(f">> iconst**2: {self.obj.iconst**2}")

        return u.real, u[self.num_eqns_real :].imag

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Equation 4 and 5 have non-linearity in them
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this load model.
        """
        # re-assemble y
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im
        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # y = [w, pf, iiconj, V, I, i, v, S]

        # fy update for eqn #3
        # start stop index of S in y vector
        idx_S_start = self.var_offset["S"]
        idx_S_end = idx_S_start + self.n_ph
        S = y[idx_S_start:idx_S_end, 0]

        # start stop index of eqn #3
        idx_eq3_start = self.n_ph + self.n_ph
        idx_eq3_end = idx_eq3_start + self.n_ph
        fy[idx_eq3_start:idx_eq3_end, 0] = S.real / np.abs(S)

        # fy update for eqn 4 -----
        # start stop index of i in y vector
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start:idx_i_end, 0]

        # start stop index of eqn 4
        idx_eq4_start = (
            self.n_ph  # 1)pf - pf_const = 0 #1
            + self.n_ph  # 2)iiconj - iconst^2 = 0
            + self.n_ph  # 3) Sreal/|S| - pf = 0
        )
        idx_eq4_end = idx_eq4_start + self.n_ph

        fy[idx_eq4_start:idx_eq4_end, 0] = -(i.multiply(i.conjugate()))

        # fy update for eqn 7 -----
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        v = y[idx_v_start:idx_v_end, 0]

        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start:idx_i_end, 0]

        idx_eq7_start = (
            self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ext_ph + self.n_ph
        )
        idx_eq7_end = idx_eq7_start + self.n_ph

        fy[idx_eq7_start:idx_eq7_end, 0] = v.multiply(i.conjugate())

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        # check that the variables received are of same shape as in the model
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        # pd_fy update for eqn 3
        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq3_re real part of eqn
        eq3_re_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset

            # get the index of S in the y vector
            S_re_col_offset = self.var_offset["S"] + offset
            S_im_col_offset = self.var_offset_complex["S"] + offset
            S_re = y_real[S_re_col_offset, 0]
            S_im = y_imag[S_im_col_offset, 0]

            # S_re
            pd_fy_split[row, S_re_col_offset] = S_im**2 / (
                (S_re**2 + S_im**2) ** (3 / 2)
            )
            # S_im
            pd_fy_split[row, self.num_vars + S_im_col_offset] = (
                -S_re * S_im / ((S_re**2 + S_im**2) ** (3 / 2))
            )

        # eq3_im there is no imaginary part of eqn 3 so no updates here

        # eq4_re
        eq4_re_start_row = self.n_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset

            # get the index of i in the y vector
            i_re_col_offset = self.var_offset["i"] + offset
            i_im_col_offset = self.var_offset_complex["i"] + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # i_re
            pd_fy_split[row, i_re_col_offset] = -2 * i_re
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = -2 * i_im

        # eq4_im: there is no imaginary parto of eqn 4 so no updates here

        # pd_fy update ofr eqn #7
        # eq7_re
        eq7_re_start_row = (
            self.n_ph + self.n_ph + self.n_ph + self.n_ph + self.n_ext_ph + self.n_ph
        )
        for offset in range(self.n_ph):
            row = eq7_re_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = i_re
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_im
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_re
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = v_im

        # eq7_im
        eq7_im_start_row = (
            self.num_eqns + self.n_ext_ph + self.n_ph
        )  # num_eqns + num_complex eqns ONLY
        for offset in range(self.n_ph):
            row = eq7_im_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = -i_im
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_re
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_im
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = -v_re

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [w, pf, iiconj, V, I, i, v, S]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        V_phasors = utils.get_vector_phasors(self.nominal_voltage)
        V_line_line = utils.get_phase_phase_values(V_phasors)
        # V_line_line_abs = {k: np.abs(v) for k, v in V_line_line.items()}
        V_line_line_abs = {k: v for k, v in V_line_line.items()}
        Vll = {k: v for k, v in V_line_line_abs.items() if k in self.P}
        Vll = np.array(list(Vll.values()))

        idx_V_start = self.var_offset["V"]
        idx_V_end = idx_V_start + self.n_ext_ph
        y_0[idx_V_start:idx_V_end, 0] = np.array(list(V_phasors.values()))

        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = Vll

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        # intitialize S
        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q
        # print(f">> P: {P}")
        # print(f">> Q: {Q}")
        # print(f">> S: {S}")
        # input("continue?")

        idx_S_start = self.var_offset["S"]
        idx_S_end = idx_S_start + self.n_ph
        y_0[idx_S_start:idx_S_end, 0] = S
        # print(f">> y_0: {pformat(y_0.toarray())}")
        # print(f">> idx_S_start: {idx_S_start}")
        # print(f">> idx_S_end: {idx_S_end}")
        # input("continue?")

        # initialize i
        # i_init = [self.obj.iconst[0][0]] * self.n_ph
        i_init = (S / Vll).conjugate()
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        y_0[idx_i_start:idx_i_end, 0] = i_init
        print(f">> i_init: {i_init}")
        # input("continue?")

        # initialize pf
        idx_pf_start = self.var_offset["pf"]
        idx_pf_end = idx_pf_start + self.n_ph
        y_0[idx_pf_start:idx_pf_end, 0] = np.array(
            list(self.obj.power_factor.values())
        ).reshape(-1, 1)

        # iiconj
        idx_iiconj_start = self.var_offset["iiconj"]
        idx_iiconj_end = idx_iiconj_start + self.n_ph
        y_0[idx_iiconj_start:idx_iiconj_end, 0] = self.obj.iconst**2

        return y_0

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        # eq3, eq4, eq7

        # eq3_re: (orig: Sreal/sqrt(Sreal^2 + Simag^2))
        eq3_re_start_row = self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset
            lagm = lagm_real[row, 0]

            S_re_row = self.var_offset["S"] + offset
            S_re_col = S_re_row
            S_im_row = self.num_vars + self.var_offset_complex["S"] + offset
            S_im_col = S_im_row

            idx_S_re = self.var_offset["S"] + offset
            idx_S_im = self.var_offset_complex["S"] + offset
            S_re = y_real[idx_S_re, 0]
            S_im = y_imag[idx_S_im, 0]

            # S_re:
            val = -3 * S_re * S_im**2 / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_re_row, S_re_col] += val * lagm

            val = (2 * S_re**2 * S_im - S_im**3) / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_re_row, S_im_col] += val * lagm

            # S_im:
            val = (2 * S_re**2 * S_im - S_im**3) / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_im_row, S_re_col] += val * lagm

            val = (2 * S_im**2 * S_re - S_re**3) / (S_re**2 + S_im**2) ** (5 / 2)
            pd_gy_split[S_im_row, S_im_col] += val * lagm

        # eq3_im: NA

        # eq4_re: (orig: -ii*)
        eq4_re_start_row = self.n_ph + self.n_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset
            lagm = lagm_real[row, 0]

            i_re_row = self.var_offset["i"] + offset
            i_re_col = i_re_row
            i_im_row = self.num_vars + self.var_offset_complex["i"] + offset
            i_im_col = i_im_row

            # i_re:
            pd_gy_split[i_re_row, i_re_col] += -2 * lagm

            # i_im:
            pd_gy_split[i_im_row, i_im_col] += -2 * lagm

        # eq4_im: NA

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq7_re: (orig: v_re * i_re + v_im * i_im)
        eq7_re_start_row = 5 * self.n_ph + self.n_ext_ph
        for offset in range(self.n_ph):
            row = eq7_re_start_row + offset
            lagm = lagm_real[row, 0]

            # v_re:
            v_re_row = v_re_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[v_re_row, i_re_col] += 1 * lagm

            # i_re:
            i_re_row = i_re_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[i_re_row, v_re_col] += 1 * lagm

            # v_im:
            v_im_row = v_im_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[self.num_vars + v_im_row, self.num_vars + i_im_col] += 1 * lagm

            # i_im:
            i_im_row = i_im_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[self.num_vars + i_im_row, self.num_vars + v_im_col] += 1 * lagm

        # eq7_im: (orig: -v_re * i_im + v_im * i_re)
        eq7_im_start_row = self.n_ext_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq7_im_start_row + offset
            lagm = lagm_imag[row, 0]

            # v_re:
            v_re_row = v_re_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[v_re_row, self.num_vars + i_im_col] += -1 * lagm

            # i_im:
            i_im_row = i_im_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[self.num_vars + i_im_row, v_re_col] += -1 * lagm

            # v_im:
            v_im_row = v_im_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[self.num_vars + v_im_row, i_re_col] += 1 * lagm

            # i_re:
            i_re_col = i_re_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[i_re_col, self.num_vars + v_im_col] += 1 * lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)


class DeltaConstantPowerLModel(LoadModel):
    def __init__(self, load_obj: Load):
        super().__init__(load_obj)

        # fmt: off
        self.n_ext_ph = 2 if self.n_ph == 1 else 3

        # book keeping of eqns
        self.num_eqns_real = 0
        self.num_eqns_complex = (           
            self.n_ext_ph # first set  : 2)-I + i = 0
            + self.n_ph # second set   : 3)V -v =0
            + self.n_ph # third set    : 4)-S + vi* =0
            + self.n_ph # fourth set   : 5)-S + u[] =0
        )
        # fmt: on

        self.num_eqns = self.num_eqns_real + self.num_eqns_complex

        # book keeping of variables
        # fmt: off
        # y = [w, V, I, v, i, S]
        self.vars_real = ["w"]
        self.vars_complex = ["V", "I", "v", "i", "S"]

        self.num_vars_real = (1)
        self.num_vars_complex = (
            self.n_ext_ph #V
            + self.n_ext_ph #I
            + self.n_ph #v
            + self.n_ph #i
            + self.n_ph #S
        )

        self.num_vars = self.num_vars_real + self.num_vars_complex
        # fmt: on

        # create dictionaries to store the offset of each variable in the y vector
        # 1. offset for real variables
        self.var_offset_real = {"w": 0}
        # 2. offset for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ext_ph,
            "v": self.n_ext_ph + self.n_ext_ph,
            "i": self.n_ext_ph + self.n_ext_ph + self.n_ph,
            "S": self.n_ext_ph + self.n_ext_ph + self.n_ph + self.n_ph,
        }
        # 3. offset for all variables
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ext_ph,
            "v": 1 + self.n_ext_ph + self.n_ext_ph,
            "i": 1 + self.n_ext_ph + self.n_ext_ph + self.n_ph,
            "S": 1 + self.n_ext_ph + self.n_ext_ph + self.n_ph + self.n_ph,
        }

        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["S"] + self.n_ph

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [V, I, w, v, i, S]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        # initialize v
        V_phasors = utils.get_vector_phasors(self.nominal_voltage)
        V_line_line = utils.get_phase_phase_values(V_phasors)
        V_line_line_abs = {k: np.abs(v) for k, v in V_line_line.items()}
        Vll = {k: v for k, v in V_line_line_abs.items() if k in self.P}
        Vll = np.array(list(Vll.values()))
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        print(f">> V_phasors: {V_phasors}")
        print(f">> V_line_line: {V_line_line}")
        print(f">> Vll: {Vll}")
        y_0[idx_v_start:idx_v_end, 0] = Vll

        idx_V_start = self.var_offset["V"]
        idx_V_end = idx_V_start + self.n_ext_ph
        y_0[idx_V_start:idx_V_end, 0] = np.array(list(V_phasors.values()))

        # initialize S
        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q
        idx_S_start = self.var_offset["S"]
        idx_S_end = idx_S_start + self.n_ph
        y_0[idx_S_start:idx_S_end, 0] = S

        idx_w = self.var_offset["w"]
        y_0[idx_w, 0] = vals["w"]

        return y_0

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys()

        if var == "w":
            assert ph is None
            assert side is None

        side_offset = 0  # stc
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self) -> np.ndarray:
        """
        1)This function creates the M matrix for powerflow
        2)First we create the identity and coefficient matrices required for each eqn
           and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        Id_ext_ph = sps.identity(self.n_ext_ph, format="coo")
        Id_ph = sps.identity(self.n_ph, format="coo")

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_ext_ph.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_ph.shape[0], 1), dtype=float)

        # for single phase load
        if self.n_ph == 1:
            if self.P.keys() == {"CA"}:
                A_T = np.array([[-1, 1]])
            else:
                A_T = np.array([[1, -1]])

        # for 2-ph load
        elif self.n_ph == 2:
            if self.P.keys() == {"AB", "BC"}:
                # fmt: off
                A_T = np.array([
                    [1, -1, 0],
                    [0, 1, -1]
                ])
                # fmt: on
            elif self.P.keys() == {"BC", "CA"}:
                # fmt: off
                A_T = np.array([
                    [0, 1, -1],
                    [-1, 0, 1]
                ])
                # fmt: on
            else:
                # fmt: off
                A_T = np.array([
                    [1, -1, 0],
                    [-1, 0, 1]
                ])
                # fmt: on

        # for 3-ph load
        else:
            # fmt: off
            A_T = np.array([
                [1, -1, 0],
                [0, 1, -1],
                [-1, 0, 1],
            ])       
            # fmt: on

        A = A_T.T

        print(f">>> A: {pformat(A)}")

        # fmt: off
        M = sps.bmat(
            [#w       V      I             v       i       S                
             [Z_w1,   None,  -Id_ext_ph,   None,   A,      None ],  # 1)-I + Ai = 0
             [Z_w2,   A.T,   None,         -Id_ph, None,   None ],  # 2)A.T V -v =0
             [Z_w3,   None,  None,         None,   None,  -Id_ph],  # 3)-S + vi* =0
             [Z_w4,   None,  None,         None,   None,  -Id_ph],  # 4)-S + u[] =0
            ]
        )
        # fmt: on

        return M

    def get_u_powerflow(self) -> np.ndarray:
        """
        Load model u has constant power S consumed by the load
        """

        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype="complex")

        # get the active and reactive power from the load object
        P = np.array(list(self.P.values()))
        Q = np.array(list(self.Q.values()))
        S = P + 1j * Q

        # update u for eqn 4
        # start stop index of eqn 5 in u vector
        # fmt: off
        idx_eq4_re_start = (
            self.n_ext_ph # 1)-I + i = 0
            + self.n_ph # 2)V - v =0
            + self.n_ph # 3) -S + vi* =0
        )
        # fmt: on
        idx_eq4_re_end = idx_eq4_re_start + self.n_ph
        u[idx_eq4_re_start:idx_eq4_re_end, 0] = S.real
        print(f">> S.real: {S.real}")

        idx_eq4_im_start = (
            self.num_eqns
            + self.n_ext_ph  # 1)-I + i = 0
            + self.n_ph  # 2)V - v =0
            + self.n_ph  # 3)-S + vi* =0
        )
        idx_eq4_im_end = idx_eq4_im_start + self.n_ph

        print(f">> idx_eq4_re_start: {idx_eq4_re_start}")
        print(f">> idx_eq4_re_end: {idx_eq4_re_end}")
        print(f">> idx_eq4_im_start: {idx_eq4_im_start}")
        print(f">> idx_eq4_im_end: {idx_eq4_im_end}")

        print(f">> u.shape: {u.shape}")
        print(
            f">>> u[idx_eq4_re_start:idx_eq4_re_end]: {u[idx_eq4_re_start:idx_eq4_re_end].shape}"
        )
        print(f">>> S.real: {S.real.shape}")
        print(
            f">>> u[idx_eq4_im_start:idx_eq4_im_end]: {u[idx_eq4_im_start:idx_eq4_im_end].shape}"
        )
        print(f">>> S.imag: {S.imag.shape}")
        # print n_ph
        print(f">>> n_ph: {self.n_ph}")
        u[idx_eq4_im_start:idx_eq4_im_end, 0] = S.imag

        # print(f">> S.imag: {S.imag}")
        # input("continue?")

        return u[: self.num_eqns], u[self.num_eqns :]

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Eqn 4 is non-linear because of the product of v and i
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        # assemble y
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # fy update for eqn 4
        # start stop index of v in y vector
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        v = y[idx_v_start:idx_v_end, 0]
        # print(f"type(v): {type(v)}")
        # print(f"v.shape: {v.shape}")

        # start stop index of i in y vector
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        i = y[idx_i_start:idx_i_end, 0]
        # print(f"type(i): {type(i)}")

        # start stop index of eqn 4
        # fmt: off
        idx_eq3_start = (
            # + 1         # first set  1)eqn for w
            self.n_ext_ph # second set 2)-I + i = 0
            + self.n_ph # third set  3)V - v =0
        )
        # fmt: on
        idx_eq3_end = idx_eq3_start + self.n_ph

        fy[idx_eq3_start:idx_eq3_end, 0] = v.multiply(i.conjugate())
        # input("continue?")

        return fy.real, fy[self.num_eqns_real :].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq3_re
        eq3_re_start_row = self.n_ext_ph + self.n_ph
        for offset in range(self.n_ph):
            row = eq3_re_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = i_re
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_im
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_re
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = v_im

        # eq3_im
        eq3_im_start_row = self.num_eqns + eq3_re_start_row
        for offset in range(self.n_ph):
            row = eq3_im_start_row + offset

            v_re_col_offset = v_re_start_offset + offset
            v_im_col_offset = v_im_start_offset + offset
            v_re = y_real[v_re_col_offset, 0]
            v_im = y_imag[v_im_col_offset, 0]

            i_re_col_offset = i_re_start_offset + offset
            i_im_col_offset = i_im_start_offset + offset
            i_re = y_real[i_re_col_offset, 0]
            i_im = y_imag[i_im_col_offset, 0]

            # v_re
            pd_fy_split[row, v_re_col_offset] = -i_im
            # v_im
            pd_fy_split[row, self.num_vars + v_im_col_offset] = i_re
            # i_re
            pd_fy_split[row, i_re_col_offset] = v_im
            # i_im
            pd_fy_split[row, self.num_vars + i_im_col_offset] = -v_re

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_pd_gy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        lagm_real: sps.coo_array,
        lagm_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.coo_array(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            )
        ).tocsc()

        eq3_idx = self.n_ext_ph + self.n_ph

        v_re_start_offset = self.var_offset["v"]
        v_im_start_offset = self.var_offset_complex["v"]
        i_re_start_offset = self.var_offset["i"]
        i_im_start_offset = self.var_offset_complex["i"]

        # eq3_re : (f(y) = v_re * i_re + v_im * i_im)
        for offset in range(self.n_ph):
            eq_row = eq3_idx + offset
            lagm = lagm_real[eq_row, 0]
            v_re_row = v_re_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[v_re_row, i_re_col] += 1 * lagm

            i_re_row = i_re_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[i_re_row, v_re_col] += 1 * lagm

            v_im_row = v_im_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[v_im_row, i_im_col] += 1 * lagm

            i_im_row = i_im_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[i_im_row, v_im_col] += 1 * lagm

        # eq3_im : (f(y) = -v_re * i_im + v_im * i_re)
        for offset in range(self.n_ph):
            eq_row = eq3_idx + offset
            lagm = lagm_imag[eq_row, 0]
            v_re_row = v_re_start_offset + offset
            i_im_col = i_im_start_offset + offset
            pd_gy_split[v_re_row, i_im_col] += -1 * lagm

            i_im_row = i_im_start_offset + offset
            v_re_col = v_re_start_offset + offset
            pd_gy_split[i_im_row, v_re_col] += -1 * lagm

            v_im_row = v_im_start_offset + offset
            i_re_col = i_re_start_offset + offset
            pd_gy_split[v_im_row, i_re_col] += 1 * lagm

            i_re_row = i_re_start_offset + offset
            v_im_col = v_im_start_offset + offset
            pd_gy_split[i_re_row, v_im_col] += 1 * lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)
