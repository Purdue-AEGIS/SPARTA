from oodesign import ShuntCapacitor
import numpy as np
import scipy.sparse as sps
import const
from const import NodeSide, StudyType
from models.model import Model
from models.model import Model, ValType
import utils
from utils import phasor_to_timedomain
from pprint import pformat
from models.component_models.equipment_model import EquipmentModel


"""
1)This file contains the class for capacitor  model.
2)The capacitor model is a subclass of the Model class.
3)The base capacitor model class shall have two subclasses based on the type of connection:Delta and Star.
4)The capacitor model class shall have the following methods:
    a)get_M_powerflow
    b)get_u_powerflow
    c)get_fy_powerflow
"""


class ShuntCapacitorModel(EquipmentModel):
    def __init__(self, cap_obj: ShuntCapacitor):
        self.obj = cap_obj
        self.n_ph = self.obj.n_ph
        self.num_term = 1
        self.nominal_voltage = self.obj.nominal_voltage

        self.C = {}
        for ph in self.get_phases():
            if ph != "N":
                # self.C[ph] = self.obj.power[ph] / (self.nominal_voltage[ph] ** 2) / 1000 / (2 * np.pi * 60) 
                self.C[ph] = self.obj.power[ph] / (self.nominal_voltage[ph] ** 2) / const.w_nominal

        print(f">> self.nomnial_voltage: {pformat(self.nominal_voltage)}")
        print(f">> self.obj.power: {pformat(self.obj.power)}")
        print(f">> self.C: {pformat(self.C)}")

    def get_basetype(self) -> str:
        return "capacitor"

    # Note: This matrix should come from adapter file


class StarShuntCapacitorModel(ShuntCapacitorModel):
    def __init__(self, cap_obj: ShuntCapacitor):
        super().__init__(cap_obj)

        # book keeping of the equations
        # fmt: off
        self.num_eqns_real = 0
        self.num_eqns_complex = (self.n_ph    # first set :1)-I +i = 0
                        + self.n_ph   # second set:2)-V +v = 0
                        + self.n_ph   # third set :3)Cv - q = 0
                        + self.n_ph   # fourth set:4)-i + jwq = 0
                        )   

        # fmt: on
        self.num_eqns = self.num_eqns_complex

        # book keeping of the variables
        # calculate the number of variables
        # y = [V, I, w, v, i, q]
        # V = [Va, Vb, Vc]
        # I = [Ia, Ib, Ic]
        # v = [va_cap, vb_cap, vc_cap]
        # i = [ia_cap, ib_cap, ic_cap]
        # q = [qa, qb, qc]

        # fmt:off
        self.num_vars_real = (1) # w
        self.num_vars_complex =(self.n_ph  # V
                        + self.n_ph  # I                        
                        + self.n_ph  # v
                        + self.n_ph  # i
                        + self.n_ph  # q
                        )

        # fmt:on
        self.num_vars = self.num_vars_real + self.num_vars_complex

        # Create dictionaries to store the offset of each variable in the y vector
        # fmt:off
        # 1. offsets for real variables
        self.var_offset_real = {"w": 0}
        # 2. offsets for complex variables
        self.var_offset_complex = {
            "V": 0,
            "I": self.n_ph,            
            "v": self.n_ph + self.n_ph,
            "i": self.n_ph + self.n_ph + self.n_ph,
            "q": self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }
        # 3. offsets for all variables 
        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + self.n_ph,            
            "v": 1 + self.n_ph + self.n_ph,
            "i": 1 + self.n_ph + self.n_ph + self.n_ph,
            "q": 1 + self.n_ph + self.n_ph + self.n_ph + self.n_ph,
        }

        # fmt:on
        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())

        assert self.num_vars == self.var_offset["q"] + self.n_ph

    # dynamic:
        # This model has same no. of variables and eqns for dynamic and powerflow study
        self.num_eqns_dynamic = self.num_eqns
        self.num_vars_dynamic = self.num_vars
        self.var_offset_dynamic = self.var_offset.copy()

       

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # [V, I, w, v, i, q]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        # initialize v
        v_phasors_dict = utils.get_vector_phasors(self.nominal_voltage)
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

        # initialize S
        # get the active and reactive power from the load object
        # P = 1e3 * np.array(list(self.P.values()))
        # Q = 1e3 * np.array(list(self.obj.power.values()))
        # S = P + 1j * Q
        # idx_S_start = self.var_offset["S"]
        # idx_S_end = idx_S_start + self.n_ph
        # guess[idx_S_start:idx_S_end, 0] = S

        # calculate iconst
        # nominal_v_phasors = np.array(
        #     list(utils.get_vector_phasors(load_dict["nominal_voltage"]).values())
        # )
        # P = np.array(list(load_dict["active_power"].values()))
        Q = np.array(list(self.obj.power.values())).reshape(-1, 1)
        S = 1j * Q
        print(f">> S: {S}")
        iconst = (S / v_phasors).conj()
        iconst = iconst.reshape(-1, 1)
        # iconst = np.abs(iconst)
        idx_i_start = self.var_offset["i"]
        idx_i_end = idx_i_start + self.n_ph
        y_0[idx_i_start:idx_i_end, 0] = iconst
        # input("continue?")

        q = -1j * iconst / const.w_nominal
        idx_q_start = self.var_offset["q"]
        idx_q_end = idx_q_start + self.n_ph
        y_0[idx_q_start:idx_q_end, 0] = q

        # C = {}
        # for ph in self.get_phases():
        #     if ph != "N":
        #         C[ph] = self.obj.power[ph] / (self.nominal_voltage[ph] ** 2) / 1000 / (2 * np.pi * 60)
        
        # C_lst = np.array(list(C.values())).reshape(-1, 1)
        # print(f">> C_lst: {C_lst}")
        # print(f">> v_phasors: {v_phasors}")
        # idx_q_start = self.var_offset["q"]
        # idx_q_end = idx_q_start + self.n_ph
        # y_0[idx_q_start:idx_q_end, 0] = C_lst * v_phasors

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

        assert var not in ["v", "i", "q"]  # to be implemented later
        # assert side == NodeSide.AT

        side_offset = 0
        if side == NodeSide.TO:
            side_offset = self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_M_powerflow_inner(self) -> sps.coo_array:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
            and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros
        """

        # Identity matrices
        Id_ph = sps.identity(self.n_ph, format="coo")  # A matrix for star capacitors

        # Zero matrices
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)

        # zero vectors for w for each eqn
        Zw = sps.lil_matrix((self.n_ph, 1), dtype=float)

        C = sps.diags(list(self.C.values()))

        # fmt:off
        M = sps.bmat([
           #[w,    V,        I,         v,          i,       q]
            [Zw,   None,     -Id_ph,    None,       Id_ph,   None],   # 1) -I + Ai = 0
            [Zw,   -Id_ph,   None,      Id_ph,      None,    None],   # 2) -A'V + v = 0 
            [Zw,   None,     None,      C,          None,    -Id_ph], # 3) Cv - q = 0
            [Zw,   None,     None,      None,      -Id_ph,   None]     # 4) -i + jwq = 0
        ], format="coo")
        # fmt:on

        return M

    def get_u_powerflow(self) -> sps.coo_array:
        """
        Capacitor model does not have any inputs therefore u is zeros of size of no. of eqns
        """
        u = sps.lil_matrix((self.num_eqns, 1), dtype=complex)
        return u.real, u[self.num_eqns_real :].imag

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Since frequency is a variable we have non-linear terms in eqn 4 shown in the matrix above.
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        # assemble y vector
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im
        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # y = [V, I, w, v, i, q]

        # index of w in y vector
        idx_w = self.var_offset["w"]
        w = y[idx_w, 0]

        # fy update for eqn 4
        # start and end index of q in y vector
        idx_q_start = self.var_offset["q"]
        idx_q_end = idx_q_start + self.n_ph

        # identify the start and end index of eqn 4 in fy vector
        # fmt:off
        idx_eq4_start = (self.n_ph    # first set :1)-I +i = 0
                        + self.n_ph   # second set:2)-V +v = 0
                        + self.n_ph )  # third set :3)Cv - q = 0)
        
        idx_eq4_end = idx_eq4_start + self.n_ph

        fy[idx_eq4_start:idx_eq4_end] = 0 + 1j * (w * y[idx_q_start:idx_q_end] )

        return fy.real, fy[self.num_eqns_real:].imag

    def get_pd_fy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array]:
        """
        1. This function returns the partial derivative of f(y) vector which contains the non-linear
            terms of all the equations
        2. It takes as an input the y vector as y_real and y_imag and the self object of the class
        3. It returns the derivative as following four blocks:
            a) derivative of fy real with respect to y_real (rr)
            b) derivative of fy real with respect to y_imag (ri)
            c) derivative of fy imag with respect to y_real (ir)
            d) derivative of fy imag with respect to y_imag (ii)
        """
        # print(f">> in get_pd_fy_split of capacitor model")
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]
        print(f">> w: {w}")

        q_re_start_offset = self.var_offset["q"]
        q_im_start_offset = self.var_offset_complex["q"]

        # the partial derivative for eqn 4

        # eq4_re derivative of real part of the non-linear term in eq4
        # fmt:off
        eq4_re_start_row = (self.n_ph    # first set :1)-I +i = 0
                        + self.n_ph   # second set:2)-V +v = 0
                        + self.n_ph )  # third set :3)Cv - q = 0))

        for offset in range(self.n_ph):
            row = eq4_re_start_row + offset

            q_re_col_offset = q_re_start_offset + offset
            q_im_col_offset = q_im_start_offset + offset
            q_re = y_real[q_re_col_offset, 0]
            q_im = y_imag[q_im_col_offset, 0]
            # print(">>> q_re: ", q_re)
            # print(">>> q_im: ", q_im)

            # write the derivative 
            # wrt w
            pd_fy_split[row, w_col_offset] = -q_im
            
            # wrt q_re : derivative is zero so no updates here
            
            # wrt q_im
            pd_fy_split[row, self.num_vars + q_im_col_offset] = -w

        # eq4_im derivative of imag part of the non-linear term in eq4
        eq4_im_start_row = self.num_eqns + eq4_re_start_row
        for offset in range(self.n_ph):
            row = eq4_im_start_row + offset

            q_re_col_offset = q_re_start_offset + offset
            q_im_col_offset = q_im_start_offset + offset
            q_re = y_real[q_re_col_offset, 0]
            q_im = y_imag[q_im_col_offset, 0]
            # print(">>> q_re: ", q_re)
            # print(">>> q_im: ", q_im)

            # write the derivative

            # wrt w
            pd_fy_split[row, w_col_offset] = q_re            
            # wrt q_re
            pd_fy_split[row, q_re_col_offset] = w
            # wrt q_im : derivative is zero so no updates here
            

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)
    
    def get_pd_gy_split(self,
                        y_real: sps.coo_array,
                        y_imag: sps.coo_array,
                        lagm_real: sps.coo_array,
                        lagm_imag: sps.coo_array
                        ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        """
        1. This function is to be called from the Optimal Powerflow
        2. This function finds the Jacobian for the new non-linear vector formed by the multiplication of
            the lagrange multiplier and the transposed Jacobian of the non-linear vector of the powerflow eqns.
        3. g(y) = pd_f(y).transpose() * lagm.
        4. This function returns the partial derivative of g(y) with respect to y for this component to be stacked
            appropriately in the overall Jacobian.
        """
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]
        assert self.num_eqns == lagm_real.shape[0]
        assert self.num_eqns_complex == lagm_imag.shape[0]

        pd_gy_split = sps.lil_matrix(
            (
                self.num_vars + self.num_vars_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype = float,
        ).tocsc()

        # eqn_4
        idx_eq4 = (self.n_ph    # first set :1)-I +i = 0
                        + self.n_ph   # second set:2)-V +v = 0
                        + self.n_ph   # third set :3)Cv - q = 0
        )

        w_offset = self.var_offset["w"]
        q_re_start_offset = self.var_offset["q"]
        q_im_start_offset = self.var_offset_complex["q"]

        # eq4_re: (fy_real: -w * q_im)
        for offset in range(self.n_ph):
            eq_row = idx_eq4 + offset
            lagm = lagm_real[eq_row,0]

            w_row = w_offset
            q_im_col = q_im_start_offset + offset
            pd_gy_split[w_row, q_im_col] = -1*lagm

            q_im_row = q_im_start_offset + offset
            w_col = w_offset
            pd_gy_split[q_im_row, w_col] = -1*lagm

        # eq4_im: (fy_imag: w * q_re)
        for offset in range(self.n_ph):
            eq_row = idx_eq4 + offset
            lagm = lagm_imag[eq_row, 0]

            w_row = w_offset
            q_re_col = q_re_start_offset + offset
            pd_gy_split[w_row, q_re_col] = 1*lagm

            q_re_row = q_re_start_offset + offset
            w_col = w_offset
            pd_gy_split[q_re_row, w_col] = 1*lagm

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

########################Dynamic simulation functions########################
# These functions are called by dynamic.py

    def get_local_idx_dynamic(
            self, var: str, ph: str |None, side: NodeSide | None
    ) -> int:
        
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in ["v", "i", "q"]  # to be implemented later
        side_offset = 0
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset
    
    def initial_guess_dynamic_zero(self, y_comp, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

       
        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)

        return y0_dyn
    
    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ):
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn
    
    def initial_guess_dynamic(self, y_comp) -> np.ndarray:
        assert len(y_comp) == self.num_vars # this y vector results from the powerflow solution

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype = complex)

        # initialization from powerflow directly
        vars_counts_real = [
            ("w", 1)
        ]

        for var, count in vars_counts_real:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start:idx_var_dyn_end] = y_comp[idx_var_pf_start:idx_var_pf_end]

        vars_counts_complex = [
            ("V", self.n_ph),
            ("I", self.n_ph),
            ("v", self.n_ph),
            ("i", self.n_ph),
            ("q", self.n_ph)
        ]

        # need conversion from phasor to time domain
        for var, count in vars_counts_complex:
            idx_var_pf_start = self.var_offset[var]
            idx_var_pf_end = idx_var_pf_start + count
            idx_var_dyn_start = self.var_offset_dynamic[var]
            idx_var_dyn_end = idx_var_dyn_start + count

            y0_dyn[idx_var_dyn_start:idx_var_dyn_end] = np.sqrt(2) * phasor_to_timedomain(y_comp[idx_var_pf_start:idx_var_pf_end])

        return y0_dyn
    
    def get_M_dynamic(self) -> sps.coo_array:
        """
        1) For this model the M matrix is same as the powerflow M matrix
        """

        M = self.get_M_powerflow_inner()

        return M
    
    def get_K_dynamic(self) -> sps.coo_array:

        Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)  # zero vector for w
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)  # zero matrix for ph
        Id_ph = sps.identity(self.n_ph, format="coo")  # identity matrix for ph

        # fmt:off
        K = sps.bmat([
           #[w,     V,        I,         v,      i,       q]
            [Z_w,   None,     Z_ph,    None,    Z_ph,    None],   # 1) -I + Ai = 0
            [Z_w,   Z_ph,     None,    Z_ph,    None,    None],   # 2) -A'V + v = 0 
            [Z_w,   None,     None,    None,    None,    None], # 3) Cv - q = 0
            [Z_w,   None,     None,    None,    None,   Id_ph]     # 4) -i + jwq = 0
        ], format="coo")
        # fmt:on

        return K
    
    
    def get_fy_dynamic(self, t : float, y : np.ndarray) -> np.ndarray:

        """
        1)For this model there are no non-linear terms in the dynamic model
        """

        fy = np.zeros(self.num_eqns_dynamic, dtype = complex)

        return fy
    
    def get_u_dynamic(self, t, y: np.ndarray) -> np.ndarray:
        """
        1)For this model there are no inputs in the dynamic model
        """
        u = np.zeros(self.num_eqns_dynamic, dtype = complex)

        return u
    








class DeltaShuntCapacitorModel(ShuntCapacitorModel):
    pass
