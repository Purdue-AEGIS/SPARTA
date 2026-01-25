import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from oodesign import Switch, SwitchState
from const import NodeSide
import utils
from models.component_models.equipment_model import EquipmentModel

#  Import needed Assimulo libraries
from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem
from models.model import Model, ValType
from pprint import pformat

# #Simulation time
# t_sim = 0.2 #seconds
# # t_sim = 1 #seconds
# # t_sim = 4000 #seconds
# # t_sim = 40 #seconds

# #fund. freq.
# we = 2*np.pi*60
# w_b = we

# #Base values
# Vdc = 1500
# # Pb  = 1e6                #W
# Pb  = 100e6                #W
# # V_b = Vdc/np.sqrt(2) #V (based on SVPWM) rms value (l-l)
# V_b = Vdc/np.sqrt(3) #V (based on SVPWM) rms value (l-l)
# V_bqd0     =  np.sqrt(2/3)*V_b # base voltage for qd0 variables (V, line-to-neutral, peak)
# I_b        =  Pb/(V_b*np.sqrt(3)) # base phase current (A, rms)
# I_bqd0     =  np.sqrt(2)*I_b # base current for qd0 variables (A, peak)
# Z_b        =  V_b**2/Pb # base impedance (ohm)

# V_s      = V_b/np.sqrt(3) # line-to-neutral voltage (V, rms)
# I_s      = Pb/3/V_s # phase current (A, rms)

# Imx = 600/I_bqd0

# n_varqd = 2
# n_ph = 3




# ------------------


class SwitchModel(EquipmentModel):
    def __init__(self, switch_obj: Switch):
        super().__init__(switch_obj)

        self.num_term = 2

        self.nominal_voltage = self.obj.nominal_voltage

        self.num_eqns_real = 0
        self.num_eqns_complex = self.n_ph + self.n_ph
        self.num_eqns = self.num_eqns_real + self.num_eqns_complex
        self.num_eqns_dynamic = self.num_eqns

        self.var_offset_real = {}

        self.var_offset_complex = {
            "V": 0,
            "I": 2 * self.n_ph,
        }

        self.var_offset = {
            "V": 0,
            "I": 2 * self.n_ph,
        }

        self.var_offset_dynamic = self.var_offset

        self.num_vars_real = 0
        self.num_vars_complex = 4 * self.n_ph
        self.num_vars = self.num_vars_real + self.num_vars_complex
        self.num_vars_dynamic = self.num_vars

        assert len(self.var_offset_real.keys()) + len(
            self.var_offset_complex.keys()
        ) == len(self.var_offset.keys())
        assert self.num_vars == self.var_offset["I"] + 2 * self.n_ph

    def get_basetype(self):
        return "switch"
    
    def get_id(self):
        return self.obj.id

    #####################
    ### powerflow related

    def initial_guess(self, vals: dict) -> sps.coo_array:
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        idx_V1_start = self.var_offset["V"]
        idx_V1_end = self.var_offset["V"] + self.n_ph
        idx_V2_start = idx_V1_end
        idx_V2_end = idx_V2_start + self.n_ph
        V_phasors = utils.get_vector_phasors(self.obj.nominal_voltage)
        V_phasors = np.array(list(V_phasors.values()))
        y_0[idx_V1_start:idx_V1_end, 0] = V_phasors
        y_0[idx_V2_start:idx_V2_end, 0] = V_phasors

        return y_0

    def get_M_powerflow_inner(self, stage=None) -> sps.coo_array:
        print(f"swtich_model.get_M_powerflow_inner called with stage : {stage}")
        input("continue?")
        Id_ph = sps.identity(self.n_ph, format="coo")
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        I1 = sps.bmat([[Id_ph, Z_ph]])
        I2 = sps.bmat([[Z_ph, Id_ph]])
        Z2nph = sps.bmat([[Z_ph, Z_ph]])
        EQ = sps.bmat([[Id_ph, -Id_ph]])

        # fmt: off
        self.sw_open_Mmat = sps.bmat([
            # V,    I
            [Z2nph, I1 ],       # I1 = 0
            [Z2nph, I2 ],       # I2 = 0
        ])

        self.sw_close_Mmat = sps.bmat([
            # V,    I
            [EQ,    None ],     # V1 - V2 = 0
            [None,  EQ   ],     # I1 - I2 = 0
        ])
        # fmt: on

        if stage is None or stage == "stage2":
            # closed
            return self.sw_close_Mmat
        
        elif stage == "stage1":
            # open
            return self.sw_open_Mmat
        
        else:
            raise ValueError("illegal switch stage")

        # if self.obj.state == SwitchState.Open:
        #     return self.sw_open_Mmat

        # elif self.obj.state == SwitchState.Closed:
        #     return self.sw_close_Mmat

        # else:
        #     raise ValueError("illegal switch state")

    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)
        return fy.real, fy[self.num_eqns_real :].imag

    def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        """
        Line model u is zeros of size of no. of eqns
        """
        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=float)
        return u[: self.num_eqns], u[self.num_eqns :]

    def get_pd_fy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_fy_split = sps.coo_array(
            # (2 * self.num_eqns, 2 * self.num_vars), dtype=float
            (
                self.num_eqns + self.num_eqns_complex,
                self.num_vars + self.num_vars_complex,
            ),
            dtype=float,
        ).tocsc()

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_local_idx(
        self, var: str, val_type: ValType, ph: str | None, side: NodeSide | None
    ) -> int:
        assert var in self.var_offset.keys() or var == "V"

        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset = 0
        if side == NodeSide.TO:
            side_offset = self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    #####################
    ### dynamic related

    def initial_guess_dynamic(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        raise NotImplementedError

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        y0_dyn[0] = y_comp[0].real
        # rest:
        y0_dyn[1:] = [np.sqrt(2) * phasor_to_timedomain(val) for val in y_comp[1:]]

        return y0_dyn
    
    def initial_guess_dynamic_zero(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return y0_dyn
    
    def initial_yp_dynamic_zero(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        return yp0_dyn

    def initial_yp_dynamic(
        self, y0_dyn_comp: np.ndarray, y0_pf_comp: np.ndarray, wnom
    ) -> np.ndarray:
        assert len(y0_pf_comp) == self.num_vars
        assert len(y0_dyn_comp) == self.num_vars_dynamic

        yp0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        raise NotImplementedError

        idx_v_start = self.var_offset_dynamic["v"] + 2 * self.n_ph
        idx_v_end = idx_v_start + self.n_ph
        v = y0_dyn_comp[idx_v_start:idx_v_end]

        idx_i_start = self.var_offset_dynamic["i"] + 2 * self.n_ph
        idx_i_end = idx_i_start + self.n_ph
        i = y0_dyn_comp[idx_i_start:idx_i_end]

        idx_p_lamda_start = self.var_offset_dynamic["lamda"]
        idx_p_lamda_end = idx_p_lamda_start + self.n_ph

        yp0_dyn[idx_p_lamda_start:idx_p_lamda_end] = v - i @ self.R

        return yp0_dyn
    
    def get_local_idx_dynamic(self, var: str, ph: str | None, side: NodeSide | None) -> int:
        assert var in self.var_offset_dynamic.keys(), f"var = {var}"

        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset = 0
        if side == NodeSide.TO:
            side_offset = self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset
    
    # def get_fy_dynamic
    def get_fy_dynamic(self, t, y, stage=None) -> sps.coo_array:
        """
        The fy vector for the line is empty
        """
        fy = np.zeros(self.num_eqns, dtype=float)
        # fy = sps.lil_matrix((self.num_eqns, 1), dtype=float)
        return fy

    # def get_u_dynamic
    def get_u_dynamic(self, t: float, y, stage=None) -> np.ndarray:
        u = np.zeros(self.num_eqns, dtype=float)
        return u
    
    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        return self.get_M_powerflow_inner(stage)

    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        K_dynamic = sps.lil_matrix((self.num_eqns_dynamic, self.num_vars_dynamic))
        return K_dynamic
