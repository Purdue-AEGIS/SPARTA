from oodesign import Line
import numpy as np
import scipy.sparse as sps
import const
from const import NodeSide, StudyType
from models.model import Model, ValType
from utils import phasor_to_timedomain, get_start_end_idx
from pprint import pformat

"""
1) class LineModel represent any line connected between two nodes.
2) Each instance of these class could have different no. of phases based on the line object passed to it.
3) Line object passed to the init function here comes from the object oriented representation of the network.
4) class LineModel has following functions:
    a)get_M_powerflow
    b)get_u_powerflow
    c)get_fy_powerflow
    TODO: M, K, fy, u for dynamic simulation
    TODO: Think on-> R, L, C matrices should be formed in the adapter and provided here, so that 
    these files need not be changed based on what input is being used.
"""


class LineModel(Model):
    def __init__(self, line_obj: Line):
        self.obj = line_obj

        # map all required items from Line object to the model object
        self.name = self.obj.name

        # the number of phases in this line
        self.n_ph = self.obj.n_ph

        self.num_term = 2

        # Since X (reactance) is given based on nominal frequency (60Hz), we can calculate the inductance.
        self.L = self.obj.reactance_mat / const.w_nominal

        # If analysis is to be done considering line capacitance C will store the input matrix
        # else it stores None.
        # TODO: the LineObject provides susceptance i.e B in microSiemens
        # (attribute of line object to be renamed in OOfile)
        # unit check?

        if self.obj.has_capacitance:
            Cmat = line_obj.admittance_mat * 1e-6 / const.w_nominal / 2
        else:
            Cmat = line_obj.admittance_mat * 0

        print(f">> ({self.obj.id}) Cmat: {Cmat}")

        # Final C matrix columns to correspond to pi-model of the line
        self.C = sps.bmat([[Cmat, None], [None, Cmat]])

        # Resistance matrix of the line
        self.R = self.obj.resistance_mat

        # {phase -> int}, e.g. {'A': 1, 'B': 1, 'C': 1}
        self.phases_dict = self.obj.phases

        # book keeping of the eqns
        self.num_eqns_real = 0
        self.num_eqns_complex = (
            2 * self.n_ph  # first set : KCL
            + 3 * self.n_ph  # second set : KVL
            + self.n_ph  # third set : flux linkage eqns for RL branches
            + 2 * self.n_ph  # fourth set : Cv - q
            + 2 * self.n_ph  # sixth set : eqns for current in capacitor branches
            + self.n_ph  # seventh set : eqns for voltage across RL
        )

        self.num_eqns = self.num_eqns_real + self.num_eqns_complex
        self.num_eqns_dynamic = self.num_eqns

        # book keeping for the variables
        # self.vars_real = ["w"]
        # self.vars_complex = ["V", "I", "v", "i", "q", "lamda"]

        # fmt: off
        self.num_vars_real = (
            1                   # w
        )

        self.num_vars_complex = (
            2 * self.n_ph         # V
            + 2 * self.n_ph       # I
            + 3 * self.n_ph       # v
            + 3 * self.n_ph       # i
            + 2 * self.n_ph       # q
            + self.n_ph           # lamda
        )

        self.num_vars = self.num_vars_real + self.num_vars_complex
        self.num_vars_dynamic = self.num_vars
        # fmt: on

        # # fmt: off
        # # y = [V, I, w, v, i, q, lamda]
        # self.num_vars = (
        #     2 * self.n_ph         # V
        #     + 2 * self.n_ph       # I
        #     + 1                   # w
        #     + 3 * self.n_ph       # v
        #     + 3 * self.n_ph       # i
        #     + 2 * self.n_ph       # q
        #     + self.n_ph           # lamda
        # )
        # # fmt: on

        # fmt: off
        self.var_offset_real = {
            "w": 0,
        }

        self.var_offset_complex = {
            "V": 0,
            "I": 2 * self.n_ph,
            "v": 2 * self.n_ph + 2 * self.n_ph,
            "i": 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph,
            "q": 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 2 * self.n_ph,
        }

        self.var_offset = {
            "w": 0,
            "V": 1,
            "I": 1 + 2 * self.n_ph,
            "v": 1 + 2 * self.n_ph + 2 * self.n_ph,
            "i": 1 + 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph,
            "q": 1 + 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 1 + 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph + 2 * self.n_ph,
        }
        # fmt: on

        assert len(self.var_offset_real.keys()) + len(self.var_offset_complex.keys()) == len(
            self.var_offset.keys()
        )
        assert self.num_vars == self.var_offset["lamda"] + self.n_ph

        self.var_offset_dynamic = {
            "w": 0,
            "V": 1,
            "I": 1 + 2 * self.n_ph,
            "v": 1 + 2 * self.n_ph + 2 * self.n_ph,
            "i": 1 + 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph,
            "q": 1 + 2 * self.n_ph + 2 * self.n_ph + 3 * self.n_ph + 3 * self.n_ph,
            "lamda": 1
            + 2 * self.n_ph
            + 2 * self.n_ph
            + 3 * self.n_ph
            + 3 * self.n_ph
            + 2 * self.n_ph,
        }

        print("------------------------------")
        print(f"comp: {self.obj.id}")
        print(f"num_vars: {self.num_vars}")
        print(f"num_vars: {self.num_vars_dynamic}")
        print(f"num_eqns: {self.num_eqns}")
        print("------------------------------")

    def get_basetype(self):
        return "line"

    def initial_guess(self, vals: dict) -> sps.coo_array:
        # check that the voltage phasors for the phases of this line have been passed as args
        assert (
            "v_phasors" in vals
        ), "unable to compute the initial_guess for the line, since 'v_phasors' were not provided in the 'vals'"

        # [w V I v i q lamda]
        y_0 = sps.lil_matrix((self.num_vars, 1), dtype=complex)

        v_phasors_dict = vals["v_phasors"]
        # remove redundant phases
        keys = list(v_phasors_dict.keys())
        keys_to_delete = [ph for ph in keys if ph not in self.obj.phases]
        for ph in keys_to_delete:
            del v_phasors_dict[ph]

        idx_v_start = self.var_offset["v"]
        idx_v_end = idx_v_start + self.n_ph
        v_phasors = np.array(list(v_phasors_dict.values())).reshape(-1, 1)
        y_0[idx_v_start:idx_v_end, 0] = v_phasors

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

        assert var not in [
            "v",
            "i",
        ], "to be implemented considering capacitance branches"

        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset = 0
        if side == NodeSide.TO:
            side_offset = self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset[var] + side_offset + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex[var] + side_offset + phase_offset

    def get_local_idx_i(self, val_type: ValType, ph: str) -> int:
        assert ph in self.get_phases()
        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        if val_type == ValType.REAL:
            return self.var_offset["i"] + 2 * self.n_ph + phase_offset
        elif val_type == ValType.IMAG:
            return self.var_offset_complex["i"] + 2 * self.n_ph + phase_offset

    def get_id(self):
        return self.obj.id

    def get_M_powerflow_inner(self, stage=None) -> sps.coo_array:
        """
        1)This function creates the M matrix for powerflow
        2) First we create the identity and coefficient matrices required for each eqn
            and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """

        # incidence matrix
        Id_ph = sps.identity(self.n_ph, format="coo")
        # i -> [ic1a, ic1b, ic1c, ic2a, ic2b, ic2c, ia, ib, ic]
        A = sps.bmat([[Id_ph, None, Id_ph], [None, Id_ph, -Id_ph]])

        # identity matrix of size number of rows for KCL equations.
        # number of such equations = 2 * n_ph
        Id_kcl = sps.identity(2 * self.n_ph, format="coo")

        # branch = n_ph + 2 * n_ph (for capacitance branches)
        Id_b = sps.identity(3 * self.n_ph, format="coo")

        # need to pad the first 6 column with zeros (capacitive currents)
        print(f">> self.n_ph: {self.n_ph}")
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        # creates a 3 * 9 matrix : [Zeros, Zeros, self.L]
        print(f">> self.L shape: {self.L.shape}")
        print(f">> Lmat value : {self.L}")
        L = sps.bmat([[Z_ph, Z_ph, self.L]])

        Id_l = sps.identity(self.n_ph, format="coo")

        # creates a 6 * 9 matrix: [self.C, Zeros]
        # note: v -> [vc1a, vc1b, vc1c, vc2a, vc2b, vc2c, va, vb, vc]
        Z_nph = sps.lil_matrix((2 * self.n_ph, self.n_ph), dtype=float)
        print(f">> self.id: {self.obj.id}")
        print(f">> self.C shape: {self.C.shape}")
        print(f">> Z_nph shape: {Z_nph.shape}")
        C = sps.bmat([[self.C, Z_nph]])

        # 2 capacitive branches
        Id_q = sps.identity(2 * self.n_ph, format="coo")

        # 6 * 9 matrix
        Id_cap = sps.identity(2 * self.n_ph, format="coo")
        Z_cap = sps.lil_matrix((2 * self.n_ph, self.n_ph), dtype=float)
        Imat_c = sps.bmat([[Id_cap, Z_cap]])

        # 3 * 9 matrix
        Id_v = sps.identity(self.n_ph, format="coo")
        Z_v = sps.lil_matrix((self.n_ph, 2 * self.n_ph), dtype=float)
        Imat_v = sps.bmat([[Z_v, Id_v]])

        # 3 * 9 matrix
        Z_r = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        R = sps.bmat([[Z_r, Z_r, self.R]])

        # zero vectors for w for each eqn.
        Z_w1 = sps.lil_matrix((Id_kcl.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((A.T.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_l.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_q.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((Imat_c.shape[0], 1), dtype=float)
        Z_w6 = sps.lil_matrix((Imat_v.shape[0], 1), dtype=float)

        # fmt: off
        M = sps.bmat(
            [   # w,      V,        I,      v,          i,          q,      lamda
                [Z_w1,   None,      -Id_kcl,None,       A,          None,   None  ], # 1.KCL:A*i-I=0
                [Z_w2,   -A.T,      None,   Id_b,       None,       None,   None  ], # 2.KVL:A'*V-v=0
                [Z_w3,   None,      None,   None,       L,          None,   -Id_l ], # 3.Li-lamda=0
                [Z_w4,   None,      None,   C,          None,       -Id_q,  None  ], # 4.C*v-q=0
                [Z_w5,   None,      None,   None,       -Imat_c,    None,   None  ], # 5.-i+jw*q =0
                [Z_w6,   None,      None,   -Imat_v,    R,          None,   None  ], # 6.-v+Ri+jw*lamda =0
            ]
        )
        # fmt: on

        # print(f"type(M): {type(M)}")

        return M

    # return [comp_fy_re, comp_fy_im]
    def get_fy_powerflow(
        self, y_re: sps.coo_array, y_im: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1. This function returns the non-linear terms of every equation
        2. Since frequency is a variable we have non-linear terms in eqn 6 and eqn 7 shown in the matrix above.
        3. This function is to be called from the newton-raphson method on every iteration.
        4. 'y' is the part of overall-y vector that pertains to this line.
        """
        y = y_re.astype(complex)
        y[self.num_vars_real :] += 1j * y_im

        fy = sps.lil_matrix((self.num_eqns, 1), dtype=complex)

        # y = [V, I, w, v, i, q, lamda]

        # index of w in y vector
        idx_w = self.var_offset["w"]
        w = y[idx_w, 0]
        # print(f"type(w): {type(w)}")

        # fy update for eqn 6
        # start and end index of q in y vector
        idx_q_start = self.var_offset["q"]
        idx_q_end = idx_q_start + (2 * self.n_ph)  # since each phase has two capacitive branches

        # identitfy the start and end index of eqn 6 in fy vector
        idx_eq5_start = (
            # 1 # first set (w) : equates w of component with the global w
            2 * self.n_ph  # second set : KCL
            + 3 * self.n_ph  # third set : KVL
            + self.n_ph  # fourth set : Li-lamda=0
            + 2 * self.n_ph  # fifth set : C*v-q=0
        )

        idx_eq5_end = idx_eq5_start + (2 * self.n_ph)

        # print(f"w: {w}")
        # print(f"idx_eq6_start: {idx_eq6_start}")
        # print(f"idx_eq6_end: {idx_eq6_end}")
        # print(f"idx_q_start: {idx_q_start}")
        # print(f"idx_q_end: {idx_q_end}")

        fy[idx_eq5_start:idx_eq5_end] = 0 + 1j * (w * y[idx_q_start:idx_q_end])

        # fy update for eqn 7
        # start and end index of lamda in y vector
        # fmt: off
        idx_lamda_start = self.var_offset["lamda"]
        # fmt: on
        idx_lamda_end = idx_lamda_start + self.n_ph

        # identitfy the start and end index of eqn 7 in fy vector
        idx_eq6_start = (
            # 1  # first set (w) : equates w of component with the global w
            2 * self.n_ph  # first set : KCL
            + 3 * self.n_ph  # second set : KVL
            + self.n_ph  # third set : Li-lamda=0
            + 2 * self.n_ph  # fourth set : Cv - q
            + 2 * self.n_ph  # sixth set :-i+jw*q =0
        )

        idx_eq6_end = idx_eq6_start + self.n_ph
        fy[idx_eq6_start:idx_eq6_end] = 0 + 1j * (w * y[idx_lamda_start:idx_lamda_end])

        return fy.real, fy[self.num_eqns_real :].imag

    # return four quadrants: [rr, ri, ir, ii]
    def get_pd_gy_split(
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

        # eq5_re: (orig: -w * q_im)
        w_row_offset = self.var_offset["w"]
        idx_eq5 = (
            # 1 # first set (w) : equates w of component with the global w
            2 * self.n_ph  # second set : KCL
            + 3 * self.n_ph  # third set : KVL
            + self.n_ph  # fourth set : Li-lamda=0
            + 2 * self.n_ph  # fifth set : C*v-q=0
        )
        q_im_start_offset = self.var_offset_complex["q"]
        for offset in range(2 * self.n_ph):
            lagm = lagm_real[idx_eq5 + offset, 0]
            col = self.num_vars + q_im_start_offset + offset
            pd_gy_split[w_row_offset, col] += -1 * lagm

        q_im_row_offset_start = self.num_vars + self.var_offset_complex["q"]
        for offset in range(2 * self.n_ph):
            lagm = lagm_real[idx_eq5 + offset, 0]
            row = q_im_row_offset_start + offset
            pd_gy_split[row, w_row_offset] += -1 * lagm

        # eq5_im: (orig: w * q_re)
        q_re_start_offset = self.var_offset["q"]
        for offset in range(2 * self.n_ph):
            lagm = lagm_imag[idx_eq5 + offset, 0]
            col = q_re_start_offset + offset
            pd_gy_split[w_row_offset, col] += 1 * lagm

        q_re_row_offset_start = self.var_offset["q"]
        for offset in range(2 * self.n_ph):
            lagm = lagm_imag[idx_eq5 + offset, 0]
            row = q_re_row_offset_start + offset
            pd_gy_split[row, w_row_offset] += 1 * lagm

        # eq6_re: (orig: -w * lamda_im)
        idx_eq6 = (
            # 1  # first set (w) : equates w of component with the global w
            2 * self.n_ph  # first set : KCL
            + 3 * self.n_ph  # second set : KVL
            + self.n_ph  # third set : Li-lamda=0
            + 2 * self.n_ph  # fourth set : Cv - q
            + 2 * self.n_ph  # sixth set :-i+jw*q =0
        )
        lamda_im_start_offset = self.var_offset_complex["lamda"]
        for offset in range(self.n_ph):
            lagm = lagm_real[idx_eq6 + offset, 0]
            col = self.num_vars + lamda_im_start_offset + offset
            pd_gy_split[w_row_offset, col] += -1 * lagm

        lamda_im_row_offset_start = self.num_vars + self.var_offset_complex["lamda"]
        for offset in range(self.n_ph):
            lagm = lagm_real[idx_eq6 + offset, 0]
            row = lamda_im_row_offset_start + offset
            col = w_row_offset
            pd_gy_split[row, col] += -1 * lagm

        # eq6_im: (orig: w * lamda_re)
        lamda_re_start_offset = self.var_offset["lamda"]
        for offset in range(self.n_ph):
            lagm = lagm_imag[idx_eq6 + offset, 0]
            col = lamda_re_start_offset + offset
            pd_gy_split[w_row_offset, col] += 1 * lagm

        lamda_re_row_offset_start = self.var_offset["lamda"]
        for offset in range(self.n_ph):
            lagm = lagm_imag[idx_eq6 + offset, 0]
            row = lamda_re_row_offset_start + offset
            pd_gy_split[row, w_row_offset] += 1 * lagm

        # input("continue...")

        rr = pd_gy_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_gy_split[0 : self.num_vars, self.num_vars :]
        ir = pd_gy_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_gy_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_objective(self, y_real, y_imag) -> float:
        objective = 0

        # real
        iline_re_start_offset = self.var_offset["i"] + 2 * self.n_ph
        for offset in range(self.n_ph):
            row = iline_re_start_offset + offset
            iline_re = y_real[row, 0]
            r = self.R[offset, offset]
            objective += iline_re**2 * r

        # imag
        iline_im_start_offset = self.var_offset_complex["i"] + 2 * self.n_ph
        for offset in range(self.n_ph):
            row = iline_im_start_offset + offset
            iline_im = y_imag[row, 0]
            r = self.R[offset, offset]
            objective += iline_im**2 * r

        return objective

    def get_pd_objective_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        assert self.num_vars == y_real.shape[0]
        assert self.num_vars_complex == y_imag.shape[0]

        pd_objective_split = sps.lil_matrix((self.num_vars + self.num_vars_complex, 1), dtype=float)

        # real
        iline_re_start_offset = self.var_offset["i"] + 2 * self.n_ph

        for offset in range(self.n_ph):
            row = iline_re_start_offset + offset
            iline_re = y_real[row, 0]
            r = self.R[offset, offset]
            pd_objective_split[row, 0] = 2 * r * iline_re

        # imag
        iline_im_start_offset = self.var_offset_complex["i"] + 2 * self.n_ph

        for offset in range(self.n_ph):
            row = iline_im_start_offset + offset
            iline_im = y_imag[row, 0]
            r = self.R[offset, offset]
            pd_objective_split[self.num_vars + row, 0] = 2 * r * iline_im

        return pd_objective_split[: self.num_vars], pd_objective_split[self.num_vars :]

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

        # i_re:
        i_re_start_row = self.var_offset["i"] + 2 * self.n_ph
        i_re_start_col = i_re_start_row

        for offset in range(self.n_ph):
            row = i_re_start_row + offset
            col = i_re_start_col + offset
            r = self.R[offset, offset]
            pd_pd_obj_split[row, col] = 2 * r

        # i_im:
        i_im_start_row = self.num_vars + self.var_offset_complex["i"] + 2 * self.n_ph
        i_im_start_col = i_im_start_row

        for offset in range(self.n_ph):
            row = i_im_start_row + offset
            col = i_im_start_col + offset
            r = self.R[offset, offset]
            pd_pd_obj_split[row, col] = 2 * r

        rr = pd_pd_obj_split[0 : self.num_vars, 0 : self.num_vars]
        ri = pd_pd_obj_split[0 : self.num_vars, self.num_vars :]
        ir = pd_pd_obj_split[self.num_vars :, 0 : self.num_vars]
        ii = pd_pd_obj_split[self.num_vars :, self.num_vars :]

        return (rr, ri, ir, ii)

    # return four quadrants: [rr, ri, ir, ii]
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

        w_col_offset = self.var_offset["w"]
        w = y_real[w_col_offset, 0]

        q_re_start_offset = self.var_offset["q"]
        q_im_start_offset = self.var_offset_complex["q"]
        lamda_re_start_offset = self.var_offset["lamda"]
        lamda_im_start_offset = self.var_offset_complex["lamda"]

        # eq5_re
        eq5_re_start_row = (
            2 * self.n_ph  # second set : KCL
            + 3 * self.n_ph  # third set : KVL
            + self.n_ph  # fourth set : Li-lamda=0
            + 2 * self.n_ph  # fifth set : C*v-q=0
        )
        for offset in range(2 * self.n_ph):
            row = eq5_re_start_row + offset

            q_re_col_offset = q_re_start_offset + offset
            q_im_col_offset = q_im_start_offset + offset
            q_re = y_real[q_re_col_offset, 0]
            q_im = y_imag[q_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = -q_im
            # derivative wrt q_re=0
            # q_im
            pd_fy_split[row, self.num_vars + q_im_col_offset] = -w

        # eq5_im
        eq5_im_start_row = self.num_eqns + eq5_re_start_row
        for offset in range(2 * self.n_ph):
            row = eq5_im_start_row + offset

            q_re_col_offset = q_re_start_offset + offset
            q_im_col_offset = q_im_start_offset + offset
            q_re = y_real[q_re_col_offset, 0]
            q_im = y_imag[q_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = q_re
            # q_re
            pd_fy_split[row, q_re_col_offset] = w

        # eq6_re
        eq6_re_start_row = (
            # 1  # first set (w) : equates w of component with the global w
            2 * self.n_ph  # first set : KCL
            + 3 * self.n_ph  # second set : KVL
            + self.n_ph  # third set : Li-lamda=0
            + 2 * self.n_ph  # fourth set : Cv - q
            + 2 * self.n_ph  # sixth set :-i+jw*q =0
        )
        for offset in range(self.n_ph):
            row = eq6_re_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = -lamda_im

            # derivative wrt lamda_re = 0

            # lamda_im
            pd_fy_split[row, self.num_vars + lamda_im_col_offset] = -w

        # eq6_im
        eq6_im_start_row = self.num_eqns + eq6_re_start_row
        for offset in range(self.n_ph):
            row = eq6_im_start_row + offset

            lamda_re_col_offset = lamda_re_start_offset + offset
            lamda_im_col_offset = lamda_im_start_offset + offset
            lamda_re = y_real[lamda_re_col_offset, 0]
            lamda_im = y_imag[lamda_im_col_offset, 0]

            # w
            pd_fy_split[row, w_col_offset] = lamda_re

            # lamda_re
            pd_fy_split[row, lamda_re_col_offset] = w

            # derivative wrt lamda_im = 0

        rr = pd_fy_split[0 : self.num_eqns, 0 : self.num_vars]
        ri = pd_fy_split[0 : self.num_eqns, self.num_vars :]
        ir = pd_fy_split[self.num_eqns :, 0 : self.num_vars]
        ii = pd_fy_split[self.num_eqns :, self.num_vars :]

        return (rr, ri, ir, ii)

    def get_u_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        """
        Line model u is zeros of size of no. of eqns
        """
        u = sps.lil_matrix((self.num_eqns + self.num_eqns_complex, 1), dtype=float)
        return u[: self.num_eqns], u[self.num_eqns :]

    def get_hy_powerflow(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array] | None:
        return None

    def get_pd_hy_split(
        self, y_real: sps.coo_array, y_imag: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array] | None:
        return None

    def get_pd_pd_hy_split(
        self,
        y_real: sps.coo_array,
        y_imag: sps.coo_array,
        mu_real: sps.coo_array,
        mu_imag: sps.coo_array,
    ) -> tuple[sps.coo_array, sps.coo_array, sps.coo_array, sps.coo_array] | None:
        return None

    # # powerflow => (M,fy, u)
    # # dynamic => TODO
    ########################################################################################################
    # dynamic equations
    """
    1) The dynamic equations only  require a K matrix
    2) A K matrix has the coefficients of the dynamic states
    2) The M matrix remains the same
    3) The non-linear vector shall also change
    4) The input vector also needs to be modified
    """

    def initial_guess_dynamic(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        y0_dyn[0] = y_comp[0].real
        # rest:
        y0_dyn[1:] = [np.sqrt(2) * phasor_to_timedomain(val) for val in y_comp[1:]]

        return y0_dyn

    def initial_guess_dynamic_zero(self, y_comp: list, wnom) -> np.ndarray:
        assert len(y_comp) == self.num_vars

        y0_dyn = np.zeros(self.num_vars_dynamic, dtype=float)
        # w:
        # y0_dyn[0] = y_comp[0].real

        # # we only init voltage:
        # pf_idx_V_start, pf_idx_V_end = get_start_end_idx(self.var_offset, "V", self.n_ph)
        # V = y_comp[pf_idx_V_start:pf_idx_V_end]
        # dyn_idx_V_start, dyn_idx_V_end = get_start_end_idx(self.var_offset_dynamic, "V", self.n_ph)
        # y0_dyn[dyn_idx_V_start:dyn_idx_V_end] = [
        #     np.sqrt(2) * phasor_to_timedomain(val) for val in V
        # ]

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
        assert var in self.var_offset_dynamic.keys()

        if var == "w":
            assert ph is None
            assert side is None

        assert var not in [
            "v",
            "i",
        ], "to be implemented considering capacitance branches"

        if side == NodeSide.FROM or side == NodeSide.AT or side is None:
            side_offset = 0
        if side == NodeSide.TO:
            side_offset = self.n_ph

        phase_offset = 0 if ph is None else self.get_phases().index(ph)

        return self.var_offset_dynamic[var] + side_offset + phase_offset

    def get_K_dynamic(self, stage=None) -> sps.coo_array:
        """
        1) This function creates the K matrix for dynamic equations
        2) First we create the identity and coefficient matrices required for each eqn
            and then place them in the matrix using sps.bmat.
        3)Id_*: identity matrix
        4)Z_*: matrix of zeros

        """
        # Note: later make these on self, so they are available to both M and K matrices
        Id_ph = sps.identity(self.n_ph, format="coo")
        Z_ph = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        Z_w = sps.lil_matrix((self.n_ph, 1), dtype=float)

        A = sps.bmat([[Id_ph, None, Id_ph], [None, Id_ph, -Id_ph]])
        Z_A = sps.lil_matrix((A.shape[0], A.shape[1]))
        Z_A_T = sps.lil_matrix((A.shape[1], A.shape[0]), dtype=float)
        Z_kcl = sps.lil_matrix((2 * self.n_ph, 2 * self.n_ph), dtype=float)
        Id_b = sps.identity(3 * self.n_ph, format="coo")
        Z_b = sps.lil_matrix((Id_b.shape[0], Id_b.shape[1]), dtype=float)
        L = sps.bmat([[Z_ph, Z_ph, self.L]])
        Z_L = sps.lil_matrix((L.shape[0], L.shape[1]), dtype=float)
        Id_l = sps.identity(self.n_ph, format="coo")
        Z_nph = sps.lil_matrix((2 * self.n_ph, self.n_ph), dtype=float)
        C = sps.bmat([[self.C, Z_nph]])
        Z_C = sps.lil_matrix((C.shape[0], C.shape[1]), dtype=float)
        Id_q = sps.identity(2 * self.n_ph, format="coo")
        Z_Id_q = sps.lil_matrix((Id_q.shape[0], Id_q.shape[1]), dtype=float)
        Id_cap = sps.identity(2 * self.n_ph, format="coo")
        Z_cap = sps.lil_matrix((2 * self.n_ph, self.n_ph), dtype=float)
        Imat_c = sps.bmat([[Id_cap, Z_cap]])
        Z_Imat_c = sps.lil_matrix((Imat_c.shape[0], Imat_c.shape[1]), dtype=float)
        Id_v = sps.identity(self.n_ph, format="coo")
        Z_v = sps.lil_matrix((self.n_ph, 2 * self.n_ph), dtype=float)
        Imat_v = sps.bmat([[Z_v, Id_v]])
        Z_Imat_v = sps.lil_matrix((Imat_v.shape[0], Imat_v.shape[1]), dtype=float)
        Z_r = sps.lil_matrix((self.n_ph, self.n_ph), dtype=float)
        R = sps.bmat([[Z_r, Z_r, self.R]])
        Z_R = sps.lil_matrix((R.shape[0], R.shape[1]), dtype=float)

        Z_w1 = sps.lil_matrix((Z_kcl.shape[0], 1), dtype=float)
        Z_w2 = sps.lil_matrix((A.T.shape[0], 1), dtype=float)
        Z_w3 = sps.lil_matrix((Id_l.shape[0], 1), dtype=float)
        Z_w4 = sps.lil_matrix((Id_q.shape[0], 1), dtype=float)
        Z_w5 = sps.lil_matrix((Imat_c.shape[0], 1), dtype=float)
        Z_w6 = sps.lil_matrix((Imat_v.shape[0], 1), dtype=float)

        # fmt: off
        K = sps.bmat([
                    # w,      V,         I,         v,          i,          q,      lamda
                    [Z_w1,   None,      Z_kcl,       None,      Z_A,       None,    None    ], # 1.KCL:A*i-I=0  (6)
                    [Z_w2,   Z_A_T,     None,       Z_b,        None,      None,    None    ], # 2.KVL:A'*V-v=0 (9)
                    [Z_w3,   None,      None,       None,       Z_L,       None,    None    ], # 3.Li-lamda=0 (3)
                    [Z_w4,   None,      None,       Z_C,        None,      Z_Id_q,  None    ], # 4.C*v-q=0 (6)
                    [Z_w5,   None,      None,       None,       Z_Imat_c,  Id_q,    None    ], # 5.-i+d(q)/dt =0 (6)
                    [Z_w6,   None,      None,       Z_Imat_v,   Z_R,        None,     Id_l   ], # 6.-v+Ri+d(lamda)/dt =0 (3)
                ]
            )
        # fmt: on

        assert K.shape[0] == self.num_eqns_dynamic
        assert K.shape[1] == self.num_vars_dynamic

        return K

    def get_M_dynamic(self, stage=None) -> sps.coo_array:
        return self.get_M_powerflow_inner(stage)

    # def get_fy_dynamic
    def get_fy_dynamic(self, t, y, yp: np.ndarray, stage) -> sps.coo_array:
        """
        The fy vector for the line is empty
        """
        fy = np.zeros(self.num_eqns, dtype=float)
        # fy = sps.lil_matrix((self.num_eqns, 1), dtype=float)
        return fy

    # def get_u_dynamic
    def get_u_dynamic(self, t: float, y) -> np.ndarray:
        u = np.zeros(self.num_eqns, dtype=float)
        return u
