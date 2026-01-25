import numpy as np
import pandas as pd
import scipy.sparse as sps

# Specify the path where you want to save the Excel file
excel_file_path = "./Powerflow_data.xlsx"


def stack_sparse_matrix(mat, n_rows, n_cols, position, offset=None):
    """
    Stack a sparse matrix in specified position of final sparse matrix.

    Args:
        mat (scipy.sparse.spmatrix): The sparse matrix to be stacked.
        n_rows (int): The number of rows in the final matrix.
        n_cols (int): The number of columns in the final matrix.
        position (str): The position where the matrix should be stacked.
    Returns:
        scipy.sparse.spmatrix: The updated sparse matrix with the input
         matrix stacked in the specified position.
    """

    # row and column represents axis 0 and axis 1 respectively
    updated_mat = sps.lil_matrix((n_rows, n_cols))
    if position == "bottom-left":
        # Determine the starting row for copying
        start_row = n_rows - mat.shape[0]
        # Copy non-zero elements to the bottom-left corner of the new matrix
        updated_mat[start_row:, : mat.shape[1]] = mat

        return updated_mat.tocoo()

    elif position == "bottom-right":
        # Determine the starting row and column for copying
        start_row = n_rows - mat.shape[0]
        start_col = n_cols - mat.shape[1]

        # Copy non-zero elements to the bottom-right corner of the new matrix
        updated_mat[start_row:, start_col:] = mat

        return updated_mat.tocoo()

    elif position == "top-left":
        # Copy non-zero elements to the top-left corner of the new matrix
        updated_mat[: mat.shape[0], : mat.shape[1]] = mat

        return updated_mat.tocoo()

    elif position == "top-right":
        # Determine the starting column for copying
        start_col = n_cols - mat.shape[1]
        # Copy non-zero elements to the top-right corner of the new matrix
        updated_mat[: mat.shape[0], start_col:] = mat

        return updated_mat.tocoo()

    elif position == "middle-right":
        # Determine the starting column for copying
        start_col = n_cols - mat.shape[1]
        start_row = n_rows - mat.shape[0] - offset
        end_row = n_rows - offset
        # Copy non-zero elements to the top-right corner of the new matrix
        updated_mat[start_row:end_row, start_col:] = mat

        return updated_mat.tocoo()

    raise ValueError("The specified position is not supported.")


def stack_numpy_vector(vec, n_rows, position, offset=None):
    """
    Stack a numpy vector vertically by adding rows at the specified position.

    Args:
        vec (numpy.ndarray): The input vector to be stacked.
        n_rows (int): The desired number of rows in the resulting stacked vector.
        position (str): The position where the rows should be added. Can be "bottom" or "top".

    Returns:
        numpy.ndarray: The stacked vector with the specified number of rows.

    """

    # Determine the number of rows to add to the vector
    rows_to_add = max(0, n_rows - vec.shape[0])

    if position == "top":
        updated_vec = np.pad(vec, ((0, rows_to_add), (0, 0)), mode="constant")

        return updated_vec

    if position == "bottom":
        updated_vec = np.pad(vec, ((rows_to_add, 0), (0, 0)), mode="constant")

        return updated_vec

    if position == "middle":
        rows_to_add = offset

        updated_vec = np.pad(
            vec,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
        )

        return updated_vec

    raise ValueError("The specified position is not supported.")


def calculate_terms(dict_1, value, dict_3=None, operation_type="add"):
    # Create a new dictionary to store the result
    result = {}

    # Iterate through the keys and perform the specified operation
    for key in dict_1:
        if operation_type == "multiply":
            result[key] = dict_1[key] * value

        elif operation_type == "divide":
            result[key] = dict_1[key] / value

        elif operation_type == "impedance":
            result[key] = dict_1[key] ** 2 * dict_3[key] / value

        elif operation_type == "denom_squared":
            result[key] = dict_1[key] / value**2

        else:
            raise ValueError("Invalid operation type specified.")

    return result


def dict_operations(dict_1, dict_2=None, dict_3=None, operation_type="add", flag=0):
    # Check if keys are same
    result, result_1 = {}, {}
    if dict_2 is None:
        for key in dict_1:
            if operation_type == "abs":
                result[key] = np.abs(dict_1[key])
    else:
        if dict_1.keys() == dict_2.keys():
            # Create a new dictionary to store the result
            # Iterate through the keys and perform the specified operation
            for key in dict_1:
                if operation_type == "add":
                    result[key] = dict_1[key] + dict_2[key]

                elif operation_type == "subtract":
                    result[key] = dict_1[key] - dict_2[key]

                elif operation_type == "multiply":
                    result[key] = dict_1[key] * dict_2[key]

                elif operation_type == "divide":
                    result[key] = dict_1[key] / dict_2[key]

                elif operation_type == "impedance":
                    result[key] = dict_1[key] ** 2 * dict_3[key] / dict_2[key] ** 2

                elif operation_type == "denom_squared":
                    result[key] = dict_1[key] / dict_2[key] ** 2

                elif operation_type == "complex":
                    result[key] = dict_1[key] + 1j * dict_2[key]

                elif operation_type == "norm":
                    result[key] = np.sqrt(dict_1[key] ** 2 + dict_2[key] ** 2)

                elif operation_type == "angle":
                    result[key] = np.arctan2(dict_1[key], dict_2[key])

                elif operation_type == "currentphasors":
                    result[key] = (
                        dict_1[key] * np.real(dict_3[key])
                        + dict_2[key] * np.imag(dict_3[key])
                    ) / (np.real(dict_3[key]) ** 2 + np.imag(dict_3[key]) ** 2)

                    result_1[key] = (
                        dict_1[key] * np.imag(dict_3[key])
                        - dict_2[key] * np.real(dict_3[key])
                    ) / (np.real(dict_3[key]) ** 2 + np.imag(dict_3[key]) ** 2)

                else:
                    raise ValueError("Invalid operation type specified.")
        else:
            raise ValueError("Keys are not the same, cannot perform operation.")
    if flag == 1:
        return result, result_1
    return result


def get_vector_phasors(F_m: dict[str, float]) -> dict[str, complex]:
    f = {}

    if "A" in F_m.keys():
        f["A"] = F_m["A"] * np.cos(0) + 1j * F_m["A"] * np.sin(0)

    if "B" in F_m.keys():
        f["B"] = F_m["B"] * np.cos(-2 * np.pi / 3) + 1j * F_m["B"] * np.sin(
            -2 * np.pi / 3
        )

    if "C" in F_m.keys():
        f["C"] = F_m["C"] * np.cos(2 * np.pi / 3) + 1j * F_m["C"] * np.sin(
            2 * np.pi / 3
        )

    return f


def get_phase_phase_values(f):
    f_line_line = {}

    if "A" in f.keys() and "B" in f.keys():
        f_line_line["AB"] = f["A"] - f["B"]

    if "B" in f.keys() and "C" in f.keys():
        f_line_line["BC"] = f["B"] - f["C"]

    if "C" in f.keys() and "A" in f.keys():
        f_line_line["CA"] = f["C"] - f["A"]

    return f_line_line


def get_line_values(f):
    A = np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
    if f.keys() == {"AB", "BC", "CA"}:
        f_line = np.array([f["AB"], f["BC"], f["CA"]])
        return A @ f_line
    elif f.keys() == {"AB", "BC"}:
        f_line = np.array([f["AB"], f["BC"], 0])
        return A @ f_line
    elif f.keys() == {"BC", "CA"}:
        f_line = np.array([0, f["BC"], f["CA"]])
        return A @ f_line
    elif f.keys() == {"AB", "CA"}:
        f_line = np.array([f["AB"], 0, f["CA"]])
        return A @ f_line
    elif f.keys() == {"AB"}:
        f_line = np.array([f["AB"], 0, 0])
        return A @ f_line
    elif f.keys() == {"BC"}:
        f_line = np.array([0, f["BC"], 0])
        return A @ f_line
    elif f.keys() == {"CA"}:
        f_line = np.array([0, 0, f["CA"]])
        return A @ f_line
    else:
        return None


def vars_lists(pf_results):
    """
    pf_results is an object of the PowerFlow class

    This function returns a list of all the variables in the TDS format

    TDS format is a list of all the variables in the following order:
    node_voltage_list
    node_current_list
    branch_voltage_list
    branch_current_list
    branch_flux_linkage_list

    return type: list
    returns: TDS_list
    """

    n_var_real = pf_results.n_var_real
    y = pf_results.y
    n_bus = pf_results.n_bus
    # n_lines = pf_results.n_lines

    # node_voltage_data = _from__phasor_to_TDS(
    #     0, n_bus, n_var_real, y)

    # node_current_list, node_curr_real, node_curr_imag = _from__phasor_to_TDS(
    #     n_bus, 2*n_bus, n_var_real, y)

    # print("node_real current", node_curr_real)
    # print("node_imag current", node_curr_imag)

    # branch_voltage_list, branch_volt_real, branch_volt_imag = _from__phasor_to_TDS(
    #     2*n_bus, 2*n_bus + n_lines, n_var_real, y)

    # print("branch_real voltage", branch_volt_real)
    # print("branch_imag voltage", branch_volt_imag)

    # branch_current_list, branch_curr_real, branch_curr_imag = _from__phasor_to_TDS(
    #     2*n_bus + n_lines, 2*n_bus + 2*n_lines, n_var_real, y)

    # print("branch_real current", branch_curr_real)
    # print("branch_imag current", branch_curr_imag)

    # branch_flux_linkage_list, branch_fl_real, branch_fl_imag = _from__phasor_to_TDS(
    #     2*n_bus + 2*n_lines, 2*n_bus + 3*n_lines, n_var_real, y)

    # print("branch_real flux linkage", branch_fl_real)
    # print("branch_imag flux linkage", branch_fl_imag)

    data = _from__phasor_to_TDS(0, n_var_real, y, 0)

    complex_power = _from__phasor_to_TDS(2 * n_var_real, n_bus, y, 1)

    # save the data to excel file
    # save_to_excel(data, complex_power, n_bus, n_lines)
    save_to_excel(data, complex_power, n_bus, 0)

    return data, complex_power


def save_to_excel_helper(
    data_sets,
    complex_power,
    writer,
    sheet_names,
    n_bus,
    n_lines,
    column_width=12,
):
    cnt = 0
    for i, sheet_name in enumerate(sheet_names):
        # Create a DataFrame from the current set of lists
        if i < 2:
            df = pd.DataFrame(
                {
                    "TDS": data_sets[0][i * n_bus : (i + 1) * n_bus],
                    "Magnitude": data_sets[1][i * n_bus : (i + 1) * n_bus],
                    "Phase": data_sets[2][i * n_bus : (i + 1) * n_bus],
                    "Real Part": data_sets[3][i * n_bus : (i + 1) * n_bus],
                    "Imaginary Part": data_sets[4][i * n_bus : (i + 1) * n_bus],
                }
            )
        else:
            # tmp: anonymous
            continue
            df = pd.DataFrame(
                {
                    "TDS": data_sets[0][
                        2 * n_bus + cnt * n_lines : 2 * n_bus + (cnt + 1) * n_lines
                    ],
                    "Magnitude": data_sets[1][
                        2 * n_bus + cnt * n_lines : 2 * n_bus + (cnt + 1) * n_lines
                    ],
                    "Phase": data_sets[2][
                        2 * n_bus + cnt * n_lines : 2 * n_bus + (cnt + 1) * n_lines
                    ],
                    "Real Part": data_sets[3][
                        2 * n_bus + cnt * n_lines : 2 * n_bus + (cnt + 1) * n_lines
                    ],
                    "Imaginary Part": data_sets[4][
                        2 * n_bus + cnt * n_lines : 2 * n_bus + (cnt + 1) * n_lines
                    ],
                }
            )
            cnt += 1

        if sheet_name == "Complex Power":
            df = pd.DataFrame(
                {
                    "TDS": complex_power[0],
                    "Magnitude": complex_power[1],
                    "Phase": complex_power[2],
                    "Real Power": complex_power[3],
                    "Reactive Power": complex_power[4],
                }
            )

        # Save the DataFrame to an Excel sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # set column width for readability
        worksheet = writer.sheets[sheet_name]
        for i, width in enumerate([column_width] * len(data_sets)):
            worksheet.set_column(i, i, width)


def save_to_excel(data_sets, complex_power, n_bus, n_lines):

    # create different sheet names list
    sheet_names = [
        "Node Voltage",
        "Node Injections",
        "Branch Voltage",
        "Branch Current",
        "Branch Flux Linkage",
        "Complex Power",
    ]

    # Create an ExcelWriter object to handle multiple sheets
    with pd.ExcelWriter(excel_file_path, engine="xlsxwriter") as writer:
        # Call the function to save each set of lists to a separate sheet
        save_to_excel_helper(
            data_sets, complex_power, writer, sheet_names, n_bus, n_lines
        )


def _from__phasor_to_TDS(a, b, y, flag):
    """
    a and b starting and ending index of the phasor list
    y is the vector of variables
    flag is 0 or 1 depending on whether the phasor is power


    tmp: anonymous
    returned list is of the form:
    [
        [list of phasor magnitudes],
        [list of magnitudes],
        [list of angles in degrees],
        [list of real parts],
        [list of imaginary parts],
    ]
    """
    # convert to numpy array

    if flag == 0:
        real_parts = np.array(y[a:b])
        imag_parts = np.array(y[b : 2 * b])
    else:
        real_parts = np.array(y[a : a + b])
        imag_parts = np.array(y[a + b : 2 * (a + b)])

    # find the angle of the complex number array in degrees
    # arctan2 is used to indicate the correct quadrant of the angle which is not possible with arctan

    angle = np.arctan2(imag_parts, real_parts)

    # find the magnitude of the complex number array
    magnitude = np.sqrt(np.square(real_parts) + np.square(imag_parts))

    # return the multiplication of magnitude and angle

    print(f">> n_each_elem: {len(real_parts)}")

    total_len = len(
        (
            list(np.sqrt(2) * magnitude * np.cos(angle)),
            list(magnitude),
            list(np.rad2deg(angle)),
            list(real_parts),
            list(imag_parts),
        )
    )
    print(f">> total_len: {total_len}")

    return (
        list(np.sqrt(2) * magnitude * np.cos(angle)),
        list(magnitude),
        list(np.rad2deg(angle)),
        list(real_parts),
        list(imag_parts),
    )


# def _from__phasor_to_TDS(a, b, n, y):
#     """
#     a and b starting and ending index of the phasor list
#     y is the vector of variables
#     flag is 0 or 1 depemding on whether the phasor is power
#     """
#     # convert to numpy array


#     real_parts = np.array(y[a:b])
#     imag_parts = np.array(y[n + a : n + b])

#     # find the angle of the complex number array in degrees
#     # arctan2 is used to indicate the correct quadrant of the angle which is not possible with arctan

#     angle = np.arctan2(imag_parts, real_parts)

#     # find the magnitude of the complex number array
#     magnitude = np.sqrt(np.square(real_parts) + np.square(imag_parts))

#     # return the multiplication of magnitude and angle

#     return (magnitude * np.cos(angle), magnitude, angle, real_parts, imag_parts)


def calc_effective_reg_ratio(tap: int, reg_type: str) -> float:
        assert reg_type in ["A", "B"], "Invalid regulator type"

        volts_per_tap = 0.00625

        if reg_type == "A":
            effective_reg_ratio = 1 + volts_per_tap * tap
        elif reg_type == "B":
            effective_reg_ratio = 1 - volts_per_tap * tap

        return effective_reg_ratio

def phasor_to_timedomain(v: complex | np.ndarray) -> float | np.ndarray:
    mag = np.abs(v)
    angle = np.angle(v)
    val = mag * np.cos(angle)
    return val

def get_start_end_idx(d: dict, key: str, count: int) -> tuple[int, int]:
    start_idx = d[key]
    end_idx = start_idx + count
    return (start_idx, end_idx)

def get_start_end_idx_with_offset(d: dict, key: str, start_offset: int, count: int) -> tuple[int, int]:
    start_idx = d[key] + start_offset
    end_idx = start_idx + count
    return (start_idx, end_idx)

def abc_to_qd(abc: np.ndarray, theta: float) -> np.ndarray:
    transform = (2/3) * np.array([
        [np.cos(theta), np.cos(theta - 2*np.pi/3), np.cos(theta + 2*np.pi/3)],
        [np.sin(theta), np.sin(theta - 2*np.pi/3), np.sin(theta + 2*np.pi/3)]
    ])
    return transform @ abc

def qd_to_abc(qd: np.ndarray, theta: float) -> np.ndarray:
    transform = np.array([
        [np.cos(theta), np.sin(theta)],
        [np.cos(theta - 2*np.pi/3), np.sin(theta - 2*np.pi/3)],
        [np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)]
    ])
    return transform @ qd