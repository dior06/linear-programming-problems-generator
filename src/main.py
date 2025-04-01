import subprocess
import numpy as np

def normalize_constraints(A, b, con_types):
    m, n = A.shape
    A_out = A.copy().astype(float)
    b_out = b.copy().astype(float)
    need_artificial = [False] * m
    for i in range(m):
        t = con_types[i]
        if t == "<=":
            pass
        elif t == ">=":
            A_out[i] = -A_out[i]
            b_out[i] = -b_out[i]
            need_artificial[i] = True
        elif t == "=":
            if b_out[i] < 0:
                A_out[i] = -A_out[i]
                b_out[i] = -b_out[i]
            need_artificial[i] = True
        else:
            raise ValueError("Unknown constraint type: " + t)
    return A_out, b_out, need_artificial

def build_phase1_table(A_le, b_le, need_art):
    m, n = A_le.shape
    slack_count = sum(not f for f in need_art)
    art_count = sum(f for f in need_art)
    total_vars = n + slack_count + art_count
    phase1_table = np.zeros((m + 1, total_vars + 1), dtype=float)
    slack_indices = []
    art_indices = []
    basis = [None] * m
    slack_pos = n
    art_pos = n + slack_count
    for i in range(m):
        phase1_table[i, :n] = A_le[i, :]
        phase1_table[i, -1] = b_le[i]
        if not need_art[i]:
            phase1_table[i, slack_pos] = 1.0
            basis[i] = slack_pos
            slack_indices.append(slack_pos)
            slack_pos += 1
        else:
            phase1_table[i, art_pos] = 1.0
            basis[i] = art_pos
            art_indices.append(art_pos)
            art_pos += 1
    for ac in art_indices:
        phase1_table[m, ac] = -1.0
    for i in range(m):
        if basis[i] in art_indices:
            phase1_table[m, :] -= phase1_table[i, :]
    return phase1_table, basis, slack_indices, art_indices

def format_number(x):
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    else:
        return f"{x:.2f}"

def make_var_names(n, slack_count=0, art_count=0):
    names = []
    for i in range(n):
        names.append(f"x{i+1}")
    for i in range(slack_count):
        names.append(f"a{i+1}")
    for i in range(art_count):
        names.append(f"s{i+1}")
    return names

def run_phase1_iterations(phase1_table, basis, art_indices, tol=1e-10, max_iter=1000, save_logs=False, n=0, slack_count=0, art_count=0):
    m = phase1_table.shape[0] - 1
    nvars = phase1_table.shape[1] - 1
    var_names = make_var_names(n, slack_count, art_count)
    col_labels = var_names + ["RHS"]
    logs = [] if save_logs else None
    def snap(iterc):
        if logs is not None:
            lines = []
            lines.append(f"\n \\textbf{{Итерация {iterc}}}\\\\")
            lines.append("\\begin{tabular}{|c|" + "c|" * (len(col_labels)) + "}")
            lines.append("\\hline")
            header = ["Базис"] + col_labels
            lines.append(" & ".join(header) + " \\\\ \\hline")
            for row_i in range(m):
                b_idx = basis[row_i]
                row_label = var_names[b_idx] if b_idx < len(var_names) else f"Row{row_i}"
                row_vals = [format_number(v) for v in phase1_table[row_i, :]]
                lines.append(" & ".join([row_label] + row_vals) + " \\\\ \\hline")
            obj_vals = [format_number(v) for v in phase1_table[m, :]]
            lines.append(" & ".join(["$z$"] + obj_vals) + " \\\\ \\hline")
            lines.append("\\end{tabular}")
            lines.append("\\vspace{5mm}")
            logs.append("\n".join(lines))
    def pivot_op(prow, pcol):
        piv = phase1_table[prow, pcol]
        phase1_table[prow, :] /= piv
        for rr in range(phase1_table.shape[0]):
            if rr != prow:
                fac = phase1_table[rr, pcol]
                phase1_table[rr, :] -= fac * phase1_table[prow, :]
    itc = 0
    while itc < max_iter:
        obj_row = phase1_table[m, :nvars]
        pivot_col = np.argmin(obj_row)
        min_val = obj_row[pivot_col]
        if min_val >= -tol:
            break
        pivot_row = -1
        best_rat = None
        for i in range(m):
            el = phase1_table[i, pivot_col]
            if el > tol:
                ratio = phase1_table[i, -1] / el
                if pivot_row < 0 or ratio < best_rat:
                    best_rat = ratio
                    pivot_row = i
        if pivot_row < 0:
            return False, phase1_table, basis, ("\n".join(logs) if logs else None)
        snap(itc)
        pivot_op(pivot_row, pivot_col)
        basis[pivot_row] = pivot_col
        itc += 1
    if itc >= max_iter:
        return False, phase1_table, basis, ("\n".join(logs) if logs else None)
    sum_art = -phase1_table[m, -1]
    feasible = (sum_art <= tol)
    return feasible, phase1_table, basis, ("\n".join(logs) if logs else None)

def prepare_phase2_table(phase1_table, basis, c, n, slack_count, art_count):
    m = phase1_table.shape[0] - 1
    new_nvars = n + slack_count
    table2 = np.zeros((m + 1, new_nvars + 1), dtype=float)
    table2[:m, :new_nvars] = phase1_table[:m, :new_nvars]
    table2[:m, -1] = phase1_table[:m, -1]
    for j in range(n):
        table2[m, j] = -c[j]
    for j in range(n, new_nvars):
        table2[m, j] = 0.0
    for i in range(m):
        colb = basis[i]
        if colb < n:
            cc = c[colb]
            if abs(cc) > 1e-12:
                table2[m, :] += cc * table2[i, :]
    return table2

def run_phase2_iterations(table2, basis, n, slack_count, tol=1e-10, max_iter=1000, save_logs=False):
    m = table2.shape[0] - 1
    nvars = n + slack_count
    var_names = make_var_names(n, slack_count, art_count=0)
    col_labels = var_names + ["RHS"]
    logs = [] if save_logs else None
    def snap(itc):
        if logs is not None:
            lines = []
            lines.append(f"\n \\textbf{{Итерация {itc}}}\\\\")
            lines.append("\\begin{tabular}{|c|" + "c|" * (len(col_labels)) + "}")
            lines.append("\\hline")
            header = ["Базис"] + col_labels
            lines.append(" & ".join(header) + " \\\\ \\hline")
            for row_i in range(m):
                b_idx = basis[row_i]
                row_label = var_names[b_idx] if b_idx < len(var_names) else f"Row{row_i}"
                row_vals = [format_number(v) for v in table2[row_i, :]]
                lines.append(" & ".join([row_label] + row_vals) + " \\\\ \\hline")
            obj_vals = [format_number(v) for v in table2[m, :]]
            lines.append(" & ".join(["$z$"] + obj_vals) + " \\\\ \\hline")
            lines.append("\\end{tabular}")
            lines.append("\\vspace{5mm}")
            logs.append("\n".join(lines))
    def pivot_op(prow, pcol):
        piv = table2[prow, pcol]
        table2[prow, :] /= piv
        for rr in range(table2.shape[0]):
            if rr != prow:
                fac = table2[rr, pcol]
                table2[rr, :] -= fac * table2[prow, :]
    itc = 0
    while itc < max_iter:
        obj_row = table2[m, :nvars]
        pivot_col = np.argmin(obj_row)
        min_val = obj_row[pivot_col]
        if min_val >= -tol:
            break
        pivot_row = -1
        best_ratio = None
        for i in range(m):
            el = table2[i, pivot_col]
            if el > tol:
                ratio = table2[i, -1] / el
                if pivot_row < 0 or ratio < best_ratio:
                    best_ratio = ratio
                    pivot_row = i
        if pivot_row < 0:
            return False, table2, basis, ("\n".join(logs) if logs else None), itc
        snap(itc)
        pivot_op(pivot_row, pivot_col)
        basis[pivot_row] = pivot_col
        itc += 1
    if itc >= max_iter:
        return False, table2, basis, ("\n".join(logs) if logs else None), itc
    return True, table2, basis, ("\n".join(logs) if logs else None), itc

def extract_solution(table2, basis, n, slack_count, art_count, tol=1e-10):
    m = table2.shape[0] - 1
    x_opt = np.zeros(n, dtype=float)
    for i in range(m):
        col_b = basis[i]
        val = table2[i, -1]
        if col_b < n:
            x_opt[col_b] = val
    real_obj = table2[m, -1]
    if any(x < -tol for x in x_opt):
        return None, None, False
    return x_opt, real_obj, True

def check_solution_feasibility(A, b, con_types, x_opt, tol=1e-10):
    m = A.shape[0]
    lhs = A.dot(x_opt)
    for i in range(m):
        if con_types[i] == "<=":
            if lhs[i] > b[i] + tol:
                return False
        elif con_types[i] == ">=":
            if lhs[i] < b[i] - tol:
                return False
        elif con_types[i] == "=":
            if abs(lhs[i] - b[i]) > tol:
                return False
        else:
            return False
    for x in x_opt:
        if x < -tol:
            return False
    return True

def solve_LP_two_phase(A, b, con_types, c):
    A_le, b_le, need_art = normalize_constraints(A, b, con_types)
    m, n = A_le.shape
    slack_count = sum(not f for f in need_art)
    art_count = sum(f for f in need_art)
    p1_table, basis, sidx, aidx = build_phase1_table(A_le, b_le, need_art)
    feasible1, p1_table, basis, log1 = run_phase1_iterations(
        p1_table,
        basis,
        aidx,
        tol=1e-10,
        max_iter=1000,
        save_logs=True,
        n=n,
        slack_count=slack_count,
        art_count=art_count
    )
    for i in range(m):
        if basis[i] >= n + slack_count:
            for j in range(n + slack_count):
                if abs(p1_table[i, j]) > 1e-10:
                    piv = p1_table[i, j]
                    p1_table[i, :] /= piv
                    for k in range(m + 1):
                        if k != i:
                            fac = p1_table[k, j]
                            p1_table[k, :] -= fac * p1_table[i, :]
                    basis[i] = j
                    break
    sum_art = -p1_table[m, -1]
    feasible1 = (sum_art <= 1e-10)
    if not feasible1:
        return (False, None, None, "Infeasible\n" + (log1 or ""))
    table2 = prepare_phase2_table(p1_table, basis, c, n, slack_count, art_count)
    logs = log1 or ""
    ok2, final_table, final_basis, log2, itc = run_phase2_iterations(
        table2, basis, n, slack_count, save_logs=True
    )
    if itc > 0:
        logs += "\n\n\\textbf{Фаза 2:}\n\n" + (log2 or "")
    if not ok2:
        return (False, None, None, "Unbounded or stuck\n" + logs)
    x_opt, f_val, nonneg_ok = extract_solution(final_table, final_basis, n, slack_count, art_count)
    if not nonneg_ok:
        return (False, None, None, "X had negative entries\n" + logs)
    feasible = check_solution_feasibility(A, b, con_types, x_opt)
    if not feasible:
        return (False, None, None, "Solution violates constraints\n" + logs)
    return (True, x_opt, f_val, logs)

def generate_LP_task(num_vars_range=(2,5), num_constraints_range=(2,5), coef_range=(-5,5)):
    n_min, n_max = num_vars_range
    m_min, m_max = num_constraints_range
    num_vars = np.random.randint(n_min, n_max+1)
    num_constraints = np.random.randint(m_min, m_max+1)
    A = np.random.randint(coef_range[0], coef_range[1]+1, (num_constraints, num_vars))
    b = np.random.randint(1, 6, size=num_constraints)
    c = np.random.randint(coef_range[0], coef_range[1]+1, size=num_vars)
    con_types = np.random.choice(["<=", ">=", "="], size=num_constraints)
    return A, b.astype(float), c.astype(float), con_types

def generate_tex(tasks, tex_filename="tasks_solutions.tex"):
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(r"\documentclass[a4paper,12pt]{article}" "\n")
        f.write(r"\usepackage[utf8]{inputenc}" "\n")
        f.write(r"\usepackage[T2A]{fontenc}" "\n")
        f.write(r"\usepackage[english,russian]{babel}" "\n")
        f.write(r"\usepackage{amsmath,amssymb}" "\n")
        f.write(r"\usepackage{geometry}" "\n")
        f.write(r"\geometry{left=2cm,right=2cm,top=2cm,bottom=2cm}" "\n")
        f.write(r"\begin{document}" "\n\n")
        f.write(r"\section*{Сгенерированные задачи ЛП и их решения}" "\n\n")
        for info in tasks:
            i = info["task_number"]
            A = info["A"]
            b = info["b"]
            c = info["c"]
            ctypes = info["con_types"]
            found = info["solution_found"]
            f.write(r"\subsection*{Задача №" + str(i) + "}" "\n")
            if not found:
                f.write(r"\textbf{Задача не имеет решения или возникла ошибка.}" "\n\n")
                continue
            f.write(r"\textbf{Функция цели: }" "\n")
            parts = []
            for idx, cf in enumerate(c):
                sign = "+" if (cf >= 0 and idx > 0) else ""
                parts.append(f"{sign}{format_number(cf)}x_{{{idx+1}}}")
            f.write(f"Найти максимум $ {' '.join(parts)} $\\\\" "\n\n")
            f.write(r"\textbf{Ограничения:}" "\n\n")
            f.write(r"\[ \begin{aligned}" "\n")
            for row_i, rowval in enumerate(A):
                rowExpr = []
                for col_j, val_j in enumerate(rowval):
                    s = "+" if (val_j >= 0 and col_j > 0) else ""
                    rowExpr.append(f"{s}{format_number(val_j)}x_{{{col_j+1}}}")
                if ctypes[row_i] == "<=":
                    rel = r"\le"
                elif ctypes[row_i] == ">=":
                    rel = r"\ge"
                else:
                    rel = "="
                f.write(" ".join(rowExpr) + f" &{rel} {format_number(b[row_i])} \\\\" "\n")
            f.write("x_i &\\ge 0,\\quad i=1,..," + str(len(c)) + "\\\\" "\n")
            f.write(r"\end{aligned}\]" "\n\n")
            val = info["optimal_value"]
            xsol = info["solution_vector"]
            f.write(r"\textbf{Оптимальное значение: }" + "$" + format_number(val) + "$" "\n\n")
            solStr = ", ".join([f"x_{{{vv+1}}}={format_number(xsol[vv])}" for vv in range(len(xsol))])
            f.write(r"\textbf{Решение: }" + "$" + solStr + "$" "\n\n")
            if info["steps"]:
                f.write(r"\textbf{Шаги двухфазного метода (лог):}" "\n\n")
                f.write(info["steps"])
                f.write("\n\n")
            f.write("\n\n")
        f.write(r"\end{document}")

def compile_tex_to_pdf(tex_filename):
    try:
        subprocess.run(["pdflatex", tex_filename], check=True)
        print("PDF создан:", tex_filename)
    except FileNotFoundError:
        print("pdflatex не найден.")
    except subprocess.CalledProcessError:
        print("Ошибка pdflatex.")

def main_demo(num_tasks=3, max_regen_attempts=100,
              output_file_tex="tasks_solutions.tex", generate_pdf=True):
    tasks_info = []
    for i in range(num_tasks):
        final_A = None
        final_b = None
        final_c = None
        final_ct = None
        final_x = None
        final_val = None
        final_steps = ""
        for k in range(max_regen_attempts):
            A_try, b_try, c_try, ct_try = generate_LP_task()
            res_feasible, x_opt, f_val1, logs = solve_LP_two_phase(A_try, b_try, ct_try, c_try)
            if res_feasible:
                final_A = A_try
                final_b = b_try
                final_c = c_try
                final_ct = ct_try   
                final_x = x_opt
                final_val = f_val1
                final_steps = logs
                break
        tasks_info.append({
            "task_number": i + 1,
            "A": final_A,
            "b": final_b,
            "c": final_c,
            "con_types": final_ct,
            "solution_found": True,
            "optimal_value": final_val,
            "solution_vector": final_x,
            "steps": final_steps
        })
    generate_tex(tasks_info, tex_filename=output_file_tex)
    if generate_pdf:
        compile_tex_to_pdf(output_file_tex)
    print(f"Сгенерировано {num_tasks} задач(и). Результаты см. в PDF -> '{output_file_tex}'")

if __name__ == "__main__":
    main_demo(num_tasks=5)
