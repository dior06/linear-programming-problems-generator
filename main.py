import os
import subprocess
import numpy as np
from scipy.optimize import linprog

def generate_LP_task(num_vars=2, num_constraints=2, coef_range=(-5, 5)):
    A = np.random.randint(coef_range[0], coef_range[1] + 1,
                          size=(num_constraints, num_vars))
    b = np.random.randint(1, 6, size=num_constraints)
    c = np.random.randint(coef_range[0], coef_range[1] + 1, size=num_vars)
    return A, b, c

def solve_LP_task(A, b, c):
    return linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c), method='highs')

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
            f.write(r"\subsection*{Задача №" + str(i) + "}\n")
            f.write(r"\textbf{Функция цели: }" "\n")
            obj_parts = []
            for idx, coef in enumerate(c):
                sign = "+" if (coef >= 0 and idx > 0) else ""
                obj_parts.append(f"{sign}{coef}x_{{{idx+1}}}")
            f.write("maximize $ " + " ".join(obj_parts) + " $\\\\\n\n")
            f.write(r"\textbf{Ограничения:}" "\n\n")
            f.write(r"\[ \begin{aligned}" "\n")
            for row_idx, row in enumerate(A):
                row_expr = []
                for var_idx, val in enumerate(row):
                    sign = "+" if (val >= 0 and var_idx > 0) else ""
                    row_expr.append(f"{sign}{val}x_{{{var_idx+1}}}")
                f.write(" ".join(row_expr) + f" &\\le {b[row_idx]} \\\\ \n")
            f.write("x_i &\\ge 0, \\quad i=1,\\dots," + str(len(c)) + " \\\\ \n")
            f.write(r"\end{aligned}\]" "\n\n")
            
            if info["solution_found"]:
                val = info["optimal_value"]
                sol_str = ", ".join([f"x_{{{j+1}}}={valj:.2f}" 
                                     for j, valj in enumerate(info["solution_vector"])])
                f.write(r"\textbf{Оптимальное значение: }" + f"${val:.2f}$\n\n")
                f.write(r"\textbf{Решение: }" + f"${sol_str}$\n\n")
            else:
                f.write(r"\textbf{Задача не имеет решения или возникла ошибка.}" "\n\n")
            f.write("\n\n")
        
        f.write(r"\end{document}")

def compile_tex_to_pdf(tex_filename):
    try:
        subprocess.run(["pdflatex", tex_filename], check=True)
        print(f"PDF-файл сформирован из {tex_filename}.")
    except FileNotFoundError:
        print("Команда pdflatex не найдена. Установите LaTeX или добавьте pdflatex в PATH.")
    except subprocess.CalledProcessError:
        print("Ошибка при компиляции LaTeX-файла.")

def main_demo(num_tasks=3,
              output_file_txt="generated_tasks_and_solutions.txt",
              output_file_tex="tasks_solutions.tex",
              generate_pdf=True):
    tasks_info = []
    
    with open(output_file_txt, "w", encoding="utf-8") as f_out:
        for i in range(num_tasks):
            A, b, c = generate_LP_task()
            res = solve_LP_task(A, b, c)
            f_out.write(f"--- Задача №{i+1} ---\n")
            f_out.write("Maximize: ")
            f_out.write(" + ".join([f"{c[j]}*x{j+1}" for j in range(len(c))]))
            f_out.write("\nSubject to:\n")
            
            for row_idx, row in enumerate(A):
                lhs_str = " + ".join([f"{row[j]}*x{j+1}" for j in range(len(row))])
                f_out.write(f"   {lhs_str} <= {b[row_idx]}\n")
            f_out.write("   x >= 0\n\n")
            
            found = False
            opt_val = None
            solution_vec = None
            
            if res.success:
                opt_val = -res.fun
                solution_vec = res.x
                found = True
                f_out.write(f"Оптимальное значение (max c^T x): {opt_val:.4f}\n")
                f_out.write("Оптимальное решение (x*): [")
                f_out.write(", ".join([f"{val:.4f}" for val in solution_vec]))
                f_out.write("]\n\n")
            else:
                f_out.write("Задача не имеет решения (infeasible / unbounded) "
                            f"или возникла ошибка: {res.message}\n\n")
            
            tasks_info.append({
                "task_number": i+1,
                "A": A,
                "b": b,
                "c": c,
                "solution_found": found,
                "optimal_value": opt_val,
                "solution_vector": solution_vec
            })
    
    generate_tex(tasks_info, tex_filename=output_file_tex)
    if generate_pdf:
        compile_tex_to_pdf(output_file_tex)

    print(f"Сгенерировано {num_tasks} задач. Результаты: {output_file_txt}")
    if generate_pdf:
        print(f"Пытаемся создать PDF из {output_file_tex} (при наличии pdflatex).")

if __name__ == "__main__":
    main_demo()
