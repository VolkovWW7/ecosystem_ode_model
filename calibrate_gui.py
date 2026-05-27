#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Версия калибровки для запуска из GUI с перенаправлением вывода.
"""
import numpy as np
from scipy.optimize import differential_evolution
import mathmodel
import json
import sys

# ==================== ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ ====================
exp_carbon = [
    (0.10, 0.001815, 0.000185), (0.15, 0.002411, 0.000250),
    (0.25, 0.002976, 0.000372), (0.35, 0.002976, 0.000595),
    (0.40, 0.003095, 0.000774), (0.45, 0.003869, 0.000536),
    (0.50, 0.005208, 0.000982), (0.55, 0.004911, 0.000714),
    (0.60, 0.005804, 0.000789), (0.65, 0.008631, 0.000774),
    (0.85, 0.007589, 0.000878), (1.05, 0.007887, 0.000417),
    (1.25, 0.009673, 0.001012),
]

exp_nitrogen_raw = [
    (0.019, 0.000216, 0.000728), (0.028, 0.001071, 0.001216),
    (0.045, 0.015476, 0.002857), (0.054, 0.015327, 0.001046),
    (0.063, 0.031994, 0.000890), (0.080, 0.004256, 0.000476),
    (0.098, 0.007946, 0.000402), (0.115, 0.006696, 0.001049),
    (0.133, 0.003423, 0.000536), (0.176, 0.006101, 0.000670),
    (0.194, 0.005119, 0.000564), (0.211, 0.002381, 0.000652),
]

exp_nitrogen = [row for row in exp_nitrogen_raw if row[0] != 0.063]

C0_FIXED_N_SERIES = 0.625
N0_FIXED_C_SERIES = 0.15

#Массив подбираемых параметров, в скобочках указаны пределы подбора
param_bounds = {
# --- КИНЕТИКА АВТОТРОФОВ (Хлорелла) ---
    'Vx_c': (8.0, 18.0),      # Макс. скорость роста Х (сейчас 13.19)
    'Kc': (0.1, 1.5),         # Константа полунасыщения по углероду С (сейчас 0.81)
    'Kn': (0.005, 0.1),       # Константа полунасыщения по азоту N (сейчас 0.02)
    'Mxx': (0.005, 0.04),     # Коэффициент смертности автотрофов (сейчас 0.019)
    
    # --- КИНЕТИКА ГЕТЕРОТРОФОВ (Бактерии) ---
    'Vy_c': (15.0, 35.0),     # Макс. скорость роста Y (сейчас 25.0)
    'Myy': (0.005, 0.08),     # Коэффициент смертности гетеротрофов (сейчас 0.03)
    'Kpp': (0.05, 0.5),       # Полунасыщение по Белкам для Y (сейчас 0.2)
    'Kfc': (0.02, 0.3),       # Полунасыщение по Жирам/Углеводам для Y (сейчас 0.1)
    'Ka': (0.05, 0.5),        # Полунасыщение по АТФ для Y (сейчас 0.2)
    
    # --- РАЗЛОЖЕНИЕ ДЕТРИТА ---
    'k_d1': (1e-5, 0.005),    # Скорость распада детрита хлореллы (сейчас 0.0007)
    'k_d2': (1e-5, 0.005),    # Скорость распада детрита бактерий (сейчас 0.0007)

    #Второстепенная кинетика (Калибровать с осторожностью, раскомментировать в случае недостаточной точности)
    #'Vm': (1.0, 10.0),
    #'k_d': (0.1, 1.0),
    #'eps2': (1.0, 10.0),
    #'Kch': (0.1, 1.0),
}

best_error = float('inf')
eval_count = 0
stop_flag = None

def objective(params_list):
    global best_error, eval_count, stop_flag
    
    if stop_flag is not None and stop_flag():
        return 1e10
    eval_count += 1
    
    p = mathmodel.get_default_params()
    for name, val in zip(param_bounds.keys(), params_list):
        p[name] = val
    
    p['total_time'] = 15000
    p['output_step'] = 150
    p = mathmodel.update_dependent_params(p)
    
    errors = []
    eps = 1e-9
    
    p['N0'] = N0_FIXED_C_SERIES
    for C0, x_exp, y_exp in exp_carbon:
        p['C0'] = C0
        try:
            sol = mathmodel.run_simulation(p, method='LSODA')
            X_mod = np.clip(sol.y[0, -1], 0, 1)
            Y_mod = np.clip(sol.y[1, -1], 0, 1)
        except Exception:
            return 1e10
        err_x = ((X_mod - x_exp) / (x_exp + eps)) ** 2
        err_y = ((Y_mod - y_exp) / (y_exp + eps)) ** 2
        errors.extend([err_x, err_y])
    
    p['C0'] = C0_FIXED_N_SERIES
    for N0, x_exp, y_exp in exp_nitrogen:
        p['N0'] = N0
        try:
            sol = mathmodel.run_simulation(p, method='LSODA')
            X_mod = np.clip(sol.y[0, -1], 0, 1)
            Y_mod = np.clip(sol.y[1, -1], 0, 1)
        except Exception:
            return 1e10
        err_x = ((X_mod - x_exp) / (x_exp + eps)) ** 2
        err_y = ((Y_mod - y_exp) / (y_exp + eps)) ** 2
        errors.extend([err_x, err_y])
    
    rmsre = np.sqrt(np.mean(errors))
    
    if eval_count % 20 == 0:
        print(f"Оценка {eval_count}: RMSRE = {rmsre:.6f}", flush=True)
    if rmsre < best_error - 1e-12:
        best_error = rmsre
        print(f">>> НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ: RMSRE={rmsre:.6f}", flush=True)
    return rmsre

def run_calibration_gui():
    """Основная функция калибровки."""
    global stop_flag
    bounds = list(param_bounds.values())
    
    print("Запуск калибровки параметров модели")
    print(f"Подбирается {len(bounds)} параметров")
    print("-" * 50)
    
    def callback_func(xk, convergence):
        if stop_flag is not None and stop_flag():
            print("Остановка калибровки...")
            return True
    
    #Алгоритм дифференциальной эволюции
    result = differential_evolution(
        objective, bounds,
        maxiter=40, 
        popsize=15,
        seed=42, 
        workers=1, 
        disp=True,
        callback=callback_func
    )
    
    print("\n" + "="*50) 
    if result.success:
        print("ОПТИМИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА")
        print(f"Лучшее RMSRE: {result.fun:.6f}")
        print("Оптимальные параметры:")
        for name, val in zip(param_bounds.keys(), result.x):
            print(f" {name} = {val:.8f}")
            
        opt_params = mathmodel.get_default_params()
        for name, val in zip(param_bounds.keys(), result.x):
            opt_params[name] = val
        opt_params = mathmodel.update_dependent_params(opt_params)
        
        # === ИЗМЕНЕНИЕ: Формируем динамическое имя файла ===
        # Ограничим до 6 знаков после запятой, чтобы имя файла не было слишком длинным
        filename = f"calibrated_params_RMSRE_{result.fun:.6f}.json"
        
        mathmodel.save_params_to_json(opt_params, filename)
        print(f"\nПараметры сохранены в '{filename}'")
        
        return filename  # Возвращаем имя файла для GUI
    else:
        print("Ошибка оптимизации:", result.message)
        return None

if __name__ == "__main__":
    run_calibration_gui()