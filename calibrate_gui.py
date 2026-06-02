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
# Переменная для хранения лучших метрик (для логов)
best_metrics = {"RMSRE": 1e10, "RMSE": 1e10, "NRMSE": 1e10}


def objective(params_list, metric_type='nrmse'):
    """
    metric_type может быть: 'rmsre', 'rmse', 'nrmse'
    """
    global best_error, eval_count, stop_flag, best_metrics
    
    if stop_flag is not None and stop_flag():
        return 1e10
    eval_count += 1
    
    p = mathmodel.get_default_params()
    for name, val in zip(param_bounds.keys(), params_list):
        p[name] = val
    
    p['total_time'] = 15000
    p['output_step'] = 150
    p = mathmodel.update_dependent_params(p)
    
    # Списки для накопления ошибок разных типов
    sq_rel_errors = []  # Для RMSRE
    sq_abs_errors = []  # Для RMSE
    
    # Списки для поиска максимумов (нужны для NRMSE)
    x_exp_all = []
    y_exp_all = []
    
    eps = 1e-9
    
    # --- Сбор данных по серии Carbon ---
    p['N0'] = N0_FIXED_C_SERIES
    for C0, x_exp, y_exp in exp_carbon:
        p['C0'] = C0
        x_exp_all.append(x_exp)
        y_exp_all.append(y_exp)
        try:
            sol = mathmodel.run_simulation(p, method='LSODA')
            X_mod = np.clip(sol.y[0, -1], 0, 1)
            Y_mod = np.clip(sol.y[1, -1], 0, 1)
        except Exception:
            return 1e10
            
        sq_abs_errors.extend([(X_mod - x_exp)**2, (Y_mod - y_exp)**2])
        sq_rel_errors.extend([((X_mod - x_exp)/(x_exp + eps))**2, ((Y_mod - y_exp)/(y_exp + eps))**2])
    
    # --- Сбор данных по серии Nitrogen ---
    p['C0'] = C0_FIXED_N_SERIES
    for N0, x_exp, y_exp in exp_nitrogen:
        p['N0'] = N0
        x_exp_all.append(x_exp)
        y_exp_all.append(y_exp)
        try:
            sol = mathmodel.run_simulation(p, method='LSODA')
            X_mod = np.clip(sol.y[0, -1], 0, 1)
            Y_mod = np.clip(sol.y[1, -1], 0, 1)
        except Exception:
            return 1e10
            
        sq_abs_errors.extend([(X_mod - x_exp)**2, (Y_mod - y_exp)**2])
        sq_rel_errors.extend([((X_mod - x_exp)/(x_exp + eps))**2, ((Y_mod - y_exp)/(y_exp + eps))**2])
    
    # --- РАСЧЕТ ВСЕХ МЕТРИК ---
    rmse = np.sqrt(np.mean(sq_abs_errors))
    rmsre = np.sqrt(np.mean(sq_rel_errors))
    
    # Нормализуем по размаху экспериментальных данных (NRMSE)
    max_exp = max(max(x_exp_all), max(y_exp_all))
    min_exp = min(min(x_exp_all), min(y_exp_all))
    range_exp = (max_exp - min_exp) if (max_exp - min_exp) > 0 else 1.0
    nrmse = rmse / range_exp

    # Словарь для маппинга
    metrics_map = {
        'rmse': rmse,
        'rmsre': rmsre,
        'nrmse': nrmse
    }
    
    current_target_value = metrics_map[metric_type]
    
    # Логирование каждые 50 итераций
    if eval_count % 50 == 0:
        print(f"Оценка {eval_count} [{metric_type.upper()}]: NRMSE={nrmse:.4f} | RMSRE={rmsre:.4f} | RMSE={rmse:.4f}", flush=True)
    
    # Фиксация рекорда по целевой метрике
    if current_target_value < best_error - 1e-12:
        best_error = current_target_value
        best_metrics = {"RMSRE": rmsre, "RMSE": rmse, "NRMSE": nrmse}
        print(f">>> НОВЫЙ ОПТИМУМ: NRMSE={nrmse:.4f}, RMSRE={rmsre:.4f}, RMSE={rmse:.4f}", flush=True)  
        
    return current_target_value

def run_calibration_gui():
    global stop_flag, best_error, eval_count
    
    bounds = list(param_bounds.values())
    param_names = list(param_bounds.keys())
    
    print("="*60)
    print("ЗАПУСК МНОГОСТУПЕНЧАТОЙ КАЛИБРОВКИ МОДЕЛИ")
    print("="*60)
    
    def callback_func(xk, convergence):
        if stop_flag is not None and stop_flag():
            print("Остановка пользователем...")
            return True

    # -----------------------------------------------------------------
    # СТУПЕНЬ 1: Глобальный поиск по NRMSE (избавляемся от плато)
    # -----------------------------------------------------------------
    print("\n[СТУПЕНЬ 1/2] Глобальный поиск (Оптимизация тренда по NRMSE)...")
    best_error = 1e10
    eval_count = 0
    
    result_stage1 = differential_evolution(
        objective, bounds,
        args=('nrmse',),  # Передаем выбор метрики в objective
        maxiter=25,       # Небольшое число итераций для грубого поиска
        popsize=12,
        seed=42, workers=1, disp=False,
        callback=callback_func
    )
    
    if stop_flag is not None and stop_flag():
        return None

    print(f"-> Ступень 1 завершена. Лучшая NRMSE: {result_stage1.fun:.4f}")
    stage1_params = result_stage1.x

    # -----------------------------------------------------------------
    # СТУПЕНЬ 2: Локальное уточнение по RMSRE (Тонкая настройка)
    # -----------------------------------------------------------------
    print("\n[СТУПЕНЬ 2/2] Тонкая подстройка (Локальный спуск по RMSRE)...")
    best_error = 1e10
    
    # Динамически сужаем границы вокруг параметров, найденных на Шаге 1 (на +/- 20%)
    narrowed_bounds = []
    for name, val in zip(param_names, stage1_params):
        orig_low, orig_high = param_bounds[name]
        low = max(orig_low, val * 0.8)
        high = min(orig_high, val * 1.2)
        narrowed_bounds.append((low, high))
        print(f" Сужены границы {name}: [{low:.4f} ... {high:.4f}]")

    # Для Ступени 2 мы можем использовать либо повторный DE с узкими границами,
    # либо быстрый градиентный метод "L-BFGS-B" от лучшей точки
    result_stage2 = differential_evolution(
        objective, narrowed_bounds,
        args=('rmsre',),  # Теперь заставляем минимизировать строго относительную ошибку
        maxiter=20,
        popsize=10,
        polish=True,      # Включаем финальную математическую полировку
        seed=42, workers=1, disp=False,
        callback=callback_func
    )

    # -----------------------------------------------------------------
    # ОБРАБОТКА И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    if result_stage2.fun < 1e9:
        print("КАЛИБРОВКА УСПЕШНО ЗАВЕРШЕНА")
        print(f"Итоговые метрики:")
        print(f"  - RMSRE (Относительная): {best_metrics['RMSRE']:.6f}")
        print(f"  - NRMSE (Нормированная): {best_metrics['NRMSE']:.6f}")
        print(f"  - RMSE  (Абсолютная):    {best_metrics['RMSE']:.6f}")
        
        print("\nОптимальные параметры:")
        for name, val in zip(param_names, result_stage2.x):
            print(f"  {name} = {val:.8f}")
            
        # Формируем итоговый словарь параметров
        opt_params = mathmodel.get_default_params()
        for name, val in zip(param_names, result_stage2.x):
            opt_params[name] = val
        opt_params = mathmodel.update_dependent_params(opt_params)
        
        filename = (
            f"calibrated_params_"
            f"NRMSE_{best_metrics['NRMSE']:.4f}_"
            f"RMSRE_{best_metrics['RMSRE']:.4f}_"
            f"RMSE_{best_metrics['RMSE']:.4f}.json"
        )
        
        return filename
    else:
        print("Ошибка: Не удалось найти стабильное решение.")
        return None

if __name__ == "__main__":
    run_calibration_gui()