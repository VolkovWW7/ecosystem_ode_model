#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Повышенная точность калибровки: очистка данных + защита + двухэтапный поиск
"""
import numpy as np
from scipy.optimize import differential_evolution
import mathmodel
import json

# ==================== ДАННЫЕ С УДАЛЕНИЕМ ВЫБРОСОВ ====================
exp_carbon = [  # без изменений
    (0.10, 0.001815, 0.000185), (0.15, 0.002411, 0.000250),
    (0.25, 0.002976, 0.000372), (0.35, 0.002976, 0.000595),
    (0.40, 0.003095, 0.000774), (0.45, 0.003869, 0.000536),
    (0.50, 0.005208, 0.000982), (0.55, 0.004911, 0.000714),
    (0.60, 0.005804, 0.000789), (0.65, 0.008631, 0.000774),
    (0.85, 0.007589, 0.000878), (1.05, 0.007887, 0.000417),
    (1.25, 0.009673, 0.001012),
]

exp_nitrogen_raw = [  # исходные
    (0.019, 0.000216, 0.000728), (0.028, 0.001071, 0.001216),
    (0.045, 0.015476, 0.002857), (0.054, 0.015327, 0.001046),
    (0.063, 0.031994, 0.000890), (0.080, 0.004256, 0.000476),
    (0.098, 0.007946, 0.000402), (0.115, 0.006696, 0.001049),
    (0.133, 0.003423, 0.000536), (0.176, 0.006101, 0.000670),
    (0.194, 0.005119, 0.000564), (0.211, 0.002381, 0.000652),
]

# Исключаем выброс (N0=0.063)
def detect_outliers(data, threshold=3):
    """Удаляет точки, где значение X выходит за threshold сигм."""
    values = [row[1] for row in data]
    mean = np.mean(values)
    std = np.std(values)
    return [row for row in data if abs(row[1] - mean) <= threshold * std]

exp_nitrogen = detect_outliers(exp_nitrogen_raw, threshold=2.5)

C0_FIXED_N_SERIES = 0.625
N0_FIXED_C_SERIES = 0.15

# ==================== ПОДБИРАЕМЫЕ ПАРАМЕТРЫ ====================
param_bounds = {
    'Vx_c': (8.0, 18.0),      # сузили диапазоны, для быстрого поиска
    'Kc': (0.2, 0.8),
    'Kn': (0.02, 0.08),
    'Mxx': (0.005, 0.02),
    'Vy_c': (15.0, 35.0),
    'Kpp': (0.1, 0.4),
    'Kfc': (0.05, 0.2),
    'Myy': (0.02, 0.08),
    'Vm': (3.0, 8.0),
    'k_d1': (1e-5, 0.005),
    'k_d2': (1e-5, 0.005),
    'k_d': (0.3, 1.2),
}

# ==================== ЦЕЛЕВАЯ ФУНКЦИЯ ====================
best_error = float('inf')
eval_count = 0

# Глобальный флаг остановки (устанавливается из GUI)
stop_flag = None

def objective(params_list):
    global best_error, eval_count, stop_flag
    
    # Проверка на остановку
    if stop_flag is not None and stop_flag():  # изменено
        return 1e10
    
    eval_count += 1
    
    p = mathmodel.get_default_params()
    for name, val in zip(param_bounds.keys(), params_list):
        p[name] = val
    
    p['total_time'] = 15000.0
    p['output_step'] = 150
    p = mathmodel.update_dependent_params(p)
    
    errors = []
    eps = 1e-9
    
    # ---- Углеродная серия ----
    p['N0'] = N0_FIXED_C_SERIES
    for C0, x_exp, y_exp in exp_carbon:
        p['C0'] = C0
        try:
            sol = mathmodel.run_simulation(p, method='LSODA')
            X_mod = np.clip(sol.y[0, -1], 0, 1)
            Y_mod = np.clip(sol.y[1, -1], 0, 1)
        except Exception:
            return 1e10
        # Логарифмическая ошибка (устойчива к порядкам величин)
        err_x = (np.log((X_mod + eps) / (x_exp + eps))) ** 2
        err_y = (np.log((Y_mod + eps) / (y_exp + eps))) ** 2
        errors.extend([err_x, err_y])
    
    # ---- Азотная серия (очищенная) ----
    p['C0'] = C0_FIXED_N_SERIES
    for N0, x_exp, y_exp in exp_nitrogen:
        p['N0'] = N0
        try:
            sol = mathmodel.run_simulation(p, method='LSODA')
            X_mod = np.clip(sol.y[0, -1], 0, 1)
            Y_mod = np.clip(sol.y[1, -1], 0, 1)
        except Exception:
            return 1e10
        err_x = (np.log((X_mod + eps) / (x_exp + eps))) ** 2
        err_y = (np.log((Y_mod + eps) / (y_exp + eps))) ** 2
        errors.extend([err_x, err_y])
    
    rmsre = np.sqrt(np.mean(errors))
    if not np.isfinite(rmsre):
        return 1e10
    
    if eval_count % 50 == 0:
        print(f"Eval {eval_count}: RMSRE={rmsre:.6f}, best={best_error:.6f}")
    if rmsre < best_error - 1e-12:
        best_error = rmsre
        print(f"\n>>> NEW BEST (eval {eval_count}): RMSRE={rmsre:.6f}")
        for name, val in zip(param_bounds.keys(), params_list):
            print(f"    {name} = {val:.8f}")
    return rmsre


#Прерывание
def run_calibration_gui():
    """Основная функция калибровки с возможностью остановки."""
    global stop_flag
    
    bounds = list(param_bounds.values())
    
    print("Запуск калибровки параметров модели")
    print(f"Подбирается {len(bounds)} параметров")
    print("-" * 50)
    
    # Используем callback для проверки остановки между поколениями
    def callback_func(xk, convergence):
        if stop_flag and stop_flag():
            print("Остановка калибровки по запросу пользователя...")
            return True  # True означает остановку
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=20, popsize=10,
            seed=42, workers=1, disp=True,
            callback=callback_func,
            polish=False
        )
        
        # ... обработка результата ...
    except Exception as e:
        print(f"Калибровка прервана: {e}")
        return


# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    bounds = list(param_bounds.values())
    print("Запуск ПОВЫШЕННОЙ ТОЧНОСТИ калибровки")
    print(f"Удалён выброс N0=0.063, осталось {len(exp_nitrogen)} точек в азотной серии")
    print("Этап 1: Грубый поиск (быстрый)...")
    
    # Этап 1
    result1 = differential_evolution(
        objective, bounds,
        maxiter=25, popsize=12,
        seed=42, workers=1, disp=True,
        updating='deferred'
    )
    
    if not result1.success:
        print("Грубый поиск не удался, но продолжим...")
    
    # Сужение границ вокруг лучшей точки
    best_x = result1.x
    bounds_tight = []
    for i, (low, high) in enumerate(bounds):
        margin = (high - low) * 0.2
        new_low = max(low, best_x[i] - margin)
        new_high = min(high, best_x[i] + margin)
        bounds_tight.append((new_low, new_high))
    
    print("\nЭтап 2: Точный поиск (суженные границы)...")
    result2 = differential_evolution(
        objective, bounds_tight,
        maxiter=40, popsize=10,
        seed=43, workers=1, disp=True,
        updating='deferred'
    )
    
    final_params = result2.x
    final_rmsre = objective(final_params)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТ КАЛИБРОВКИ ПОВЫШЕННОЙ ТОЧНОСТИ")
    print(f"Финальный RMSRE = {final_rmsre:.6f}")
    if final_rmsre < 0.5:
        print("✓ Хорошая точность (<50% ошибки)")
    elif final_rmsre < 1.0:
        print("✓ Приемлемая точность (<100% ошибки)")
    else:
        print("⚠ Точность низкая, требуется ручная настройка")
    
    print("\nОптимальные параметры:")
    for name, val in zip(param_bounds.keys(), final_params):
        print(f"{name} = {val:.8f}")

    # Сохранение
    opt_params = mathmodel.get_default_params()
    for name, val in zip(param_bounds.keys(), final_params):
        opt_params[name] = val
    opt_params = mathmodel.update_dependent_params(opt_params)
    mathmodel.save_params_to_json(opt_params, "calibrated_params_advanced.json")
    print("\nПараметры сохранены в 'calibrated_params_advanced.json'")