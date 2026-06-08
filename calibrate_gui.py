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
from matplotlib.figure import Figure

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
    'Ax': (0.1, 5.0),         # Коэффициент влияния среды на смертность (сейчас 1.0)
    'Ki': (0.0, 0.2),         # Константа ингибирования избытком азота (сейчас 0.0)
    
    # --- КИНЕТИКА ГЕТЕРОТРОФОВ (Бактерии) ---
    'Vy_c': (15.0, 35.0),     # Макс. скорость роста Y (сейчас 25.0)
    'Myy': (0.005, 0.08),     # Коэффициент смертности гетеротрофов (сейчас 0.03)
    'Kpp': (0.05, 0.5),       # Полунасыщение по Белкам для Y (сейчас 0.2)
    'Kfc': (0.02, 0.3),       # Полунасыщение по Жирам/Углеводам для Y (сейчас 0.1)
    'Ka': (0.05, 0.5),        # Полунасыщение по АТФ для Y (сейчас 0.2)
    'Ay': (0.1, 5.0),         # Влияние АТФ на смертность гетеротрофов (сейчас 1.0)
    
    # --- ПОТРЕБЛЕНИЕ И ФЕРМЕНТАЦИЯ ДЕТРИТА ---
    'Vm': (1.0, 10.0),        # Макс. скорость расщепления ферментов (сейчас 5.0)
    'Kp': (0.1, 5.0),         # Полунасыщение по белкам при потреблении (сейчас 1.0)
    'Kf': (0.1, 5.0),         # Полунасыщение по жирам при потреблении (сейчас 1.0)
    'Kch': (0.1, 1.0),        # Полунасыщение по углеводам при потреблении (сейчас 0.5)
    
    # --- РАЗЛОЖЕНИЕ ДЕТРИТА И ЭФФЕКТИВНОСТЬ ---
    'k_d1': (1e-5, 0.005),    # Скорость распада детрита хлореллы (сейчас 0.0007)
    'k_d2': (1e-5, 0.005),    # Скорость распада детрита бактерий (сейчас 0.0007)
    'k_d': (0.1, 1.0),        # Коэффициент автолиза/синтеза детрита (сейчас 0.5)
    'eps2': (1.0, 10.0),      # Энергетический выход на единицу субстрата (сейчас 5.0)
    
    # --- РАСПРЕДЕЛЕНИЕ ЭНЕРГИИ АТФ (Сумма не критична, это веса) ---
    'aP': (0.1, 0.6),         # Доля энергии на синтез белков (сейчас 0.3333)
    'aF': (0.1, 0.6),         # Доля энергии на синтез жиров (сейчас 0.3333)
    'aCh': (0.1, 0.6),        # Доля энергии на синтез углеводов (сейчас 0.3333)

    # --- СТЕХИОМЕТРИЯ БИОМАССЫ (Определяет bXCh и bYCh) ---
    # Границы зажаты так, чтобы сумма bXP + bXF не превысила 1.0 (иначе доля углеводов станет отрицательной)
    'bXP': (0.50, 0.65),      # Доля белков в биомассе автотрофов (сейчас 0.58)
    'bXF': (0.15, 0.25),      # Доля жиров в биомассе автотрофов (сейчас 0.22)
    'bYP': (0.50, 0.70),      # Доля белков в биомассе гетеротрофов (сейчас 0.60)
    'bYF': (0.05, 0.15),      # Доля жиров в биомассе гетеротрофов (сейчас 0.10)
    
    # --- ЭЛЕМЕНТНЫЙ СОСТАВ КОМПОНЕНТОВ (Определяет gXC, gXN, gYC, gYN) ---
    'gPC': (0.50, 0.90),      # Содержание углерода в белках (базовое: 0.77)
    'gPN': (0.10, 0.30),      # Содержание азота в белках (базовое: 0.23)
    'gFC': (0.50, 1.00),      # Содержание углерода в жирах (базовое: 1.0)
    'gChC': (0.40, 1.00),     # Содержание углерода в углеводах (базовое: 1.0)
}

best_error = float('inf')
eval_count = 0
stop_flag = None
# Переменная для хранения лучших метрик (для логов)
best_metrics = {"RMSRE": 1e10, "RMSE": 1e10, "NRMSE": 1e10}


def objective(params_list, metric_type='nrmse', kinetic_model='mitscherlich'):
    """
    metric_type может быть: 'rmsre', 'rmse', 'nrmse'
    """
    global best_error, eval_count, stop_flag, best_metrics
    
    if stop_flag is not None and stop_flag():
        return 1e10
    eval_count += 1
    
    p = mathmodel.get_default_params()
    p['kinetic_model'] = kinetic_model  # ИСПРАВЛЕНИЕ: Передаем тип кинетики в модель
    
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

        # ЖЕСТКИЙ ШТРАФ ЗА ВЫМИРАНИЕ (Защита от читерства алгоритма)
        # Если модель уводит популяцию в ноль, возвращаем огромную ошибку
        if X_mod < 1e-4 or Y_mod < 1e-4:
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
            
        # ЖЕСТКИЙ ШТРАФ ЗА ВЫМИРАНИЕ (Защита от читерства алгоритма)
        # Если модель уводит популяцию в ноль, возвращаем огромную ошибку
        if X_mod < 1e-4 or Y_mod < 1e-4:
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

def run_calibration_gui(kinetic_model='mitscherlich'):
    global stop_flag, best_error, eval_count
    
    # 1. Исправляем и очищаем входящую переменную от tuple
    if isinstance(kinetic_model, tuple):
        # Если прилетел кортеж, берем его первый элемент (обычно это строка)
        kinetic_model = kinetic_model[0]
    
    # 2. Приводим к стандартной строке без пробелов в нижнем регистре
    kinetic_model_str = str(kinetic_model).strip().lower()
    
    # 3. Принудительно сопоставляем с внутренними идентификаторами моделей ядра
    if "liebig" in kinetic_model_str or "либих" in kinetic_model_str:
        model_label = "Liebig"
    else:
        model_label = "Mitscherlich"

    bounds = list(param_bounds.values())
    param_names = list(param_bounds.keys())
    
    print("="*60)
    print(f"ЗАПУСК МНОГОСТУПЕНЧАТОЙ КАЛИБРОВКИ ДЛЯ МОДЕЛИ: {kinetic_model.upper()}")
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
        args=('nrmse', kinetic_model),  # передаем kinetic_model в args
        maxiter=25,       # Небольшое число итераций для грубого поиска
        popsize=10,
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

    result_stage2 = differential_evolution(
        objective, narrowed_bounds,
        args=('rmsre', kinetic_model),  # передаем kinetic_model в args
        maxiter=25,
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
        opt_params['kinetic_model'] = kinetic_model  #сохраняем выбранную модель в файл конфигурации
        
        for name, val in zip(param_names, result_stage2.x):
            opt_params[name] = val
        opt_params = mathmodel.update_dependent_params(opt_params)
        
        filename = (
            f"{model_label}",
            f"NRMSE_{best_metrics['NRMSE']:.4f}_"
            f"RMSRE_{best_metrics['RMSRE']:.4f}_"
            f"RMSE_{best_metrics['RMSE']:.4f}.json"
        )

        mathmodel.save_params_to_json(opt_params, filename)
        
        return filename
    else:
        print("Ошибка: Не удалось найти стабильное решение.")
        return None

def graph_validation(p):
    """
    Строит графики верификации: сравнение финальных точек модели 
    с экспериментальными данными из текущего модуля.
    """
    fig = Figure(figsize=(12, 5))
    
    # --- ЛЕВЫЙ ГРАФИК: СЕРИЯ ПО УГЛЕРОДУ ---
    ax1 = fig.add_subplot(121)
    p_c = p.copy()
    p_c['N0'] = N0_FIXED_C_SERIES
    p_c['total_time'] = p.get('total_time', 500)
    p_c['output_step'] = p_c['total_time'] / 100
    
    # Строим плавную теоретическую кривую модели
    c_vals = [row[0] for row in exp_carbon]
    c_grid = np.linspace(min(c_vals), max(c_vals), 100)
    x_mod_c, y_mod_c = [], []
    
    for c0 in c_grid:
        p_c['C0'] = c0
        #  Обновляем внутренние зависимости от нового C0
        p_c = mathmodel.update_dependent_params(p_c)
        sol = mathmodel.run_simulation(p_c, method='LSODA')
        x_mod_c.append(np.clip(sol.y[0, -1], 0, 1))
        y_mod_c.append(np.clip(sol.y[1, -1], 0, 1))
        
    ax1.plot(c_grid, x_mod_c, 'g-', lw=2, label='Модель: Автотрофы (X)')
    ax1.plot(c_grid, y_mod_c, 'r-', lw=2, label='Модель: Гетеротрофы (Y)')
    
    # Наносим точки реального эксперимента
    exp_c_arr = np.array(exp_carbon)
    ax1.scatter(exp_c_arr[:, 0], exp_c_arr[:, 1], color='darkgreen', edgecolors='black', s=40, label='Эксп. точки X')
    ax1.scatter(exp_c_arr[:, 0], exp_c_arr[:, 2], color='darkred', marker='x', s=50, lw=2, label='Эксп. точки Y')
    
    ax1.set_title('Серия по углероду (при фиксированном N0 = 0.15)')
    ax1.set_xlabel('Начальный минеральный углерод (C0)')
    ax1.set_ylabel('Концентрация в t = 15000')
    ax1.grid(True)
    ax1.legend()

    # --- ПРАВЫЙ ГРАФИК: СЕРИЯ ПО АЗОТУ ---
    ax2 = fig.add_subplot(122)
    p_n = p.copy()
    p_n['C0'] = C0_FIXED_N_SERIES
    p_n['total_time'] = p.get('total_time', 500)
    p_n['output_step'] = p_n['total_time'] / 100
    
    # Строим плавную теоретическую кривую модели
    n_vals = [row[0] for row in exp_nitrogen]
    n_grid = np.linspace(min(n_vals), max(n_vals), 100)
    x_mod_n, y_mod_n = [], []
    
    for n0 in n_grid:
        p_n['N0'] = n0
        # Обновляем внутренние зависимости от нового N0
        p_n = mathmodel.update_dependent_params(p_n)
        sol = mathmodel.run_simulation(p_n, method='LSODA')
        x_mod_n.append(np.clip(sol.y[0, -1], 0, 1))
        y_mod_n.append(np.clip(sol.y[1, -1], 0, 1))
        
    ax2.plot(n_grid, x_mod_n, 'g-', lw=2, label='Модель: Автотрофы (X)')
    ax2.plot(n_grid, y_mod_n, 'r-', lw=2, label='Модель: Гетеротрофы (Y)')
    
    # Наносим точки реального эксперимента
    exp_n_arr = np.array(exp_nitrogen)
    ax2.scatter(exp_n_arr[:, 0], exp_n_arr[:, 1], color='darkgreen', edgecolors='black', s=40, label='Эксп. точки X')
    ax2.scatter(exp_n_arr[:, 0], exp_n_arr[:, 2], color='darkred', marker='x', s=50, lw=2, label='Эксп. точки Y')
    
    ax2.set_title('Серия по азоту (при фиксированном C0 = 0.625)')
    ax2.set_xlabel('Начальный минеральный азот (N0)')
    ax2.set_ylabel('Концентрация в t = 15000')
    ax2.grid(True)
    ax2.legend()
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    run_calibration_gui()
