#Математическая модель микробиоты автотрофов и гететрофов
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import json

# --- МАТЕМАТИЧЕСКАЯ МОДЕЛЬ ---

def get_default_params():
    """Возвращает базовый словарь параметров из спецификации модели."""
    p = {
        'kinetic_model': 'mitscherlich', # По умолчанию мультипликативная
        # Автотрофы
        'Vx_c': 13.19696,    # было 10.0
        'Kc': 0.815067,      # было 0.3
        'Kn': 0.020917,      # было 0.03
        'Ki': 0.0,
        'Mxx': 0.01938,      # было 0.005
        'Ax': 1.0,
        # Гетеротрофы
        'Vy_c': 25.0, 'Kpp': 0.2, 'Kfc': 0.1, 'Ka': 0.2, 'Myy': 0.03, 'Ay': 1.0,
        # Потребление
        'Vm': 5.0, 'Kp': 1.0, 'Kf': 1.0, 'Kch': 0.5,
        # Детрит и АТФ
        'k_d1': 0.0007, 'k_d2': 0.0007, 'k_d': 0.5, 'eps2': 5.0,
        'aP': 0.3333, 'aF': 0.3333, 'aCh': 0.3333,
        # Стехиометрия биомассы
        'bXP': 0.58, 'bXF': 0.22, 'bYP': 0.60, 'bYF': 0.10,
        # Элементный состав
        'gPC': 0.77, 'gPN': 0.23, 'gFC': 1.0, 'gChC': 1.0,
        # Среда и начальные условия
        'C0': 0.11, 'N0': 0.15,
        'X0': 0.0024, 
        'Y0': 0.00025, 
        'Dcl0': 0.04, 
        'Dps0': 0.04, 
        'Ac0': 0.9,
        # Время
        'total_time': 20000.0, 'output_step': 10.0
    }
    
    # Производные коэффициенты
    p['bXCh'] = 1.0 - p['bXP'] - p['bXF']
    p['bYCh'] = 1.0 - p['bYP'] - p['bYF']
    p['gXC'] = p['bXP'] * p['gPC'] + p['bXF'] * p['gFC'] + p['bXCh'] * p['gChC']
    p['gXN'] = p['bXP'] * p['gPN']
    p['gYC'] = p['bYP'] * p['gPC'] + p['bYF'] * p['gFC'] + p['bYCh'] * p['gChC']
    p['gYN'] = p['bYP'] * p['gPN']
    
    return p

def update_dependent_params(p):
    """Пересчитывает зависимые коэффициенты на основе базовых."""
    p['bXCh'] = 1.0 - p['bXP'] - p['bXF']
    p['bYCh'] = 1.0 - p['bYP'] - p['bYF']
    p['gXC'] = p['bXP']*p['gPC'] + p['bXF']*p['gFC'] + p['bXCh']*p['gChC']
    p['gXN'] = p['bXP']*p['gPN']
    p['gYC'] = p['bYP']*p['gPC'] + p['bYF']*p['gFC'] + p['bYCh']*p['gChC']
    p['gYN'] = p['bYP']*p['gPN']
    return p

def sys(t, x, p):
    """
    Функция дифференциальных уравнений. t- время интегрирования, x =[ X,Y,Dcl,Dps,P,F,Ch,Ac], p - словарь параметров системы
    """
    #распаковка основных параметров
    X, Y, Dcl, Dps, P, F, Ch, Ac = x

    # Ограничение параметров. не позволяет использовать отрицательные значения
    X = max(0.0, X)
    Y = max(0.0, Y)
    Dcl = max(0.0, Dcl)
    Dps = max(0.0, Dps)
    P = max(0.0, P)
    F = max(0.0, F)
    Ch = max(0.0, Ch)
    Ac = np.clip(Ac, 0.0, 1.0)

    C = max(1e-9, p['C0'] - (p['gXC']*(X + Dcl) + p['gYC']*(Y + Dps) + p['gPC']*P + p['gFC']*F + p['gChC']*Ch))
    N = max(1e-9, p['N0'] - (p['gXN']*(X + Dcl) + p['gYN']*(Y + Dps) + p['gPN']*P))

    #Ph_c = p['Vx_c'] * C * N / ((p['Kc'] + C) * (p['Kn'] + N + p['Ki'] * N**4)) #модель митчерлиха
    # Вычисляем отдельные компоненты насыщения для Углерода и Азота
    f_C = C / (p['Kc'] + C)
    f_N = N / (p['Kn'] + N + p['Ki'] * N**4)

    # Переключение логики: Митчерлих (умножение) или Либих (минимум)
    if p.get('kinetic_model', 'mitscherlich') == 'liebig':
        Ph_c = p['Vx_c'] * min(f_C, f_N)
    else:
        Ph_c = p['Vx_c'] * f_C * f_N
    
    mX_c = p['Mxx'] * (1 + C*N + p['Ki'] * N**4) / (1 + p['Ax'] * C * N) #смертность

    v_atp = Ac * (2*(1 - Ac) - 0.1) / ((1 - Ac) + 0.1)
    atp_val = max(0.0, v_atp)

    f_c = p['Vy_c'] * P * F * Ch / ((p['Kpp'] + P) * (p['Kfc'] + F) * (p['Kfc'] + Ch)) * Ac / (p['Ka'] + Ac)

    gP_c = p['Vm'] * P / (p['Kp'] + P)
    gF_c = p['Vm'] * F / (p['Kf'] + F)
    gCh_c = p['Vm'] * Ch / (p['Kch'] + Ch)
    mY_c = p['Myy'] * (0.01 + Ac) / (0.01 + p['Ay'] * Ac)

    dx = [0.0]*8
    dx[0] = (Ph_c - mX_c) * X
    dx[1] = (f_c - mY_c) * Y
    dx[2] = mX_c * X - p['k_d1'] * Dcl
    dx[3] = mY_c * Y - p['k_d2'] * Dps
    dx[4] = (p['bXP']*p['k_d1']*Dcl + p['bYP']*p['k_d2']*Dps - (p['bYP']*f_c + p['aP']*gP_c*atp_val)*Y)
    dx[5] = (p['bXF']*p['k_d1']*Dcl + p['bYF']*p['k_d2']*Dps - (p['bYF']*f_c + p['aF']*gF_c*atp_val)*Y)
    dx[6] = (p['bXCh']*p['k_d1']*Dcl + p['bYCh']*p['k_d2']*Dps - (p['bYCh']*f_c + p['aCh']*gCh_c*atp_val)*Y)
    dx[7] = 10 * (5*atp_val*(p['aP']*gP_c + p['aF']*gF_c + p['aCh']*gCh_c)*Y/(0.0001+Y) - f_c - p['k_d']*Ac/(p['eps2']+Ac))
    return dx

def run_simulation(p,method):
    p0_val = (p['Dcl0'] + p['Dps0']) / 6
    x0 = [p['X0'], p['Y0'], p['Dcl0'], p['Dps0'], p0_val, p0_val, p0_val, p['Ac0']]
    t_span = (0, p['total_time'])
    t_eval = np.arange(0, p['total_time'] + p['output_step'], p['output_step'])
    return solve_ivp(sys, t_span, x0, args=(p,), method=method, t_eval=t_eval)

# --- ФУНКЦИИ ГРАФИКОВ ---
def _setup_ax(ax, title, xlabel='Время', ylabel='Концентрация'):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

def graph_Trofs(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[0], 'g-', lw=2, label='Автотрофы (X)')
    ax.plot(sol.t, sol.y[1], 'r-', lw=2, label='Гетеротрофы (Y)')
    _setup_ax(ax, 'Динамика автотрофов и гетеротрофов')
    return fig

def graph_Detrit(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[2], 'b-', label='Детрит хлореллы')
    ax.plot(sol.t, sol.y[3], 'm-', label='Детрит бактерий')
    _setup_ax(ax, 'Концентрация детрита')
    return fig

def graph_PFC(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[4], label='Белки (P)')
    ax.plot(sol.t, sol.y[5], label='Жиры (F)')
    ax.plot(sol.t, sol.y[6], label='Углеводы (Ch)')
    _setup_ax(ax, 'Концентрация БЖУ')
    return fig

def graph_Ac(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[7], 'k-', label='АТФ (Ac)')
    _setup_ax(ax, 'Внутриклеточная концентрация АТФ')
    return fig

def graph_minerals(sol, params):
    """График динамики минеральных веществ (Углерод и Азот) на раздельных панелях"""
    fig = Figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)  # Верхний график для Углерода
    ax2 = fig.add_subplot(212)  # Нижний график для Азота
    
    # Получаем стехиометрические коэффициенты
    gXC = params.get('gXC', 1.0)
    gYC = params.get('gYC', 1.0)
    gXN = params.get('gXN', 0.15)
    gYN = params.get('gYN', 0.15)
    gPC = params.get('gPC', 0.77)
    gFC = params.get('gFC', 1.0)
    gChC = params.get('gChC', 1.0)
    gPN = params.get('gPN', 0.23)
    
    C0 = params.get('C0', 0.11)
    N0 = params.get('N0', 0.15)
    
    X, Y = sol.y[0], sol.y[1]
    Dcl, Dps = sol.y[2], sol.y[3]
    P, F, Ch = sol.y[4], sol.y[5], sol.y[6]
    
    # Считаем свободные минеральные формы строго по уравнениям системы
    C_min = C0 - (gXC*(X + Dcl) + gYC*(Y + Dps) + gPC*P + gFC*F + gChC*Ch)
    N_min = N0 - (gXN*(X + Dcl) + gYN*(Y + Dps) + gPN*P)
    
    # 1. Панель Углерода
    ax1.plot(sol.t, C_min, 'b-', lw=2, label='Минеральный Углерод (C)')
    ax1.set_ylabel('Концентрация C')
    ax1.set_title('Свободный минеральный Углерод в среде')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Панель Азота
    ax2.plot(sol.t, N_min, 'm-', lw=2, label='Минеральный Азот (N)')
    ax2.set_xlabel('Время (t)')
    ax2.set_ylabel('Концентрация N')
    ax2.set_title('Свободный минеральный Азот в среде')
    ax2.grid(True)
    ax2.legend()
    
    fig.tight_layout() # Авто-выравнивание отступов, чтобы заголовки не налезали на оси
    return fig

def graph_conservation(sol, params):
    """График проверки закона сохранения вещества (Материальный баланс)"""
    fig = Figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)  # Верхний график для общего Углерода
    ax2 = fig.add_subplot(212)  # Нижний график для общего Азота
    
    gXC = params.get('gXC', 1.0)
    gYC = params.get('gYC', 1.0)
    gXN = params.get('gXN', 0.15)
    gYN = params.get('gYN', 0.15)
    gPC = params.get('gPC', 0.77)
    gFC = params.get('gFC', 1.0)
    gChC = params.get('gChC', 1.0)
    gPN = params.get('gPN', 0.23)
    
    C0 = params.get('C0', 0.11)
    N0 = params.get('N0', 0.15)
    
    X, Y = sol.y[0], sol.y[1]
    Dcl, Dps = sol.y[2], sol.y[3]
    P, F, Ch = sol.y[4], sol.y[5], sol.y[6]
    
    C_min = C0 - (gXC*(X + Dcl) + gYC*(Y + Dps) + gPC*P + gFC*F + gChC*Ch)
    N_min = N0 - (gXN*(X + Dcl) + gYN*(Y + Dps) + gPN*P)
    
    # Суммируем ВСЕ фракции веществ в системе
    Total_C = C_min + (gXC*(X + Dcl) + gYC*(Y + Dps) + gPC*P + gFC*F + gChC*Ch)
    Total_N = N_min + (gXN*(X + Dcl) + gYN*(Y + Dps) + gPN*P)
    
    # 1. Баланс Углерода
    ax1.plot(sol.t, Total_C, 'b--', lw=2.5, label='Общий Углерод системы')
    ax1.set_ylabel('Сумма по всем пулам')
    ax1.set_title(f'Материальный баланс Углерода (Постоянная линия на уровне C0={C0})')
    ax1.set_ylim(C0 * 0.9, C0 * 1.1)  # Рамки видимости для проверки стабильности линии
    ax1.grid(True)
    ax1.legend()
    
    # 2. Баланс Азота
    ax2.plot(sol.t, Total_N, 'm--', lw=2.5, label='Общий Азот системы')
    ax2.set_xlabel('Время (t)')
    ax2.set_ylabel('Сумма по всем пулам')
    ax2.set_title(f'Материальный баланс Азота (Постоянная линия на уровне N0={N0})')
    ax2.set_ylim(N0 * 0.9, N0 * 1.1)
    ax2.grid(True)
    ax2.legend()
    
    fig.tight_layout()
    return fig
    
    # --- ФУНКЦИИ ДЛЯ РАБОТЫ С JSON И ОТЧЁТАМИ ---
def save_params_to_json(params, filepath):
    to_save = {k: v for k, v in params.items() 
               if k not in ['bXCh', 'bYCh', 'gXC', 'gXN', 'gYC', 'gYN']}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)

def load_params_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    params = get_default_params()
    params.update(loaded)
    params = update_dependent_params(params)
    return params