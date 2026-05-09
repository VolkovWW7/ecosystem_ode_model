import sys
from PySide6.QtWidgets import QApplication
import mathmodel
from gui import BioOxWindow

class BioOxController:
    def __init__(self):
        # 1. Инициализируем окно
        self.window = BioOxWindow()
        
        # 2. Подключаем сигналы кнопок
        self.window.btn_run.clicked.connect(self.run_calculation)
        
        # Заполняем интерфейс начальными данными из модели (опционально)
        # Для простоты в gui.py уже стоят значения по умолчанию

    def get_params_from_gui(self):
        """Собирает все значения из DoubleSpinBox в словарь параметров."""
        # Получаем стандартный набор параметров
        p = mathmodel.get_default_params()
        
        # Перезаписываем те значения, которые есть в GUI
        p['X0'] = self.window.sp_X0.value()
        p['Y0'] = self.window.sp_Y0.value()
        p['C0'] = self.window.sp_C0.value()
        p['N0'] = self.window.sp_N0.value()
        p['Vx_c'] = self.window.sp_Vx_c.value()
        p['Vy_c'] = self.window.sp_Vy_c.value()
        p['Vm'] = self.window.sp_Vm.value()
        p['total_time'] = self.window.sp_total_time.value()
        p['output_step'] = self.window.sp_output_step.value()
        
        # Пересчитываем зависимые коэффициенты
        p['bXCh'] = 1.0 - p['bXP'] - p['bXF']
        p['bYCh'] = 1.0 - p['bYP'] - p['bYF']
        p['gXC'] = p['bXP'] * p['gPC'] + p['bXF'] * p['gFC'] + p['bXCh'] * p['gChC']
        p['gXN'] = p['bXP'] * p['gPN']
        p['gYC'] = p['bYP'] * p['gPC'] + p['bYF'] * p['gFC'] + p['bYCh'] * p['gChC']
        p['gYN'] = p['bYP'] * p['gPN']
        
        return p

    def run_calculation(self):
        """Основной цикл: Сбор параметров -> Расчет -> Визуализация."""
        try:
            # Блокируем кнопку на время расчета
            self.window.btn_run.setEnabled(False)
            self.window.btn_run.setText("Считаю...")
            QApplication.processEvents() # Чтобы интерфейс не завис

            # 1. Собираем данные
            params = self.get_params_from_gui()
            
            # 2. Запускаем интегратор
            solution = mathmodel.run_simulation(params)
            
            # 3. Генерируем объекты Figure Matplotlib
            figs = {
                "trofs": mathmodel.graph_Trofs(solution),
                "detrit": mathmodel.graph_Detrit(solution),
                "pfc": mathmodel.graph_PFC(solution),
                "atp": mathmodel.graph_Ac(solution)
            }
            
            # 4. Передаем их в GUI для отрисовки на вкладках
            self.window.display_figures(figs)
            
        except Exception as e:
            print(f"Ошибка при расчете: {e}")
        finally:
            self.window.btn_run.setEnabled(True)
            self.window.btn_run.setText("Запустить расчет")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Создаем контроллер
    controller = BioOxController()
    controller.window.show()
    
    sys.exit(app.exec())