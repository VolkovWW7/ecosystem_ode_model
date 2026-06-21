import sys
import os
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide6.QtCore import QThread, Signal, Slot
import numpy as np
import mathmodel
from gui import BioOxWindow
import calibrate_gui

# Класс для перенаправления sys.stdout в Qt Signal
class LogRedirector:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        if text.strip(): # Игнорируем пустые переносы строк для красоты
            self.signal.emit(text)

    def flush(self):
        pass

# Поток для фонового выполнения калибровки
class CalibrationWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)  

    def __init__(self, kinetic_model):
            super().__init__()
            self.kinetic_model = kinetic_model # Сохраняем выбранный тип уравнений

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(self.log_signal)
        
        try:
            # Вызов функции теперь возвращает имя файла (или None)
            saved_file = calibrate_gui.run_calibration_gui(self.kinetic_model)
            
            if saved_file:
                self.finished_signal.emit(True, saved_file)
            else:
                self.finished_signal.emit(False, "")
        except Exception as e:
            print(f"Ошибка в ходе калибровки: {str(e)}")
            self.finished_signal.emit(False, "")
        finally:
            sys.stdout = old_stdout


class BioOxController:
    def __init__(self):
        self.window = BioOxWindow()
        self.last_solution = None
        self.calibration_thread = None # Ссылка на поток каллибровки

        # Подключение сигналов
        self.window.btn_run.clicked.connect(self.run_calculation)
        self.window.btn_import.clicked.connect(self.window.import_params)
        self.window.btn_export.clicked.connect(self.window.export_params)
        self.window.btn_report.clicked.connect(self.create_report)

        # === ПОДКЛЮЧЕНИЕ СИГНАЛА КАЛИБРОВКИ ===
        self.window.btn_start_calibrate.clicked.connect(self.start_calibration)

    #функция "поиск решения"
    def start_calibration(self):
        # Проверяем, не запущен ли расчёт уже сейчас
        if self.calibration_thread and self.calibration_thread.isRunning():
            QMessageBox.warning(self.window, "Внимание", "Калибровка уже выполняется!")
            return

        # перед новым запуском
        #self.window.txt_calibrate_log.clear()
        self.window.btn_start_calibrate.setEnabled(False)
        self.window.btn_start_calibrate.setText("Идёт калибровка...")
        # Извлекаем выбранную пользователем модель из GUI параметров
        current_model = self.window.get_all_parameters()['kinetic_model']

        # Создаем и настраиваем рабочий поток
        self.calibration_thread = CalibrationWorker(current_model)
        self.calibration_thread.log_signal.connect(self.append_log)
        self.calibration_thread.finished_signal.connect(self.calibration_finished)
        
        # Запуск потока (вызовет метод run() в фоне)
        self.calibration_thread.start()

    @Slot(str)
    def append_log(self, text):
        # Метод безопасно добавляет текст в QTextEdit из фонового потока
        self.window.txt_calibrate_log.append(text)

    @Slot(bool, str)  # <-- Принимает два аргумента
    def calibration_finished(self, success, filename):
        self.window.btn_start_calibrate.setEnabled(True)
        self.window.btn_start_calibrate.setText("Запустить калибровку")
        
        if success:
            QMessageBox.information(
                self.window, 
                "Успех", 
                f"Калибровка успешно завершена!\n\nРезультаты сохранены в файл:\n{filename}"
            )
            
            if filename and os.path.exists(filename):
                new_params = mathmodel.load_params_from_json(filename)
                # Отбираем только те параметры, которые есть в виджетах GUI
                gui_params = {k: new_params[k] for k in self.window.params_widgets.keys() if k in new_params}
                self.window.set_parameters(gui_params)
                self.window.txt_calibrate_log.append(f"\n[Система]: Новые калиброванные параметры автоматически загружены в поля ввода.")

                # === Автоматическое построение графика верификации по итогам калибровки ===
                full_params = mathmodel.get_default_params()
                full_params.update(gui_params)
                full_params = mathmodel.update_dependent_params(full_params)
                val_fig = calibrate_gui.graph_validation(full_params)
                self.window.display_validation_figure(val_fig)
        else:
            QMessageBox.critical(self.window, "Ошибка", "В процессе оптимизации произошла ошибка.")

    def run_calculation(self):
        try:
            self.window.btn_run.setEnabled(False)
            self.window.btn_run.setText("Расчет...")
            QApplication.processEvents()

            full_params = mathmodel.get_default_params()
            gui_params = self.window.get_all_parameters()
            full_params.update(gui_params)
            full_params = mathmodel.update_dependent_params(full_params)

            solution = mathmodel.run_simulation(full_params,"LSODA")
            self.last_solution = solution

            #== Формируем графики для отправки в GUI (внутри run_calculation) ==
            figs = {
                "trofs": mathmodel.graph_Trofs(solution),
                "growth": mathmodel.graph_autotroph_growth(solution, full_params),
                "detrit": mathmodel.graph_Detrit(solution),
                "pfc": mathmodel.graph_PFC(solution),
                "atp": mathmodel.graph_Ac(solution),
                'minerals': mathmodel.graph_minerals(solution, full_params),     
                'conservation': mathmodel.graph_conservation(solution, full_params)
            }
            #Вывод графиков модели
            self.window.display_figures(figs)

            # === Обновление графика верификации при обычном расчете ===
            val_fig = calibrate_gui.graph_validation(full_params)
            self.window.display_validation_figure(val_fig)

            try:
                metrics = calibrate_gui.evaluate_metrics(full_params)
                if metrics:
                    log_text = (
                        f"\n[Метрики качества для текущих параметров]:\n"
                        f"  NRMSE = {metrics['NRMSE']:.6f}\n"
                        f"  RMSRE = {metrics['RMSRE']:.6f}\n"
                        f"  RMSE  = {metrics['RMSE']:.6f}"
                    )
                    self.window.txt_calibrate_log.append(log_text)
            except Exception as e:
                print(f"Ошибка при вычислении метрик: {e}")
                self.window.txt_calibrate_log.append(f"Ошибка при вычислении метрик: {e}")

        except Exception as e:
            QMessageBox.critical(self.window, "Ошибка", str(e))
        finally:
            self.window.btn_run.setEnabled(True)
            self.window.btn_run.setText("Запустить расчет")

    def create_report(self):
        if self.last_solution is None:
            QMessageBox.warning(self.window, "Нет данных", "Сначала выполните расчёт.")
            return

        base_path, _ = QFileDialog.getSaveFileName(
            self.window, "Сохранить отчёт", "", "Базовое имя (*)")
        if not base_path:
            return

        base_name = os.path.splitext(base_path)[0]

        # == 1. Сразу собираем АКТУАЛЬНЫЕ параметры из GUI для всего отчета ==
        full_params = mathmodel.get_default_params()
        full_params.update(self.window.get_all_parameters())
        full_params = mathmodel.update_dependent_params(full_params)

        #== CSV ==
        csv_path = base_name + ".csv"
        t = self.last_solution.t
        y = self.last_solution.y
        header = (
            "Time,"
            "Autotrophs_X,"
            "Heterotrophs_Y,"
            "Detritus_cl,"
            "Detritus_ps,"
            "Proteins_P,"
            "Lipids_F,"
            "Carbohydrates_Ch,"
            "ATP_Ac"
        )
        data = np.column_stack((t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]))
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        #==Экспорт графиков в PNG (высокое качество) и SVG (вектор)==
        # Генерируем фигуры через функции вашего mathmodel.py
        fig_dynamics = mathmodel.graph_Trofs(self.last_solution) # если функции так называются
        fig_pfc = mathmodel.graph_PFC(self.last_solution)
        fig_atp = mathmodel.graph_Ac(self.last_solution)
        
        # Сохраняем динамику биомассы
        fig_dynamics.savefig(f"{base_name}_biomass.png", dpi=300, bbox_inches='tight')
        fig_dynamics.savefig(f"{base_name}_biomass.svg", bbox_inches='tight')

        # Сохраняем график прироста автотрофов
        fig_growth = mathmodel.graph_autotroph_growth(self.last_solution, full_params)
        fig_growth.savefig(f"{base_name}_growth.png", dpi=300, bbox_inches='tight')
        fig_growth.savefig(f"{base_name}_growth.svg", bbox_inches='tight')
        
        # Сохраняем БЖУ
        fig_pfc.savefig(f"{base_name}_biochemical.png", dpi=300, bbox_inches='tight')
        fig_pfc.savefig(f"{base_name}_biochemical.svg", bbox_inches='tight')
        
        # Сохраняем АТФ
        fig_atp.savefig(f"{base_name}_atp.png", dpi=300, bbox_inches='tight')
        fig_atp.savefig(f"{base_name}_atp.svg", bbox_inches='tight')

        # Сохраняем график минеральных веществ        
        fig_minerals = mathmodel.graph_minerals(self.last_solution, full_params)
        fig_minerals.savefig(f"{base_name}_minerals.png", dpi=300, bbox_inches='tight')
        fig_minerals.savefig(f"{base_name}_minerals.svg", bbox_inches='tight')

        # Сохраняем график закона сохранения вещества
        fig_conservation = mathmodel.graph_conservation(self.last_solution, full_params)
        fig_conservation.savefig(f"{base_name}_conservation.png", dpi=300, bbox_inches='tight')
        fig_conservation.savefig(f"{base_name}_conservation.svg", bbox_inches='tight')

        # Строим текущий график верификации (из калибровки - фиттинга)
        fig_val = calibrate_gui.graph_validation(full_params)
        fig_val.savefig(f"{base_name}_validation.png", dpi=300, bbox_inches='tight')

        # == JSON (текущие параметры) ==
        json_path = base_name + ".json"
        mathmodel.save_params_to_json(full_params, json_path)

        report_message = (
            "Графики успешно экспортированы в форматах PNG и SVG!\n\n"
            f"Сохранено:\n"
            f"• Таблица данных: {os.path.basename(csv_path)}\n"
            f"• Файл параметров: {os.path.basename(json_path)}\n"
            f"• Изображения с префиксом: {os.path.basename(base_name)}_*.png/svg"
        )
        QMessageBox.information(self.window, "Отчёт готов", report_message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = BioOxController()
    controller.window.showMaximized()
    sys.exit(app.exec())
 