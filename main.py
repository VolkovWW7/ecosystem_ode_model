import sys
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
import numpy as np
import mathmodel
from gui import BioOxWindow

class BioOxController:
    def __init__(self):
        self.window = BioOxWindow()
        self.last_solution = None

        # Подключение сигналов
        self.window.btn_run.clicked.connect(self.run_calculation)
        self.window.btn_import.clicked.connect(self.window.import_params)
        self.window.btn_export.clicked.connect(self.window.export_params)
        self.window.btn_report.clicked.connect(self.create_report)

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

            figs = {
                "trofs": mathmodel.graph_Trofs(solution),
                "detrit": mathmodel.graph_Detrit(solution),
                "pfc": mathmodel.graph_PFC(solution),
                "atp": mathmodel.graph_Ac(solution)
            }
            self.window.display_figures(figs)

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

        import os
        base_name = os.path.splitext(base_path)[0]

        # CSV
        csv_path = base_name + ".csv"
        t = self.last_solution.t
        y = self.last_solution.y
        header = "time,X,Y,Dcl,Dps,P,F,Ch,Ac"
        data = np.column_stack((t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]))
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        # PDF
        pdf_path = base_name + ".pdf"
        mathmodel.save_report_pdf(self.last_solution, pdf_path)

        # JSON (текущие параметры)
        json_path = base_name + ".json"
        full_params = mathmodel.get_default_params()
        full_params.update(self.window.get_all_parameters())
        full_params = mathmodel.update_dependent_params(full_params)
        mathmodel.save_params_to_json(full_params, json_path)

        QMessageBox.information(self.window, "Отчёт готов",
                                f"Сохранено:\n{csv_path}\n{pdf_path}\n{json_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = BioOxController()
    controller.window.show()
    sys.exit(app.exec())
