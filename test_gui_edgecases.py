import sys
from PyQt5.QtWidgets import QApplication
from src.gui import MainWindow

def test_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    
    print("Testing Robot Config Tab...")
    window.tab_config.table.item(0, 0).setText("abc")
    window.tab_config._validate()
    print("Robot Config result after invalid int:", window.tab_config.lbl_result.text())

    print("Testing IK Tab...")
    window.tab_ik.guess_table.item(0, 0).setText("abc")
    try:
        window.tab_ik._solve()
        print("IK Solve survived invalid data")
    except Exception as e:
        print("IK Solve CRASHED on invalid data:", type(e).__name__, e)

    print("Testing Validation Tab...")
    window.tab_val._validate()
    print("Validation Tab survived without IK solutions")

if __name__ == "__main__":
    test_gui()
