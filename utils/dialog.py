from PyQt6 import QtWidgets


def askYesNo(text:str)->bool:
    # Create a message box
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)  # Set the icon to Question
    msg_box.setWindowTitle("Confirmation")  # Set the title of the message box
    msg_box.setText(text)  # Main question
    msg_box.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No
    )  # Add Yes and No buttons

    user_choice = msg_box.exec()

    return user_choice == QtWidgets.QMessageBox.StandardButton.Yes

def showErrorMessage(errorMessage:str)->None:
    # Create a message box
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(
        QtWidgets.QMessageBox.Icon.Critical
    )  # Set the icon to Critical (error)
    msg_box.setWindowTitle("Error")  # Set the title of the message box
    msg_box.setText("An error occurred!")  # Set the main text
    msg_box.setInformativeText(errorMessage)  # Optional additional text
    msg_box.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Ok
    )  # Add standard buttons
    msg_box.exec()  # Display the message box
