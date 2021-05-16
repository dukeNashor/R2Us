#ifndef SBIRMAINWINDOW_H
#define SBIRMAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class SBIRMainWindow;
}

class SBIRMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit SBIRMainWindow(QWidget *parent = nullptr);
    ~SBIRMainWindow();

private:
    Ui::SBIRMainWindow *ui;
};

#endif // SBIRMAINWINDOW_H
