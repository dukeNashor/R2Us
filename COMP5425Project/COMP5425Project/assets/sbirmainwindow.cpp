#include "sbirmainwindow.h"
#include "ui_sbirmainwindow.h"

SBIRMainWindow::SBIRMainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SBIRMainWindow)
{
    ui->setupUi(this);
}

SBIRMainWindow::~SBIRMainWindow()
{
    delete ui;
}
