#include "Game.h"

using namespace game;


HDC game::g_hdc = NULL;

HPEN g_hPen[7] = { 0 };
HBRUSH g_hBrush[7] = { 0 };
int g_iPenStyle[7] = { PS_SOLID, PS_DASH, PS_DOT, PS_DASHDOT, PS_DASHDOTDOT, PS_NULL, PS_INSIDEFRAME };
int g_iBrushStyle[6] = { HS_VERTICAL, HS_HORIZONTAL, HS_CROSS, HS_DIAGCROSS, HS_FDIAGONAL, HS_BDIAGONAL };


BOOL game::Game_Init(HWND hwnd)
{
    g_hdc = GetDC(hwnd);
    srand(0);

    for (int i = 0; i < 7; ++i)
    {
        g_hPen[i] = CreatePen(g_iPenStyle[i], 1, RGB(rand() % 256, rand() % 256, rand() % 256));
        if (i == 6)
            g_hBrush[i] = CreateSolidBrush(RGB(rand() % 256, rand() % 256, rand() % 256));
        else
            g_hBrush[i] = CreateHatchBrush(g_iBrushStyle[i], RGB(rand() % 256, rand() % 256, rand() % 256));
    }

    Game_Paint(hwnd);
    ReleaseDC(hwnd, g_hdc);
    return TRUE;
}

BOOL game::Game_CleanUp(HWND hwnd)
{
    for (int i = 0; i < 7; ++i)
    {
        DeleteObject(g_hPen[i]);
        DeleteObject(g_hBrush[i]);
    }

    return TRUE;
}

VOID game::Game_Paint(HWND hwnd)
{
    int y = 0;

    for (int i = 0; i < 7; ++i)
    {
        y = (i + 1) * 70;
        SelectObject(g_hdc, g_hPen[i]);
        MoveToEx(g_hdc, 30, y, NULL);
        LineTo(g_hdc, 100, y);
    }

    int x1 = 120;
    int x2 = 190;

    for (int i = 0; i < 7; ++i)
    {
        SelectObject(g_hdc, g_hBrush[i]);
        Rectangle(g_hdc, x1, 70, x2, y);
        x1 += 90;
        x2 += 90;
    }

    wchar_t text[] = L"超级机器人 版本 v0.0.1";
    SetTextColor(g_hdc, RGB(0, 100, 200));
    SetBkMode(g_hdc, TRANSPARENT);
    TextOut(g_hdc, 0, 0, text, wcslen(text));
    SetTextColor(g_hdc, RGB(100, 200, 0));
    RECT textRect{ 0, 50, 300, 100 };
    DrawText(g_hdc, text, wcslen(text), &textRect, DT_SINGLELINE | DT_VCENTER);

}
