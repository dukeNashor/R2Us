#pragma once

#include "framework.h"
#include "WindowsProject.h"

namespace game
{

    BOOL Game_Init(HWND hwnd);

    VOID Game_Paint(HWND hwnd);

    BOOL Game_CleanUp(HWND hwnd);

    extern HDC g_hdc;
}