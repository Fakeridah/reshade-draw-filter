/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

namespace reshade
{
	class ini_file;
}

// Set during DLL_PROCESS_ATTACH when a graphics capture tool is already loaded
// into the process. Export entry points use this to pass straight through to
// the original system graphics DLL without creating ReShade proxy objects.
extern bool g_reshade_auto_disabled;

bool is_capture_tool_loaded(const reshade::ini_file &config);
void update_capture_tool_state();
