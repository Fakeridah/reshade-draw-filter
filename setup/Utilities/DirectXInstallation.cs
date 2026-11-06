using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ReShade.Setup.Pages;

namespace ReShade.Setup.Utilities
{
	public static class DirectXInstallation
	{
		public static string ResolveExecutablePath(string path)
		{
			string target = PEInfo.ReadResourceString(path, 201);
			if (!string.IsNullOrEmpty(target))
			{
				target = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(path), target));
				if (File.Exists(target))
					return target;
			}
			return path;
		}

		public static Api SelectAutomaticApi(IEnumerable<string> modules, bool isUnrealEngine)
		{
			// Unreal Engine games need an API-specific proxy. Prefer the explicitly
			// imported API instead of falling back to DXGI, since UE D3D12 builds can
			// bypass a local dxgi.dll proxy entirely.
			if (isUnrealEngine)
			{
				if (modules.Any(m => m.Equals("d3d12.dll", StringComparison.OrdinalIgnoreCase)))
					return Api.D3D12;

				if (modules.Any(m => m.Equals("d3d11.dll", StringComparison.OrdinalIgnoreCase)))
					return Api.D3D11;

				// UE D3D12 builds commonly import only dxgi.dll and load d3d12.dll
				// dynamically, so a static import scan cannot see the actual RHI.
				if (modules.Any(m => m.Equals("dxgi.dll", StringComparison.OrdinalIgnoreCase)))
					return Api.D3D12;
			}

			// Keep DXGI for non-Unreal games and ambiguous executables.
			return Api.DXGI;
		}
	}
}
