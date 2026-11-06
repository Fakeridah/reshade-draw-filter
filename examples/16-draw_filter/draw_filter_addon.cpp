/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: BSD-3-Clause OR MIT
 */

#include <imgui.h>
#include <reshade.hpp>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace reshade::api;

namespace
{
	std::shared_mutex s_mutex;
	std::vector<uint32_t> s_blocked_index_counts;

	struct draw_match
	{
		uint32_t index_count = 0;
		uint32_t instance_count = 0;
		uint32_t first_index = 0;
		int32_t vertex_offset = 0;
		uint32_t first_instance = 0;
		uintptr_t index_buffer = 0;
		uint32_t index_format = DXGI_FORMAT_UNKNOWN;
		uint32_t index_buffer_offset = 0;
		uintptr_t vertex_buffer = 0;
		uint32_t vertex_buffer_stride = 0;
		uint32_t vertex_buffer_offset = 0;
		uintptr_t input_layout = 0;
		uintptr_t vertex_shader = 0;
		uintptr_t pixel_shader = 0;
		uint32_t render_target_format = DXGI_FORMAT_UNKNOWN;
		uint32_t depth_stencil_format = DXGI_FORMAT_UNKNOWN;
		D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		bool valid = false;
	};

	std::vector<draw_match> s_blocked_draws;
	// Vertex buffers can be shared between a visible pass and a depth-only pass.
	// These targets are learned from filtered visible draws and are used only for
	// matching depth-only draws.
	struct depth_mesh_target
	{
		draw_match match = {};
		std::vector<uint32_t> vertex_indices;
	};
	std::vector<depth_mesh_target> s_depth_mesh_targets;

	struct partial_depth_buffer
	{
		uintptr_t original_index_buffer = 0;
		uint32_t index_format = DXGI_FORMAT_UNKNOWN;
		uint32_t index_buffer_offset = 0;
		uint32_t draw_first_index = 0;
		uint32_t draw_index_count = 0;
		std::vector<std::pair<uint32_t, uint32_t>> zero_ranges;
		ID3D11Buffer *modified_index_buffer = nullptr;
	};
	std::mutex s_partial_depth_mutex;
	std::vector<partial_depth_buffer> s_partial_depth_buffers;
	thread_local bool s_in_partial_depth_rewrite = false;
	std::atomic_bool s_enabled = true;
	std::atomic_uint64_t s_skipped_draws = 0;
	uint32_t s_input_index_count = 0;
	std::atomic_uint s_pick_limit = 32;
	// Keep enough query slots for a very busy frame. D3D11 has no practical
	// result-count limit; this only bounds the single D3D12 query heap.
	constexpr uint32_t s_d3d12_query_capacity = 65536;
	std::mutex s_d3d12_query_mutex;
	device *s_d3d12_query_device = nullptr;
	query_heap s_d3d12_query_heap = {};
	std::atomic_bool s_pick_requested = false;
	std::atomic_uint s_pick_target_width = 0;
	std::atomic_uint s_pick_target_height = 0;
	std::atomic_uint s_pick_target_format = 0;
	// The picker brackets the real draw with a one-pixel occlusion query. It no
	// longer replays the draw on the application's context.
	constexpr bool s_enable_native_center_probe = true;

	struct draw_state
	{
		resource render_target = {};
		std::vector<resource_view> render_targets;
		resource_view depth_stencil = {};
		bool in_render_pass = false;
		pipeline current_pipeline = {};
		primitive_topology topology = primitive_topology::undefined;
		std::vector<viewport> viewports;
		std::vector<rect> scissors;
		resource index_buffer = {};
		uint64_t index_buffer_offset = 0;
		uint32_t index_size = 0;
	};

	struct pick_candidate
	{
		uint64_t draw_order = 0;
		uint32_t index_count = 0;
		uint32_t instance_count = 0;
		uint32_t first_index = 0;
		int32_t vertex_offset = 0;
		uint32_t first_instance = 0;
		D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		draw_match match = {};
		std::string event_name;
		uint64_t samples_passed = 0;
		ID3D11Query *query = nullptr;
		query_heap api_query_heap = {};
		uint32_t api_query_index = UINT32_MAX;
	};

	struct marker_state
	{
		std::vector<std::string> event_stack;
		std::string last_marker;
	};

	struct probe_state
	{
		pick_candidate candidate = {};
		command_list *api_cmd_list = nullptr;
		device *api_device = nullptr;
		query_heap api_query_heap = {};
		uint32_t api_query_index = UINT32_MAX;
		std::vector<resource_view> old_render_targets;
		resource_view api_old_depth_stencil = {};
		std::vector<rect> api_old_scissors;
		bool api_in_render_pass = false;
		int32_t api_center_x = 0;
		int32_t api_center_y = 0;
		bool is_d3d12 = false;
		ID3D11DeviceContext *context = nullptr;
		ID3D11Device *device = nullptr;
		ID3D11RasterizerState *old_rasterizer = nullptr;
		ID3D11RasterizerState *probe_rasterizer = nullptr;
		ID3D11DepthStencilState *old_depth_stencil = nullptr;
		ID3D11DepthStencilState *probe_depth_stencil = nullptr;
		ID3D11BlendState *old_blend = nullptr;
		ID3D11BlendState *probe_blend = nullptr;
		D3D11_RECT old_scissors[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
		UINT old_scissor_count = 0;
		UINT old_stencil_reference = 0;
		FLOAT old_blend_factor[4] = {};
		UINT old_sample_mask = 0;
		bool query_started = false;
	};

	enum class pick_status
	{
		idle,
		armed,
		scanning,
		waiting,
		ready,
		no_result,
		unsupported,
		disabled,
	};

	std::mutex s_pick_mutex;
	std::mutex s_state_mutex;
	pick_status s_pick_state = pick_status::idle;
	bool s_pick_armed = false;
	bool s_pick_active = false;
	bool s_pick_finished = false;
	uint32_t s_pick_wait_frames = 0;
	uint64_t s_pick_draw_order = 0;
	std::vector<pick_candidate> s_pending_pick;
	std::vector<pick_candidate> s_pick_results;
	std::mutex s_probe_mutex;
	std::unordered_map<command_list *, probe_state> s_active_probes;
	std::mutex s_marker_mutex;
	std::unordered_map<command_list *, marker_state> s_marker_states;
	std::set<resource> s_swapchain_back_buffers;
	std::unordered_map<command_list *, draw_state> s_command_states;
	std::atomic_bool s_after_draw_event_registered = false;
	std::atomic_uint s_special_draw_trace_count = 0;
	thread_local bool s_in_center_probe = false;

	std::mutex s_isolated_shader_mutex;
	ID3D11PixelShader *s_isolated_discard_pixel_shader = nullptr;
	ID3D11DepthStencilState *s_isolated_no_depth_state = nullptr;

	void add_draw_match(const draw_match &match);
	bool capture_draw_match(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance, draw_match &match);

	bool center_picker_needs_state()
	{
		std::lock_guard lock(s_pick_mutex);
		return s_pick_armed || s_pick_active;
	}

	bool is_observer_compatibility_path()
	{
		static const bool result = [] {
			char module_path[MAX_PATH] = {};
			const DWORD length = GetModuleFileNameA(nullptr, module_path, ARRAYSIZE(module_path));
			if (length == 0 || length >= ARRAYSIZE(module_path))
				return false;
			const char *module_name = std::strrchr(module_path, '\\');
			module_name = module_name != nullptr ? module_name + 1 : module_path;
			return _stricmp(module_name, "ObserverSystemRedux.exe") == 0;
		}();
		return result;
	}

	bool ensure_isolated_discard_pixel_shader(ID3D11DeviceContext *context)
	{
		std::lock_guard lock(s_isolated_shader_mutex);
		if (s_isolated_discard_pixel_shader != nullptr && s_isolated_no_depth_state != nullptr)
			return true;
		if (context == nullptr)
			return false;

		ID3D11Device *device = nullptr;
		context->GetDevice(&device);
		if (device == nullptr)
			return false;

		if (s_isolated_no_depth_state == nullptr)
		{
			D3D11_DEPTH_STENCIL_DESC depth_desc = {};
			depth_desc.DepthEnable = FALSE;
			depth_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			depth_desc.DepthFunc = D3D11_COMPARISON_ALWAYS;
			depth_desc.StencilEnable = FALSE;
			depth_desc.StencilWriteMask = 0;
			if (FAILED(device->CreateDepthStencilState(&depth_desc, &s_isolated_no_depth_state)))
			{
				device->Release();
				return false;
			}
		}

		static constexpr char source[] = "void main() { discard; }";
		ID3DBlob *shader_code = nullptr;
		ID3DBlob *errors = nullptr;
		const HRESULT compile_result = D3DCompile(
			source, sizeof(source) - 1, "DrawFilter288Discard", nullptr, nullptr,
			"main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader_code, &errors);
		if (FAILED(compile_result))
		{
			const char *message = errors != nullptr ? static_cast<const char *>(errors->GetBufferPointer()) : "unknown error";
			char log_message[512] = {};
			std::snprintf(log_message, sizeof(log_message), "Draw Filter: failed to compile the isolated 288 discard pixel shader: %s", message);
			reshade::log::message(reshade::log::level::error, log_message);
			if (errors != nullptr) errors->Release();
			device->Release();
			return false;
		}

		const HRESULT create_result = device->CreatePixelShader(shader_code->GetBufferPointer(), shader_code->GetBufferSize(), nullptr, &s_isolated_discard_pixel_shader);
		if (FAILED(create_result))
			reshade::log::message(reshade::log::level::error, "Draw Filter: failed to create the isolated 288 discard pixel shader.");
		if (errors != nullptr) errors->Release();
		if (shader_code != nullptr) shader_code->Release();
		device->Release();
		return SUCCEEDED(create_result);
	}

	void destroy_isolated_discard_pixel_shader()
	{
		std::lock_guard lock(s_isolated_shader_mutex);
		if (s_isolated_discard_pixel_shader != nullptr)
		{
			s_isolated_discard_pixel_shader->Release();
			s_isolated_discard_pixel_shader = nullptr;
		}
		if (s_isolated_no_depth_state != nullptr)
		{
			s_isolated_no_depth_state->Release();
			s_isolated_no_depth_state = nullptr;
		}
	}

	bool draw_without_pixel_output(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
	{
		if (cmd_list == nullptr || cmd_list->get_device()->get_api() != device_api::d3d11)
			return false;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr || !ensure_isolated_discard_pixel_shader(context))
			return false;

		ID3D11PixelShader *old_pixel_shader = nullptr;
		ID3D11DepthStencilState *old_depth_stencil = nullptr;
		UINT old_stencil_reference = 0;
		context->PSGetShader(&old_pixel_shader, nullptr, nullptr);
		context->OMGetDepthStencilState(&old_depth_stencil, &old_stencil_reference);
		context->PSSetShader(s_isolated_discard_pixel_shader, nullptr, 0);
		context->OMSetDepthStencilState(s_isolated_no_depth_state, old_stencil_reference);
		cmd_list->draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
		context->OMSetDepthStencilState(old_depth_stencil, old_stencil_reference);
		context->PSSetShader(old_pixel_shader, nullptr, 0);
		if (old_pixel_shader != nullptr)
			old_pixel_shader->Release();
		if (old_depth_stencil != nullptr)
			old_depth_stencil->Release();
		return true;
	}

	const char *pick_status_text(const pick_status status)
	{
		switch (status)
		{
		case pick_status::armed: return "Waiting for next frame";
		case pick_status::scanning: return "Scanning screen center";
		case pick_status::waiting: return "Waiting for GPU result";
		case pick_status::ready: return "Results ready";
		case pick_status::no_result: return "No indexed draw covered the center";
		case pick_status::unsupported: return "Center picking requires D3D11 or D3D12";
		case pick_status::disabled: return "Center picking disabled for stability";
		default: return "Idle";
		}
	}

	void add_index_count(uint32_t index_count)
	{
		if (index_count == 0)
			return;

		std::vector<draw_match> recent_matches;
		{
			std::lock_guard lock(s_pick_mutex);
			for (const pick_candidate &candidate : s_pick_results)
				if (candidate.index_count == index_count && candidate.match.valid)
					recent_matches.push_back(candidate.match);
		}
		if (!recent_matches.empty())
		{
			for (const draw_match &match : recent_matches)
				add_draw_match(match);
			return;
		}

		std::unique_lock lock(s_mutex);
		if (std::find(s_blocked_index_counts.cbegin(), s_blocked_index_counts.cend(), index_count) == s_blocked_index_counts.cend())
			s_blocked_index_counts.push_back(index_count);
	}

	void on_init_swapchain(swapchain *swapchain, bool)
	{
		std::lock_guard lock(s_state_mutex);
		for (uint32_t i = 0; i < swapchain->get_back_buffer_count(); ++i)
			s_swapchain_back_buffers.insert(swapchain->get_back_buffer(i));
	}

	void on_destroy_swapchain(swapchain *swapchain, bool)
	{
		std::lock_guard lock(s_state_mutex);
		for (uint32_t i = 0; i < swapchain->get_back_buffer_count(); ++i)
			s_swapchain_back_buffers.erase(swapchain->get_back_buffer(i));
	}

	void on_bind_render_targets_and_depth_stencil(command_list *cmd_list, uint32_t count, const resource_view *rtvs, resource_view dsv)
	{
		if (cmd_list == nullptr || !center_picker_needs_state())
			return;

		std::lock_guard lock(s_state_mutex);
		draw_state &state = s_command_states[cmd_list];
		state.render_targets.clear();
		if (count != 0 && rtvs != nullptr)
			state.render_targets.assign(rtvs, rtvs + count);
		// D3D11 permits null entries in an RTV array. Do not pass a null view
		// handle to get_resource_from_view(), which dereferences it in the native
		// implementation (Observer emits such binds while the picker is armed).
		state.render_target = count != 0 && rtvs != nullptr && rtvs[0] != 0 ? cmd_list->get_device()->get_resource_from_view(rtvs[0]) : resource {};
		state.depth_stencil = dsv;
	}

	bool on_begin_render_pass(command_list *cmd_list, uint32_t count, const render_pass_render_target_desc *rts, const render_pass_depth_stencil_desc *ds, render_pass_flags)
	{
		if (cmd_list == nullptr || !center_picker_needs_state())
			return false;

		std::lock_guard lock(s_state_mutex);
		draw_state &state = s_command_states[cmd_list];
		state.render_targets.clear();
		if (count != 0 && rts != nullptr)
		{
			state.render_targets.reserve(count);
			for (uint32_t i = 0; i < count; ++i)
				state.render_targets.push_back(rts[i].view);
		}
		state.render_target = !state.render_targets.empty() && state.render_targets[0] != 0 ? cmd_list->get_device()->get_resource_from_view(state.render_targets[0]) : resource {};
		state.depth_stencil = ds != nullptr ? ds->view : resource_view {};
		state.in_render_pass = true;
		return false;
	}

	bool on_end_render_pass(command_list *cmd_list)
	{
		if (cmd_list == nullptr || !center_picker_needs_state())
			return false;

		std::lock_guard lock(s_state_mutex);
		const auto it = s_command_states.find(cmd_list);
		if (it == s_command_states.end())
			return false;
		it->second.render_targets.clear();
		it->second.render_target = {};
		it->second.depth_stencil = {};
		it->second.in_render_pass = false;
		return false;
	}

	void on_bind_pipeline_states(command_list *cmd_list, uint32_t count, const dynamic_state *states, const uint32_t *values)
	{
		if (cmd_list == nullptr || !center_picker_needs_state() || states == nullptr || values == nullptr)
			return;

		std::lock_guard lock(s_state_mutex);
		draw_state &state = s_command_states[cmd_list];
		for (uint32_t i = 0; i < count; ++i)
			if (states[i] == dynamic_state::primitive_topology)
				state.topology = static_cast<primitive_topology>(values[i]);
	}

	void on_bind_viewports(command_list *cmd_list, uint32_t first, uint32_t count, const viewport *viewports)
	{
		if (cmd_list == nullptr || !center_picker_needs_state())
			return;

		std::lock_guard lock(s_state_mutex);
		draw_state &state = s_command_states[cmd_list];
		if (first == 0 && count == 0)
		{
			state.viewports.clear();
			return;
		}
		if (viewports == nullptr)
			return;
		if (state.viewports.size() < first + count)
			state.viewports.resize(first + count);
		std::copy(viewports, viewports + count, state.viewports.begin() + first);
	}

	void on_bind_scissor_rects(command_list *cmd_list, uint32_t first, uint32_t count, const rect *rects)
	{
		if (cmd_list == nullptr || !center_picker_needs_state())
			return;

		std::lock_guard lock(s_state_mutex);
		draw_state &state = s_command_states[cmd_list];
		if (first == 0 && count == 0)
		{
			state.scissors.clear();
			return;
		}
		if (rects == nullptr)
			return;
		if (state.scissors.size() < first + count)
			state.scissors.resize(first + count);
		std::copy(rects, rects + count, state.scissors.begin() + first);
	}

	void on_bind_pipeline(command_list *cmd_list, pipeline_stage, pipeline pipeline)
	{
		if (cmd_list == nullptr || !center_picker_needs_state())
			return;

		std::lock_guard lock(s_state_mutex);
		s_command_states[cmd_list].current_pipeline = pipeline;
	}

	bool is_back_buffer(resource resource)
	{
		std::lock_guard lock(s_state_mutex);
		return resource != 0 && s_swapchain_back_buffers.find(resource) != s_swapchain_back_buffers.end();
	}

	static void release_query(ID3D11Query *&query)
	{
		if (query != nullptr)
		{
			query->Release();
			query = nullptr;
		}
	}

	bool ensure_d3d12_query_heap(device *query_device)
	{
		if (query_device == nullptr || query_device->get_api() != device_api::d3d12)
			return false;

		std::lock_guard lock(s_d3d12_query_mutex);
		if (s_d3d12_query_device == query_device && s_d3d12_query_heap != 0)
			return true;

		if (s_d3d12_query_device != nullptr && s_d3d12_query_heap != 0)
			s_d3d12_query_device->destroy_query_heap(s_d3d12_query_heap);
		s_d3d12_query_device = nullptr;
		s_d3d12_query_heap = {};

		if (!query_device->create_query_heap(query_type::occlusion, s_d3d12_query_capacity, &s_d3d12_query_heap))
			return false;

		s_d3d12_query_device = query_device;
		return true;
	}

	void destroy_d3d12_query_heap()
	{
		std::lock_guard lock(s_d3d12_query_mutex);
		if (s_d3d12_query_device != nullptr && s_d3d12_query_heap != 0)
			s_d3d12_query_device->destroy_query_heap(s_d3d12_query_heap);
		s_d3d12_query_device = nullptr;
		s_d3d12_query_heap = {};
	}

	bool capture_draw_match(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance, draw_match &match)
	{
		match = {};
		if (cmd_list == nullptr || cmd_list->get_device()->get_api() != device_api::d3d11)
			return false;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
			return false;

		D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		context->IAGetPrimitiveTopology(&topology);

		ID3D11Buffer *index_buffer = nullptr;
		DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
		UINT index_buffer_offset = 0;
		context->IAGetIndexBuffer(&index_buffer, &index_format, &index_buffer_offset);

		ID3D11Buffer *vertex_buffer = nullptr;
		UINT vertex_buffer_stride = 0;
		UINT vertex_buffer_offset = 0;
		context->IAGetVertexBuffers(0, 1, &vertex_buffer, &vertex_buffer_stride, &vertex_buffer_offset);

		ID3D11InputLayout *input_layout = nullptr;
		context->IAGetInputLayout(&input_layout);

		ID3D11VertexShader *vertex_shader = nullptr;
		ID3D11PixelShader *pixel_shader = nullptr;
		context->VSGetShader(&vertex_shader, nullptr, nullptr);
		context->PSGetShader(&pixel_shader, nullptr, nullptr);

		ID3D11RenderTargetView *render_target = nullptr;
		ID3D11DepthStencilView *depth_stencil = nullptr;
		context->OMGetRenderTargets(1, &render_target, &depth_stencil);

		D3D11_RENDER_TARGET_VIEW_DESC render_target_desc = {};
		D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_desc = {};
		if (render_target != nullptr)
			render_target->GetDesc(&render_target_desc);
		if (depth_stencil != nullptr)
			depth_stencil->GetDesc(&depth_stencil_desc);

		match.index_count = index_count;
		match.instance_count = instance_count;
		match.first_index = first_index;
		match.vertex_offset = vertex_offset;
		match.first_instance = first_instance;
		match.index_buffer = reinterpret_cast<uintptr_t>(index_buffer);
		match.index_format = static_cast<uint32_t>(index_format);
		match.index_buffer_offset = index_buffer_offset;
		match.vertex_buffer = reinterpret_cast<uintptr_t>(vertex_buffer);
		match.vertex_buffer_stride = vertex_buffer_stride;
		match.vertex_buffer_offset = vertex_buffer_offset;
		match.input_layout = reinterpret_cast<uintptr_t>(input_layout);
		match.vertex_shader = reinterpret_cast<uintptr_t>(vertex_shader);
		match.pixel_shader = reinterpret_cast<uintptr_t>(pixel_shader);
		match.render_target_format = static_cast<uint32_t>(render_target_desc.Format);
		match.depth_stencil_format = static_cast<uint32_t>(depth_stencil_desc.Format);
		match.topology = topology;
		match.valid = index_buffer != nullptr;

		if (render_target != nullptr) render_target->Release();
		if (depth_stencil != nullptr) depth_stencil->Release();
		if (vertex_shader != nullptr) vertex_shader->Release();
		if (pixel_shader != nullptr) pixel_shader->Release();
		if (input_layout != nullptr) input_layout->Release();
		if (vertex_buffer != nullptr) vertex_buffer->Release();
		if (index_buffer != nullptr) index_buffer->Release();
		return match.valid;
	}

	void hash_draw_value(uint64_t &hash, uint64_t value)
	{
		// FNV-1a over the little-endian bytes keeps the displayed key stable
		// for the lifetime of this game process.
		for (unsigned int i = 0; i < sizeof(value); ++i)
		{
			hash ^= static_cast<uint8_t>(value >> (i * 8));
			hash *= 1099511628211ull;
		}
	}

	uint64_t draw_signature_key(const draw_match &match)
	{
		uint64_t hash = 14695981039346656037ull;
		hash_draw_value(hash, match.index_count);
		hash_draw_value(hash, match.instance_count);
		hash_draw_value(hash, match.first_index);
		hash_draw_value(hash, static_cast<uint32_t>(match.vertex_offset));
		hash_draw_value(hash, match.first_instance);
		hash_draw_value(hash, match.index_buffer);
		hash_draw_value(hash, match.index_format);
		hash_draw_value(hash, match.index_buffer_offset);
		hash_draw_value(hash, match.vertex_buffer);
		hash_draw_value(hash, match.vertex_buffer_stride);
		hash_draw_value(hash, match.vertex_buffer_offset);
		hash_draw_value(hash, match.input_layout);
		hash_draw_value(hash, match.vertex_shader);
		hash_draw_value(hash, match.pixel_shader);
		hash_draw_value(hash, match.render_target_format);
		hash_draw_value(hash, match.depth_stencil_format);
		hash_draw_value(hash, static_cast<uint32_t>(match.topology));
		return hash;
	}

	bool same_geometry_match(const draw_match &lhs, const draw_match &rhs)
	{
		return lhs.valid && rhs.valid &&
			lhs.index_count == rhs.index_count &&
			lhs.instance_count == rhs.instance_count &&
			lhs.first_index == rhs.first_index &&
			lhs.vertex_offset == rhs.vertex_offset &&
			lhs.first_instance == rhs.first_instance &&
			lhs.index_buffer == rhs.index_buffer &&
			lhs.index_format == rhs.index_format &&
			lhs.index_buffer_offset == rhs.index_buffer_offset &&
			lhs.vertex_buffer == rhs.vertex_buffer &&
			lhs.vertex_buffer_stride == rhs.vertex_buffer_stride &&
			lhs.vertex_buffer_offset == rhs.vertex_buffer_offset &&
			lhs.input_layout == rhs.input_layout &&
			lhs.topology == rhs.topology;
	}

	bool same_draw_match(const draw_match &lhs, const draw_match &rhs)
	{
		return same_geometry_match(lhs, rhs) &&
			lhs.vertex_shader == rhs.vertex_shader &&
			lhs.pixel_shader == rhs.pixel_shader &&
			lhs.render_target_format == rhs.render_target_format &&
			lhs.depth_stencil_format == rhs.depth_stencil_format;
	}

	bool same_vertex_buffer(const draw_match &lhs, const draw_match &rhs)
	{
		return lhs.valid && rhs.valid &&
			lhs.vertex_buffer != 0 &&
			lhs.vertex_buffer == rhs.vertex_buffer &&
			lhs.vertex_buffer_stride == rhs.vertex_buffer_stride &&
			lhs.vertex_buffer_offset == rhs.vertex_buffer_offset;
	}

	bool is_depth_only_draw(const draw_match &match)
	{
		return match.valid &&
			match.pixel_shader == 0 &&
			match.render_target_format == DXGI_FORMAT_UNKNOWN &&
			match.depth_stencil_format != DXGI_FORMAT_UNKNOWN;
	}

	void add_draw_match(const draw_match &match)
	{
		if (!match.valid)
			return;

		std::unique_lock lock(s_mutex);
		if (std::find_if(s_blocked_draws.cbegin(), s_blocked_draws.cend(), [&match](const draw_match &item) { return same_draw_match(item, match); }) == s_blocked_draws.cend())
			s_blocked_draws.push_back(match);
	}

	bool read_index_values(ID3D11DeviceContext *context, ID3D11Buffer *source, DXGI_FORMAT index_format, UINT index_buffer_offset, uint32_t first_index, uint32_t index_count, std::vector<uint32_t> &values)
	{
		values.clear();
		if (context == nullptr || source == nullptr || index_count == 0)
			return false;

		const uint32_t index_size = index_format == DXGI_FORMAT_R16_UINT ? 2u : index_format == DXGI_FORMAT_R32_UINT ? 4u : 0u;
		if (index_size == 0)
			return false;

		D3D11_BUFFER_DESC source_desc = {};
		source->GetDesc(&source_desc);
		const uint64_t byte_start = static_cast<uint64_t>(index_buffer_offset) + static_cast<uint64_t>(first_index) * index_size;
		const uint64_t byte_end = byte_start + static_cast<uint64_t>(index_count) * index_size;
		if (byte_end > source_desc.ByteWidth)
			return false;

		ID3D11Device *device = nullptr;
		context->GetDevice(&device);
		if (device == nullptr)
			return false;

		D3D11_BUFFER_DESC staging_desc = source_desc;
		staging_desc.Usage = D3D11_USAGE_STAGING;
		staging_desc.BindFlags = 0;
		staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		staging_desc.MiscFlags = 0;
		staging_desc.StructureByteStride = 0;
		ID3D11Buffer *staging = nullptr;
		const HRESULT create_result = device->CreateBuffer(&staging_desc, nullptr, &staging);
		device->Release();
		if (FAILED(create_result))
			return false;

		context->CopyResource(staging, source);
		D3D11_MAPPED_SUBRESOURCE mapped = {};
		const HRESULT map_result = context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
		if (FAILED(map_result))
		{
			staging->Release();
			return false;
		}

		values.resize(index_count);
		const uint8_t *bytes = static_cast<const uint8_t *>(mapped.pData) + byte_start;
		for (uint32_t i = 0; i < index_count; ++i)
		{
			if (index_format == DXGI_FORMAT_R16_UINT)
			{
				uint16_t value = 0;
				std::memcpy(&value, bytes + i * index_size, sizeof(value));
				values[i] = value;
			}
			else
			{
				uint32_t value = 0;
				std::memcpy(&value, bytes + i * index_size, sizeof(value));
				values[i] = value;
			}
		}

		context->Unmap(staging, 0);
		staging->Release();
		return true;
	}

	bool capture_depth_mesh_target(command_list *cmd_list, const draw_match &match, depth_mesh_target &target)
	{
		if (cmd_list == nullptr || cmd_list->get_device()->get_api() != device_api::d3d11 || !match.valid || match.vertex_buffer == 0)
			return false;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr)
			return false;

		ID3D11Buffer *index_buffer = nullptr;
		DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
		UINT index_buffer_offset = 0;
		context->IAGetIndexBuffer(&index_buffer, &index_format, &index_buffer_offset);
		if (index_buffer == nullptr || reinterpret_cast<uintptr_t>(index_buffer) != match.index_buffer || index_format != static_cast<DXGI_FORMAT>(match.index_format) || index_buffer_offset != match.index_buffer_offset)
		{
			if (index_buffer != nullptr)
				index_buffer->Release();
			return false;
		}

		std::vector<uint32_t> raw_indices;
		const bool read_succeeded = read_index_values(context, index_buffer, index_format, index_buffer_offset, match.first_index, match.index_count, raw_indices);
		index_buffer->Release();
		if (!read_succeeded)
			return false;

		target = {};
		target.match = match;
		target.vertex_indices.reserve(raw_indices.size());
		for (const uint32_t raw_index : raw_indices)
		{
			const int64_t vertex_index = static_cast<int64_t>(raw_index) + match.vertex_offset;
			if (vertex_index >= 0 && vertex_index <= UINT32_MAX)
				target.vertex_indices.push_back(static_cast<uint32_t>(vertex_index));
		}
		std::sort(target.vertex_indices.begin(), target.vertex_indices.end());
		target.vertex_indices.erase(std::unique(target.vertex_indices.begin(), target.vertex_indices.end()), target.vertex_indices.end());
		return !target.vertex_indices.empty();
	}

	void add_depth_mesh_target(command_list *cmd_list, const draw_match &match)
	{
		if (!match.valid || match.vertex_buffer == 0)
			return;

		{
			std::shared_lock lock(s_mutex);
			if (std::find_if(s_depth_mesh_targets.cbegin(), s_depth_mesh_targets.cend(), [&match](const depth_mesh_target &item) { return same_geometry_match(item.match, match); }) != s_depth_mesh_targets.cend())
				return;
		}

		depth_mesh_target target;
		if (!capture_depth_mesh_target(cmd_list, match, target))
			return;

		std::unique_lock lock(s_mutex);
		if (std::find_if(s_depth_mesh_targets.cbegin(), s_depth_mesh_targets.cend(), [&match](const depth_mesh_target &item) { return same_geometry_match(item.match, match); }) == s_depth_mesh_targets.cend())
			s_depth_mesh_targets.push_back(std::move(target));
	}

	void clear_partial_depth_buffers()
	{
		std::lock_guard lock(s_partial_depth_mutex);
		for (partial_depth_buffer &entry : s_partial_depth_buffers)
		{
			if (entry.modified_index_buffer != nullptr)
				entry.modified_index_buffer->Release();
		}
		s_partial_depth_buffers.clear();
	}

	bool get_depth_targets_for_draw(const draw_match &current, std::vector<depth_mesh_target> &targets)
	{
		targets.clear();
		if (!current.valid || !is_depth_only_draw(current))
			return false;

		std::shared_lock lock(s_mutex);
		for (const depth_mesh_target &target : s_depth_mesh_targets)
		{
			if (!same_vertex_buffer(target.match, current))
				continue;
			targets.push_back(target);
		}
		return !targets.empty();
	}

	ID3D11Buffer *get_or_create_partial_depth_buffer(ID3D11DeviceContext *context, ID3D11Buffer *original, DXGI_FORMAT index_format, UINT index_buffer_offset, uint32_t draw_first_index, uint32_t draw_index_count, const std::vector<std::pair<uint32_t, uint32_t>> &ranges)
	{
		if (context == nullptr || original == nullptr || ranges.empty())
			return nullptr;

		const uint32_t index_size = index_format == DXGI_FORMAT_R16_UINT ? 2u : index_format == DXGI_FORMAT_R32_UINT ? 4u : 0u;
		if (index_size == 0)
			return nullptr;

		std::lock_guard lock(s_partial_depth_mutex);
		for (partial_depth_buffer &entry : s_partial_depth_buffers)
		{
			if (entry.original_index_buffer == reinterpret_cast<uintptr_t>(original) &&
				entry.index_format == static_cast<uint32_t>(index_format) &&
				entry.index_buffer_offset == index_buffer_offset &&
				entry.draw_first_index == draw_first_index &&
				entry.draw_index_count == draw_index_count &&
				entry.zero_ranges == ranges)
				return entry.modified_index_buffer;
		}

		D3D11_BUFFER_DESC original_desc = {};
		original->GetDesc(&original_desc);
		for (const auto &[first_index, index_count] : ranges)
		{
			const uint64_t byte_end = static_cast<uint64_t>(index_buffer_offset) + (static_cast<uint64_t>(first_index) + index_count) * index_size;
			if (byte_end > original_desc.ByteWidth)
				return nullptr;
		}

		ID3D11Device *device = nullptr;
		context->GetDevice(&device);
		if (device == nullptr)
			return nullptr;

		D3D11_BUFFER_DESC modified_desc = original_desc;
		modified_desc.Usage = D3D11_USAGE_DEFAULT;
		modified_desc.CPUAccessFlags = 0;
		modified_desc.BindFlags |= D3D11_BIND_INDEX_BUFFER;
		ID3D11Buffer *modified = nullptr;
		if (FAILED(device->CreateBuffer(&modified_desc, nullptr, &modified)))
		{
			device->Release();
			return nullptr;
		}
		device->Release();

		context->CopyResource(modified, original);
		for (const auto &[first_index, index_count] : ranges)
		{
			const UINT byte_start = index_buffer_offset + first_index * index_size;
			const UINT byte_end = byte_start + index_count * index_size;
			const UINT byte_count = byte_end - byte_start;
			std::vector<uint8_t> zero_bytes(byte_count, 0);
			D3D11_BOX box = { byte_start, 0, 0, byte_end, 1, 1 };
			context->UpdateSubresource(modified, 0, &box, zero_bytes.data(), 0, 0);
		}

		partial_depth_buffer entry;
		entry.original_index_buffer = reinterpret_cast<uintptr_t>(original);
		entry.index_format = static_cast<uint32_t>(index_format);
		entry.index_buffer_offset = index_buffer_offset;
		entry.draw_first_index = draw_first_index;
		entry.draw_index_count = draw_index_count;
		entry.zero_ranges = ranges;
		entry.modified_index_buffer = modified;
		s_partial_depth_buffers.push_back(std::move(entry));
		return modified;
	}

	bool draw_depth_without_target_geometry(command_list *cmd_list, const draw_match &current, const std::vector<depth_mesh_target> &targets)
	{
		if (cmd_list == nullptr || cmd_list->get_device()->get_api() != device_api::d3d11 || current.topology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST || targets.empty())
			return false;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr)
			return false;

		ID3D11Buffer *original = nullptr;
		DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
		UINT index_buffer_offset = 0;
		context->IAGetIndexBuffer(&original, &index_format, &index_buffer_offset);
		if (original == nullptr || reinterpret_cast<uintptr_t>(original) != current.index_buffer || index_format != static_cast<DXGI_FORMAT>(current.index_format) || index_buffer_offset != current.index_buffer_offset)
		{
			if (original != nullptr)
				original->Release();
			return false;
		}

		std::vector<uint32_t> raw_indices;
		if (!read_index_values(context, original, index_format, index_buffer_offset, current.first_index, current.index_count, raw_indices))
		{
			original->Release();
			return false;
		}

		std::vector<bool> zero_indices(current.index_count, false);
		for (uint32_t triangle = 0; triangle + 2 < current.index_count; triangle += 3)
		{
			bool belongs_to_target = false;
			for (const depth_mesh_target &target : targets)
			{
				const int64_t vertex0 = static_cast<int64_t>(raw_indices[triangle + 0]) + current.vertex_offset;
				const int64_t vertex1 = static_cast<int64_t>(raw_indices[triangle + 1]) + current.vertex_offset;
				const int64_t vertex2 = static_cast<int64_t>(raw_indices[triangle + 2]) + current.vertex_offset;
				if (vertex0 >= 0 && vertex1 >= 0 && vertex2 >= 0 &&
					std::binary_search(target.vertex_indices.cbegin(), target.vertex_indices.cend(), static_cast<uint32_t>(vertex0)) &&
					std::binary_search(target.vertex_indices.cbegin(), target.vertex_indices.cend(), static_cast<uint32_t>(vertex1)) &&
					std::binary_search(target.vertex_indices.cbegin(), target.vertex_indices.cend(), static_cast<uint32_t>(vertex2)))
				{
					belongs_to_target = true;
					break;
				}
			}

			if (belongs_to_target)
				zero_indices[triangle + 0] = zero_indices[triangle + 1] = zero_indices[triangle + 2] = true;
		}

		std::vector<std::pair<uint32_t, uint32_t>> ranges;
		for (uint32_t i = 0; i < zero_indices.size();)
		{
			if (!zero_indices[i])
			{
				++i;
				continue;
			}
			const uint32_t start = i;
			while (i < zero_indices.size() && zero_indices[i])
				++i;
			ranges.emplace_back(current.first_index + start, i - start);
		}
		if (ranges.empty())
		{
			original->Release();
			return false;
		}

		ID3D11Buffer *modified = get_or_create_partial_depth_buffer(context, original, index_format, index_buffer_offset, current.first_index, current.index_count, ranges);
		if (modified == nullptr)
		{
			original->Release();
			return false;
		}

		context->IASetIndexBuffer(modified, index_format, index_buffer_offset);
		s_in_partial_depth_rewrite = true;
		cmd_list->draw_indexed(current.index_count, current.instance_count, current.first_index, current.vertex_offset, current.first_instance);
		s_in_partial_depth_rewrite = false;
		context->IASetIndexBuffer(original, index_format, index_buffer_offset);
		original->Release();
		return true;
	}

	void add_draw_match(const pick_candidate &candidate)
	{
		add_draw_match(candidate.match);
	}

	void trace_special_draw(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
	{
		if (index_count == 0)
			return;
		if (s_special_draw_trace_count.fetch_add(1, std::memory_order_relaxed) >= 512)
			return;
		if (cmd_list == nullptr || cmd_list->get_device()->get_api() != device_api::d3d11)
			return;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr)
			return;

		D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		context->IAGetPrimitiveTopology(&topology);

		ID3D11Buffer *index_buffer = nullptr;
		DXGI_FORMAT index_format = DXGI_FORMAT_UNKNOWN;
		UINT index_offset = 0;
		context->IAGetIndexBuffer(&index_buffer, &index_format, &index_offset);
		ID3D11Buffer *vertex_buffer = nullptr;
		UINT vertex_stride = 0;
		UINT vb_offset = 0;
		context->IAGetVertexBuffers(0, 1, &vertex_buffer, &vertex_stride, &vb_offset);
		ID3D11InputLayout *input_layout = nullptr;
		context->IAGetInputLayout(&input_layout);

		ID3D11VertexShader *vertex_shader = nullptr;
		ID3D11PixelShader *pixel_shader = nullptr;
		context->VSGetShader(&vertex_shader, nullptr, nullptr);
		context->PSGetShader(&pixel_shader, nullptr, nullptr);

		ID3D11RenderTargetView *render_target = nullptr;
		ID3D11DepthStencilView *depth_stencil = nullptr;
		context->OMGetRenderTargets(1, &render_target, &depth_stencil);

		D3D11_RENDER_TARGET_VIEW_DESC render_target_desc = {};
		D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_desc = {};
		if (render_target != nullptr)
			render_target->GetDesc(&render_target_desc);
		if (depth_stencil != nullptr)
			depth_stencil->GetDesc(&depth_stencil_desc);

		D3D11_VIEWPORT viewport = {};
		UINT viewport_count = 1;
		context->RSGetViewports(&viewport_count, &viewport);

		const char *context_type = "unknown";
		switch (context->GetType())
		{
		case D3D11_DEVICE_CONTEXT_IMMEDIATE: context_type = "immediate"; break;
		case D3D11_DEVICE_CONTEXT_DEFERRED: context_type = "deferred"; break;
		default: break;
		}

		char trace[768] = {};
		std::snprintf(trace, sizeof(trace),
			"Draw Filter trace: index=%u inst=%u first=%u vertex=%d first_inst=%u ctx=%s topo=%u ib=%p ib_fmt=%u ib_off=%u vb=%p vb_stride=%u vb_off=%u il=%p vs=%p ps=%p rtv_fmt=%u dsv_fmt=%u viewport=%.0fx%.0f",
			index_count, instance_count, first_index, vertex_offset, first_instance, context_type,
			static_cast<unsigned int>(topology), static_cast<void *>(index_buffer), static_cast<unsigned int>(index_format), index_offset,
			static_cast<void *>(vertex_buffer), vertex_stride, vb_offset, static_cast<void *>(input_layout),
			static_cast<void *>(vertex_shader), static_cast<void *>(pixel_shader), static_cast<unsigned int>(render_target_desc.Format),
			static_cast<unsigned int>(depth_stencil_desc.Format), viewport.Width, viewport.Height);
		reshade::log::message(reshade::log::level::info, trace);

		if (render_target != nullptr) render_target->Release();
		if (depth_stencil != nullptr) depth_stencil->Release();
		if (vertex_shader != nullptr) vertex_shader->Release();
		if (pixel_shader != nullptr) pixel_shader->Release();
		if (input_layout != nullptr) input_layout->Release();
		if (vertex_buffer != nullptr) vertex_buffer->Release();
		if (index_buffer != nullptr) index_buffer->Release();
	}

	void cleanup_probe_state(probe_state &probe, bool release_query_object)
	{
		if (probe.is_d3d12)
		{
			if (probe.api_cmd_list != nullptr)
			{
				if (probe.query_started && probe.api_query_heap != 0)
					probe.api_cmd_list->end_query(probe.api_query_heap, query_type::occlusion, probe.api_query_index);
				probe.query_started = false;
				// The probe leaves the original RTV/DSV bound. Unbinding them before the
				// draw makes some D3D12 drivers report zero occlusion samples, and is
				// illegal while a render pass is active anyway.
				probe.api_cmd_list->bind_scissor_rects(0, static_cast<uint32_t>(probe.api_old_scissors.size()), probe.api_old_scissors.data());
			}
			probe = {};
			return;
		}

		if (probe.context != nullptr)
		{
			if (probe.query_started && probe.candidate.query != nullptr)
				probe.context->End(probe.candidate.query);
			probe.query_started = false;
			if (probe.old_blend != nullptr || probe.probe_blend != nullptr)
				probe.context->OMSetBlendState(probe.old_blend, probe.old_blend_factor, probe.old_sample_mask);
			if (probe.old_depth_stencil != nullptr || probe.probe_depth_stencil != nullptr)
				probe.context->OMSetDepthStencilState(probe.old_depth_stencil, probe.old_stencil_reference);
			probe.context->RSSetScissorRects(probe.old_scissor_count, probe.old_scissors);
			probe.context->RSSetState(probe.old_rasterizer);
		}

		if (probe.probe_blend != nullptr) probe.probe_blend->Release();
		if (probe.old_blend != nullptr) probe.old_blend->Release();
		if (probe.probe_depth_stencil != nullptr) probe.probe_depth_stencil->Release();
		if (probe.old_depth_stencil != nullptr) probe.old_depth_stencil->Release();
		if (probe.probe_rasterizer != nullptr) probe.probe_rasterizer->Release();
		if (probe.old_rasterizer != nullptr) probe.old_rasterizer->Release();
		if (release_query_object)
			release_query(probe.candidate.query);
		if (probe.device != nullptr) probe.device->Release();
		probe = {};
	}

	bool begin_d3d12_center_probe(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance, const draw_state &state, uint64_t draw_order, probe_state &probe)
	{
		device *api_device = cmd_list != nullptr ? cmd_list->get_device() : nullptr;
		const bool has_render_target = std::any_of(state.render_targets.cbegin(), state.render_targets.cend(), [](const resource_view view) { return view != 0; });
		if (api_device == nullptr || api_device->get_api() != device_api::d3d12 ||
			(!has_render_target && state.depth_stencil == 0) || state.viewports.empty() ||
			draw_order >= s_d3d12_query_capacity || !ensure_d3d12_query_heap(api_device))
			return false;

		const viewport &current_viewport = state.viewports[0];
		if (current_viewport.width < 1.0f || current_viewport.height < 1.0f)
			return false;

		// DX12 draw passes frequently use intermediate render targets, omit a DSV,
		// or leave the scissor state at its default. Use the center of the active
		// viewport instead of requiring it to match the swap-chain dimensions
		// (which is the DX11 path's restriction).
		const int32_t center_x = static_cast<int32_t>(current_viewport.x + current_viewport.width * 0.5f);
		const int32_t center_y = static_cast<int32_t>(current_viewport.y + current_viewport.height * 0.5f);
		if (center_x < static_cast<int32_t>(current_viewport.x) || center_x >= static_cast<int32_t>(current_viewport.x + current_viewport.width) ||
			center_y < static_cast<int32_t>(current_viewport.y) || center_y >= static_cast<int32_t>(current_viewport.y + current_viewport.height))
			return false;
		if (!state.scissors.empty())
		{
			bool inside_scissor = false;
			for (const rect &scissor : state.scissors)
				inside_scissor |= center_x >= scissor.left && center_x < scissor.right && center_y >= scissor.top && center_y < scissor.bottom;
			if (!inside_scissor)
				return false;
		}

		probe = {};
		probe.is_d3d12 = true;
		probe.api_cmd_list = cmd_list;
		probe.api_device = api_device;
		probe.api_query_heap = s_d3d12_query_heap;
		probe.api_query_index = static_cast<uint32_t>(draw_order);
		probe.old_render_targets = state.render_targets;
		probe.api_old_depth_stencil = state.depth_stencil;
		probe.api_old_scissors = state.scissors;
		probe.api_in_render_pass = state.in_render_pass;
		probe.api_center_x = center_x;
		probe.api_center_y = center_y;
		probe.candidate.index_count = index_count;
		probe.candidate.instance_count = instance_count;
		probe.candidate.first_index = first_index;
		probe.candidate.vertex_offset = vertex_offset;
		probe.candidate.first_instance = first_instance;
		probe.candidate.topology = static_cast<D3D11_PRIMITIVE_TOPOLOGY>(state.topology);
		probe.candidate.api_query_heap = probe.api_query_heap;
		probe.candidate.api_query_index = probe.api_query_index;

		// Do not modify the application draw here. The center probe is replayed from
		// the after-draw callback, otherwise the original draw itself would be
		// scissored to one pixel and cause a visible flash in the game.
		return true;
	}

	void replay_d3d12_center_probe(command_list *cmd_list, probe_state &probe)
	{
		if (cmd_list == nullptr || !probe.is_d3d12 || probe.api_query_heap == 0 || probe.api_query_index == UINT32_MAX)
			return;

		const rect center_rect = { probe.api_center_x, probe.api_center_y, probe.api_center_x + 1, probe.api_center_y + 1 };
		cmd_list->bind_scissor_rects(0, 1, &center_rect);

		// Outside a native render pass, unbind application outputs for the replay.
		// This keeps the probe from writing color/depth while the occlusion query
		// still counts rasterized samples. Render-pass attachments cannot legally be
		// changed in the middle of a pass, so keep those bound in that case.
		if (!probe.api_in_render_pass)
			cmd_list->bind_render_targets_and_depth_stencil(0, nullptr, {});

		cmd_list->begin_query(probe.api_query_heap, query_type::occlusion, probe.api_query_index);
		probe.query_started = true;
		s_in_center_probe = true;
		cmd_list->draw_indexed(probe.candidate.index_count, probe.candidate.instance_count, probe.candidate.first_index, probe.candidate.vertex_offset, probe.candidate.first_instance);
		s_in_center_probe = false;
		cmd_list->end_query(probe.api_query_heap, query_type::occlusion, probe.api_query_index);
		probe.query_started = false;

		if (!probe.api_in_render_pass)
			cmd_list->bind_render_targets_and_depth_stencil(static_cast<uint32_t>(probe.old_render_targets.size()), probe.old_render_targets.data(), probe.api_old_depth_stencil);
		cmd_list->bind_scissor_rects(0, static_cast<uint32_t>(probe.api_old_scissors.size()), probe.api_old_scissors.data());
	}

	const char *topology_text(D3D11_PRIMITIVE_TOPOLOGY topology)
	{
		switch (topology)
		{
		case D3D11_PRIMITIVE_TOPOLOGY_POINTLIST: return "Point list";
		case D3D11_PRIMITIVE_TOPOLOGY_LINELIST: return "Line list";
		case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP: return "Line strip";
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST: return "Triangle list";
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: return "Triangle strip";
		case D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ: return "Line list + adjacency";
		case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ: return "Line strip + adjacency";
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ: return "Triangle list + adjacency";
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ: return "Triangle strip + adjacency";
		default:
			if (topology >= D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST && topology <= D3D11_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST)
				return "Patch list";
			return "Undefined/other";
		}
	}

	uint32_t estimate_primitive_count(D3D11_PRIMITIVE_TOPOLOGY topology, uint32_t index_count)
	{
		switch (topology)
		{
		case D3D11_PRIMITIVE_TOPOLOGY_POINTLIST: return index_count;
		case D3D11_PRIMITIVE_TOPOLOGY_LINELIST: return index_count / 2;
		case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP: return index_count > 1 ? index_count - 1 : 0;
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST: return index_count / 3;
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: return index_count > 2 ? index_count - 2 : 0;
		case D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ: return index_count / 4;
		case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ: return index_count > 3 ? index_count - 3 : 0;
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ: return index_count / 6;
		case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ: return index_count > 5 ? index_count - 5 : 0;
		default:
			if (topology >= D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST && topology <= D3D11_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST)
				return index_count / (static_cast<uint32_t>(topology) - static_cast<uint32_t>(D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST) + 1);
			return 0;
		}
	}

	bool begin_center_probe(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance, const draw_state &state, uint64_t draw_order, probe_state &probe)
	{
		if constexpr (!s_enable_native_center_probe)
			return false;
		if (cmd_list != nullptr && cmd_list->get_device()->get_api() == device_api::d3d12)
			return begin_d3d12_center_probe(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, state, draw_order, probe);
		if (cmd_list->get_device()->get_api() != device_api::d3d11)
			return false;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr)
			return false;
		// The picker temporarily changes rasterizer state around the draw. Never
		// mutate a deferred command list, since doing so can corrupt the recorded
		// command stream.
		if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
			return false;

		ID3D11Device *device = nullptr;
		context->GetDevice(&device);
		if (device == nullptr)
			return false;

		D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
		UINT viewport_count = ARRAYSIZE(viewports);
		context->RSGetViewports(&viewport_count, viewports);
		if (viewport_count == 0 || viewports[0].Width < 1.0f || viewports[0].Height < 1.0f)
		{
			device->Release();
			return false;
		}

		ID3D11RenderTargetView *current_render_target = nullptr;
		ID3D11DepthStencilView *current_depth_stencil_view = nullptr;
		context->OMGetRenderTargets(1, &current_render_target, &current_depth_stencil_view);
		// DX11 games often render geometry into an intermediate target, use a
		// depth-only pass, or bind no DSV for a color pass. The old implementation
		// required a DSV and an exact swap-chain-size match, which silently skipped
		// every draw in those games. A query only needs one active output target.
		const bool has_output_target = current_render_target != nullptr || current_depth_stencil_view != nullptr;
		if (current_render_target != nullptr) current_render_target->Release();
		if (current_depth_stencil_view != nullptr) current_depth_stencil_view->Release();
		if (!has_output_target)
		{
			device->Release();
			return false;
		}

		probe.context = context;
		probe.device = device;
		probe.candidate.index_count = index_count;
		probe.candidate.instance_count = instance_count;
		probe.candidate.first_index = first_index;
		probe.candidate.vertex_offset = vertex_offset;
		probe.candidate.first_instance = first_instance;

		const bool observer_compatibility = is_observer_compatibility_path();
		if (!observer_compatibility)
		{
			D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
			context->IAGetPrimitiveTopology(&topology);
			probe.candidate.topology = topology;
			capture_draw_match(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, probe.candidate.match);
		}

		context->RSGetState(&probe.old_rasterizer);
		D3D11_RASTERIZER_DESC rasterizer_desc = {};
		if (probe.old_rasterizer != nullptr)
			probe.old_rasterizer->GetDesc(&rasterizer_desc);
		else
		{
			rasterizer_desc.FillMode = D3D11_FILL_SOLID;
			rasterizer_desc.CullMode = D3D11_CULL_BACK;
			rasterizer_desc.DepthClipEnable = TRUE;
		}

		probe.old_scissor_count = ARRAYSIZE(probe.old_scissors);
		context->RSGetScissorRects(&probe.old_scissor_count, probe.old_scissors);
		LONG center_x = static_cast<LONG>(viewports[0].TopLeftX + viewports[0].Width * 0.5f);
		LONG center_y = static_cast<LONG>(viewports[0].TopLeftY + viewports[0].Height * 0.5f);
		center_x = std::max(static_cast<LONG>(viewports[0].TopLeftX), std::min(center_x, static_cast<LONG>(viewports[0].TopLeftX + viewports[0].Width) - 1));
		center_y = std::max(static_cast<LONG>(viewports[0].TopLeftY), std::min(center_y, static_cast<LONG>(viewports[0].TopLeftY + viewports[0].Height) - 1));
		if (rasterizer_desc.ScissorEnable && probe.old_scissor_count != 0)
		{
			bool inside_scissor = false;
			for (UINT i = 0; i < probe.old_scissor_count; ++i)
				inside_scissor |= center_x >= probe.old_scissors[i].left && center_x < probe.old_scissors[i].right && center_y >= probe.old_scissors[i].top && center_y < probe.old_scissors[i].bottom;
			if (!inside_scissor)
			{
				cleanup_probe_state(probe, false);
				return false;
			}
		}

		D3D11_QUERY_DESC query_desc = { D3D11_QUERY_OCCLUSION, 0 };
		if (FAILED(device->CreateQuery(&query_desc, &probe.candidate.query)))
		{
			cleanup_probe_state(probe, true);
			return false;
		}

		rasterizer_desc.ScissorEnable = TRUE;
		if (FAILED(device->CreateRasterizerState(&rasterizer_desc, &probe.probe_rasterizer)))
		{
			cleanup_probe_state(probe, true);
			return false;
		}

		if (!observer_compatibility)
		{
			context->OMGetDepthStencilState(&probe.old_depth_stencil, &probe.old_stencil_reference);
			D3D11_DEPTH_STENCIL_DESC depth_desc = {};
			if (probe.old_depth_stencil != nullptr)
				probe.old_depth_stencil->GetDesc(&depth_desc);
			// The picker is interested in geometry that covers the center, including
			// geometry hidden behind the currently visible layer. Keep the original
			// depth state for restoration, but disable depth/stencil tests for the
			// one-pixel query and disable all writes.
			depth_desc.DepthEnable = FALSE;
			depth_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			depth_desc.StencilEnable = FALSE;
			depth_desc.StencilWriteMask = 0;
			if (FAILED(device->CreateDepthStencilState(&depth_desc, &probe.probe_depth_stencil)))
			{
				cleanup_probe_state(probe, true);
				return false;
			}

			context->OMGetBlendState(&probe.old_blend, probe.old_blend_factor, &probe.old_sample_mask);
			D3D11_BLEND_DESC blend_desc = {};
			if (probe.old_blend != nullptr)
				probe.old_blend->GetDesc(&blend_desc);
			blend_desc.IndependentBlendEnable = TRUE;
			for (D3D11_RENDER_TARGET_BLEND_DESC &render_target : blend_desc.RenderTarget)
				render_target.RenderTargetWriteMask = 0;
			if (FAILED(device->CreateBlendState(&blend_desc, &probe.probe_blend)))
			{
				cleanup_probe_state(probe, true);
				return false;
			}
		}

		const D3D11_RECT center_rect = { center_x, center_y, center_x + 1, center_y + 1 };
		context->RSSetState(probe.probe_rasterizer);
		context->RSSetScissorRects(1, &center_rect);
		if (!observer_compatibility)
		{
			context->OMSetDepthStencilState(probe.probe_depth_stencil, probe.old_stencil_reference);
			context->OMSetBlendState(probe.probe_blend, probe.old_blend_factor, probe.old_sample_mask);
		}
		context->Begin(probe.candidate.query);
		probe.query_started = true;
		return true;
	}

	bool begin_center_probe_safe(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance, const draw_state &state, uint64_t draw_order, probe_state &probe)
	{
		// Unreal can expose transient DX11 objects while a draw is being issued.
		// Isolate native-driver faults here so a bad probe is treated as a miss.
		__try
		{
			return begin_center_probe(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, state, draw_order, probe);
		}
		__except (EXCEPTION_EXECUTE_HANDLER)
		{
			reshade::log::message(reshade::log::level::error, "Draw Filter: DX11 center probe faulted; skipping this draw.");
			cleanup_probe_state(probe, true);
			return false;
		}
	}

	void on_begin_debug_event(command_list *cmd_list, const char *label)
	{
		if (cmd_list == nullptr || label == nullptr || *label == '\0')
			return;

		std::lock_guard lock(s_marker_mutex);
		marker_state &state = s_marker_states[cmd_list];
		state.event_stack.emplace_back(label);
		state.last_marker.clear();
	}

	void on_end_debug_event(command_list *cmd_list)
	{
		if (cmd_list == nullptr)
			return;

		std::lock_guard lock(s_marker_mutex);
		const auto it = s_marker_states.find(cmd_list);
		if (it == s_marker_states.end() || it->second.event_stack.empty())
			return;
		it->second.event_stack.pop_back();
		if (it->second.event_stack.empty() && it->second.last_marker.empty())
			s_marker_states.erase(it);
	}

	void on_insert_debug_marker(command_list *cmd_list, const char *label)
	{
		if (cmd_list == nullptr)
			return;

		std::lock_guard lock(s_marker_mutex);
		marker_state &state = s_marker_states[cmd_list];
		state.last_marker = label != nullptr ? label : "";
	}

	std::string take_event_name(command_list *cmd_list)
	{
		if (cmd_list == nullptr)
			return {};

		std::lock_guard lock(s_marker_mutex);
		const auto it = s_marker_states.find(cmd_list);
		if (it == s_marker_states.end())
			return {};

		marker_state &state = it->second;
		std::string result;
		for (const std::string &label : state.event_stack)
		{
			if (label.empty())
				continue;
			if (!result.empty())
				result += " > ";
			result += label;
		}
		if (result.empty())
			result = state.last_marker;
		state.last_marker.clear();
		if (state.event_stack.empty() && state.last_marker.empty())
			s_marker_states.erase(it);
		return result;
	}

	void on_after_draw_indexed(command_list *cmd_list, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
	{
		probe_state probe;
		{
			std::lock_guard lock(s_probe_mutex);
			const auto it = s_active_probes.find(cmd_list);
			if (it == s_active_probes.end())
				return;
			probe = std::move(it->second);
			s_active_probes.erase(it);
		}

		if (probe.is_d3d12)
			replay_d3d12_center_probe(cmd_list, probe);

		pick_candidate candidate = probe.candidate;
		cleanup_probe_state(probe, false);
		std::lock_guard lock(s_pick_mutex);
		if (s_pick_active && !s_pick_finished)
			s_pending_pick.push_back(candidate);
		else
			release_query(candidate.query);
	}

	void enable_center_picker_after_draw_event()
	{
		if (!s_after_draw_event_registered.exchange(true, std::memory_order_acq_rel))
			reshade::register_event<reshade::addon_event::after_draw_indexed>(on_after_draw_indexed);
	}

	void disable_center_picker_after_draw_event()
	{
		if (s_after_draw_event_registered.exchange(false, std::memory_order_acq_rel))
			reshade::unregister_event<reshade::addon_event::after_draw_indexed>(on_after_draw_indexed);
	}

	void cleanup_active_probes()
	{
		std::vector<std::pair<command_list *, probe_state>> probes;
		{
			std::lock_guard lock(s_probe_mutex);
			probes.reserve(s_active_probes.size());
			for (auto &entry : s_active_probes)
				probes.emplace_back(entry.first, std::move(entry.second));
			s_active_probes.clear();
		}

		for (auto &entry : probes)
			cleanup_probe_state(entry.second, true);
	}

	void try_finish_pick(command_queue *queue)
	{
		std::lock_guard lock(s_pick_mutex);
		if (!s_pick_finished)
			return;

		device *queue_device = queue != nullptr ? queue->get_device() : nullptr;
		const device_api queue_api = queue_device != nullptr ? queue_device->get_api() : static_cast<device_api>(0);
		if (queue_device == nullptr || (queue_api != device_api::d3d11 && queue_api != device_api::d3d12))
		{
			s_pick_state = pick_status::unsupported;
			s_pick_active = false;
			s_pick_finished = false;
			return;
		}

		ID3D11DeviceContext *context = queue_api == device_api::d3d11 ? reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(queue->get_native())) : nullptr;
		if (queue_api == device_api::d3d11 && context == nullptr)
		{
			s_pick_state = pick_status::unsupported;
			s_pick_active = false;
			s_pick_finished = false;
			return;
		}

		if (s_pending_pick.empty())
		{
			s_pick_results.clear();
			s_pick_state = pick_status::no_result;
			s_pick_active = false;
			s_pick_finished = false;
			return;
		}

		if (queue_api == device_api::d3d11 && s_pick_wait_frames == 0)
			context->Flush();

		// The present queue is not guaranteed to be the same wrapped device that
		// recorded the command list (Observer creates more than one D3D12 device).
		// Query heaps are device-owned, so resolve them through the device that
		// created the active picker heap instead of blindly using the present queue.
		device *query_result_device = queue_device;
		if (queue_api == device_api::d3d12)
		{
			std::lock_guard query_lock(s_d3d12_query_mutex);
			if (s_d3d12_query_device != nullptr)
				query_result_device = s_d3d12_query_device;
		}

		bool all_ready = true;
		size_t ready_count = 0;
		size_t nonzero_count = 0;
		uint64_t max_samples = 0;
		std::vector<uint64_t> d3d12_query_results;
		query_heap d3d12_query_heap = {};
		bool d3d12_query_batch_ready = false;
		if (queue_api == device_api::d3d12 && query_result_device != nullptr)
		{
			uint32_t max_query_index = 0;
			bool same_query_heap = true;
			for (const pick_candidate &candidate : s_pending_pick)
			{
				if (candidate.api_query_heap == 0)
				{
					same_query_heap = false;
					break;
				}
				if (d3d12_query_heap == 0)
					d3d12_query_heap = candidate.api_query_heap;
				else if (d3d12_query_heap != candidate.api_query_heap)
					same_query_heap = false;
				max_query_index = std::max(max_query_index, candidate.api_query_index);
			}
			if (same_query_heap && d3d12_query_heap != 0)
			{
				d3d12_query_results.resize(static_cast<size_t>(max_query_index) + 1);
				d3d12_query_batch_ready = query_result_device->get_query_heap_results(d3d12_query_heap, query_type::occlusion, 0, static_cast<uint32_t>(d3d12_query_results.size()), d3d12_query_results.data(), sizeof(uint64_t));
			}
		}
		for (pick_candidate &candidate : s_pending_pick)
		{
			uint64_t samples = 0;
			bool ready = false;
			if (queue_api == device_api::d3d11)
			{
				const HRESULT result = context->GetData(candidate.query, &samples, sizeof(samples), D3D11_ASYNC_GETDATA_DONOTFLUSH);
				ready = result != S_FALSE;
				if (ready)
					samples = SUCCEEDED(result) ? samples : 0;
			}
			else
			{
				ready = d3d12_query_batch_ready && candidate.api_query_index < d3d12_query_results.size();
				if (ready)
					samples = d3d12_query_results[candidate.api_query_index];
			}
			if (!ready)
			{
				all_ready = false;
				continue;
			}
			++ready_count;
			max_samples = std::max(max_samples, samples);
			if (samples != 0)
				++nonzero_count;
			candidate.samples_passed = samples;
		}

		if (!all_ready && s_pick_wait_frames++ < 16)
			return;

		std::vector<pick_candidate> visible;
		for (pick_candidate &candidate : s_pending_pick)
			if (candidate.samples_passed != 0)
				visible.push_back(candidate);

		std::sort(visible.begin(), visible.end(), [](const pick_candidate &lhs, const pick_candidate &rhs) { return lhs.draw_order > rhs.draw_order; });

		s_pick_results.clear();
		const uint32_t result_limit = std::max(1u, s_pick_limit.load(std::memory_order_relaxed));
		for (pick_candidate &candidate : visible)
		{
			if (s_pick_results.size() >= result_limit)
				break;
			candidate.query = nullptr;
			candidate.api_query_heap = {};
			candidate.api_query_index = UINT32_MAX;
			s_pick_results.push_back(candidate);
		}

		const size_t queried_count = s_pending_pick.size();
		for (pick_candidate &candidate : s_pending_pick)
			release_query(candidate.query);
		s_pending_pick.clear();
		char pick_log[192] = {};
		std::snprintf(pick_log, sizeof(pick_log), "Draw Filter center pick: %llu queried, %llu ready, %llu nonzero, max samples %llu, %llu visible, %llu returned.",
			static_cast<unsigned long long>(queried_count), static_cast<unsigned long long>(ready_count), static_cast<unsigned long long>(nonzero_count),
			static_cast<unsigned long long>(max_samples), static_cast<unsigned long long>(visible.size()), static_cast<unsigned long long>(s_pick_results.size()));
		reshade::log::message(reshade::log::level::info, pick_log);
		s_pick_state = s_pick_results.empty() ? pick_status::no_result : pick_status::ready;
		s_pick_active = false;
		s_pick_finished = false;
		s_pick_wait_frames = 0;
	}

	bool on_draw_indexed(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
	{
		if (s_in_partial_depth_rewrite)
			return false;
		if (s_in_center_probe)
			return false;

		bool blocked = false;
		bool partial_depth_rewrite_requested = false;
		draw_match partial_depth_match;
		std::vector<depth_mesh_target> partial_depth_targets;
		if (s_enabled.load(std::memory_order_relaxed))
		{
			bool globally_blocked = false;
			bool need_match = false;
			{
				std::shared_lock lock(s_mutex);
				globally_blocked = std::find(s_blocked_index_counts.cbegin(), s_blocked_index_counts.cend(), index_count) != s_blocked_index_counts.cend();
				need_match = globally_blocked || !s_blocked_draws.empty() || !s_depth_mesh_targets.empty();
			}

			draw_match current_match;
			bool current_match_valid = false;
			if (need_match)
				current_match_valid = capture_draw_match(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, current_match);

			{
				std::shared_lock lock(s_mutex);
				blocked = globally_blocked;
				if (!blocked && current_match_valid && !s_blocked_draws.empty())
					blocked = std::find_if(s_blocked_draws.cbegin(), s_blocked_draws.cend(), [&current_match](const draw_match &item) { return same_draw_match(item, current_match); }) != s_blocked_draws.cend();
			}

			// Learn the actual vertex footprint of every filtered visible draw.
			// Its depth-only pass may use another index buffer, so matching only by
			// vertex-buffer identity is not sufficient.
			if (blocked && current_match_valid && !is_depth_only_draw(current_match))
				add_depth_mesh_target(cmd_list, current_match);

			// Do not skip an entire aggregated UE4 Early-Z draw. It can contain
			// other mesh sections that share the same vertex buffer and use another
			// index buffer. Replay it with only triangles whose vertex footprint
			// belongs to the filtered target mesh replaced by degenerate triangles.
			if (!blocked && current_match_valid && is_depth_only_draw(current_match) && get_depth_targets_for_draw(current_match, partial_depth_targets))
			{
				partial_depth_rewrite_requested = true;
				partial_depth_match = current_match;
			}
		}

		if (partial_depth_rewrite_requested && draw_depth_without_target_geometry(cmd_list, partial_depth_match, partial_depth_targets))
		{
			s_skipped_draws.fetch_add(1, std::memory_order_relaxed);
			return true;
		}

		// Keep the normal filtering path identical to the original add-on. The
		// center picker is a one-shot diagnostic operation and must not add any
		// state/event work to ordinary filtered draws.
		bool picker_in_progress = false;
		{
			std::lock_guard lock(s_pick_mutex);
			picker_in_progress = s_pick_armed || s_pick_active;
		}
		if (!picker_in_progress)
		{
			if (blocked)
			{
				s_skipped_draws.fetch_add(1, std::memory_order_relaxed);
				return true;
			}
			return false;
		}

		bool scan_this_draw = false;
		uint64_t draw_order = 0;
		{
			std::lock_guard lock(s_pick_mutex);
			if (s_pick_armed)
			{
				s_pick_armed = false;
				s_pick_active = true;
				s_pick_finished = false;
				s_pick_wait_frames = 0;
				s_pick_draw_order = 0;
				s_pending_pick.clear();
				s_pick_state = pick_status::scanning;
				reshade::log::message(reshade::log::level::info, "Draw Filter center scan started.");
			}
			if (s_pick_active && !s_pick_finished)
			{
				scan_this_draw = true;
				draw_order = s_pick_draw_order++;
			}
		}

		std::string event_name;
		if (scan_this_draw || blocked)
			event_name = take_event_name(cmd_list);

		if (scan_this_draw && !blocked)
		{
			draw_state state;
			{
				std::lock_guard lock(s_state_mutex);
				const auto it = s_command_states.find(cmd_list);
				if (it != s_command_states.end())
					state = it->second;
			}

			probe_state probe;
			if (begin_center_probe_safe(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, state, draw_order, probe))
			{
				probe.candidate.draw_order = draw_order;
				probe.candidate.event_name = std::move(event_name);
				std::lock_guard lock(s_probe_mutex);
				if (s_active_probes.find(cmd_list) == s_active_probes.end())
					s_active_probes.emplace(cmd_list, std::move(probe));
				else
					cleanup_probe_state(probe, true);
			}
		}

		if (blocked)
		{
			s_skipped_draws.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		if (!scan_this_draw)
			event_name.clear();

		return false;
	}

	void on_present(command_queue *, swapchain *, const rect *, const rect *, uint32_t, const rect *)
	{
		// Some UE4 deferred/immediate-context paths do not emit the custom
		// after_draw_indexed callback. Never let the one-pixel center probe leave
		// its temporary D3D11 state active into the next frame.
		cleanup_active_probes();

		{
			std::lock_guard lock(s_pick_mutex);
			if (s_pick_active && !s_pick_finished)
			{
				s_pick_finished = true;
				s_pick_state = pick_status::waiting;
				reshade::log::message(reshade::log::level::info, "Draw Filter center scan frame finished.");
			}
		}
		// No more application draw calls are expected after this point in the
		// frame. Keep the normal filter path free of the after-draw callback while
		// the GPU query is being resolved.
		disable_center_picker_after_draw_event();
	}

	void on_finish_present(command_queue *queue, swapchain *)
	{
		// Arm only after the present has finished. Otherwise clicking the button
		// can make the first indexed draw of ReShade's own overlay become the probe
		// draw, which is unsafe for Unreal's DX11 context.
		{
			std::lock_guard lock(s_pick_mutex);
			if (s_pick_requested.exchange(false, std::memory_order_relaxed))
			{
				s_pick_armed = true;
				s_pick_state = pick_status::armed;
			}
		}
		try_finish_pick(queue);
	}

	void on_destroy_device(device *)
	{
		cleanup_active_probes();
		destroy_isolated_discard_pixel_shader();

		std::lock_guard pick_lock(s_pick_mutex);
		for (pick_candidate &candidate : s_pending_pick)
			release_query(candidate.query);
		s_pending_pick.clear();
		s_pick_results.clear();
		s_pick_armed = false;
		s_pick_active = false;
		s_pick_finished = false;
		{
			std::unique_lock filter_lock(s_mutex);
			s_depth_mesh_targets.clear();
		}
		clear_partial_depth_buffers();
		destroy_d3d12_query_heap();

		std::lock_guard state_lock(s_state_mutex);
		s_command_states.clear();
		s_swapchain_back_buffers.clear();
	}

	void draw_settings_overlay(effect_runtime *runtime)
	{
		{
			std::lock_guard lock(s_pick_mutex);
			if (s_pick_active && !s_pick_finished)
			{
				s_pick_finished = true;
				s_pick_state = pick_status::waiting;
			}
		}
		bool pick_in_progress = false;
		{
			std::lock_guard lock(s_pick_mutex);
			pick_in_progress = s_pick_active || s_pick_armed || s_pick_requested.load(std::memory_order_relaxed);
		}
		if (runtime != nullptr && pick_in_progress)
		{
			// Do not query the current back buffer on every overlay draw. Some UE4
			// render paths expose transient resources here, so only inspect it for
			// an explicitly requested center-pick operation.
			const resource_desc back_buffer_desc = runtime->get_device()->get_resource_desc(runtime->get_current_back_buffer());
			s_pick_target_width.store(back_buffer_desc.texture.width, std::memory_order_relaxed);
			s_pick_target_height.store(back_buffer_desc.texture.height, std::memory_order_relaxed);
			s_pick_target_format.store(static_cast<uint32_t>(back_buffer_desc.texture.format), std::memory_order_relaxed);
			try_finish_pick(runtime->get_command_queue());
		}

		bool enabled = s_enabled.load(std::memory_order_relaxed);
		if (ImGui::Checkbox("Enable draw filter", &enabled))
			s_enabled.store(enabled, std::memory_order_relaxed);

		ImGui::TextUnformatted("Match field: draw signature or index_count");
		ImGui::SetItemTooltip("'Add to filter' from a center result matches one exact draw. Manual index input keeps the legacy global index_count behavior.");

		const uint32_t input_step = 1;
		const uint32_t input_step_fast = 100;
		const bool submitted = ImGui::InputScalar("Index count", ImGuiDataType_U32, &s_input_index_count, &input_step, &input_step_fast, "%u", ImGuiInputTextFlags_EnterReturnsTrue);
		ImGui::SameLine();
		const bool add_clicked = ImGui::Button("Add");
		if ((submitted || add_clicked) && s_input_index_count != 0)
		{
			add_index_count(s_input_index_count);
			s_input_index_count = 0;
		}

		ImGui::Text("Skipped draws this session: %llu", static_cast<unsigned long long>(s_skipped_draws.load(std::memory_order_relaxed)));
		uint32_t pick_limit = s_pick_limit.load(std::memory_order_relaxed);
		const uint32_t pick_limit_step = 1;
		const uint32_t pick_limit_step_fast = 32;
		if (ImGui::InputScalar("Max center results", ImGuiDataType_U32, &pick_limit, &pick_limit_step, &pick_limit_step_fast, "%u"))
			s_pick_limit.store(std::max(1u, pick_limit), std::memory_order_relaxed);
		ImGui::SetItemTooltip("Maximum number of indexed draw records returned by the one-shot center picker. Default is 32; enter a larger value when needed.");

		if constexpr (!s_enable_native_center_probe)
		{
			ImGui::TextDisabled("Center picker temporarily disabled for isolation test.");
		}
		else
		{
			pick_status center_pick_state;
			std::vector<pick_candidate> center_results;
			{
				std::lock_guard lock(s_pick_mutex);
				center_pick_state = s_pick_state;
				center_results = s_pick_results;
			}

			if (center_pick_state == pick_status::scanning || center_pick_state == pick_status::waiting || center_pick_state == pick_status::armed)
				ImGui::TextDisabled("Get center object index: %s", pick_status_text(center_pick_state));
			else if (ImGui::Button("Get center object index"))
			{
				bool arm_picker = false;
				{
					std::lock_guard lock(s_pick_mutex);
					if (!s_pick_active && !s_pick_armed && !s_pick_requested.load(std::memory_order_relaxed))
					{
						s_pick_results.clear();
						s_pending_pick.clear();
						s_pick_requested.store(true, std::memory_order_relaxed);
						s_pick_state = pick_status::armed;
						arm_picker = true;
					}
				}
				if (arm_picker)
					enable_center_picker_after_draw_event();
			}

			ImGui::SameLine();
			ImGui::TextDisabled("(%s)", pick_status_text(center_pick_state));

			if (!center_results.empty())
			{
				ImGui::Separator();
				ImGui::TextUnformatted("Center candidates (latest draw first):");
				for (size_t i = 0; i < center_results.size(); ++i)
				{
					const pick_candidate &candidate = center_results[i];
					ImGui::PushID(static_cast<int>(i));
					ImGui::Text("Draw #%llu  Key=%016llX  index_count=%u  instances=%u", static_cast<unsigned long long>(candidate.draw_order), static_cast<unsigned long long>(draw_signature_key(candidate.match)), candidate.index_count, candidate.instance_count);
					ImGui::TextDisabled("%s  |  Est. primitives: %u", topology_text(candidate.topology), estimate_primitive_count(candidate.topology, candidate.index_count));
					ImGui::TextDisabled("Event: %s", candidate.event_name.empty() ? "No GPU event marker" : candidate.event_name.c_str());
					ImGui::SameLine();
					if (ImGui::SmallButton("Add to filter"))
					{
						if (candidate.match.valid)
							add_draw_match(candidate.match);
						else
							add_index_count(candidate.index_count);
					}
					ImGui::PopID();
				}
			}
		}

		std::vector<uint32_t> blocked;
		std::vector<draw_match> exact_blocked;
		{
			std::shared_lock lock(s_mutex);
			blocked = s_blocked_index_counts;
			exact_blocked = s_blocked_draws;
		}

		if (blocked.empty() && exact_blocked.empty())
		{
			ImGui::TextDisabled("No draw calls are blocked.");
		}
		if (!blocked.empty())
		{
			ImGui::Separator();
			ImGui::TextUnformatted("Global blocked index counts:");
			for (const uint32_t index_count : blocked)
			{
				ImGui::PushID(static_cast<int>(index_count));
				ImGui::Text("%u", index_count);
				ImGui::SameLine();
				if (ImGui::SmallButton("Remove"))
				{
					std::unique_lock lock(s_mutex);
					s_blocked_index_counts.erase(
						std::remove(s_blocked_index_counts.begin(), s_blocked_index_counts.end(), index_count),
						s_blocked_index_counts.end());
					s_depth_mesh_targets.erase(
						std::remove_if(s_depth_mesh_targets.begin(), s_depth_mesh_targets.end(), [index_count](const depth_mesh_target &target) { return target.match.index_count == index_count; }),
						s_depth_mesh_targets.end());
					clear_partial_depth_buffers();
				}
				ImGui::PopID();
			}

		}

		if (!exact_blocked.empty())
		{
			ImGui::Separator();
			ImGui::TextUnformatted("Exact draw matches:");
			for (size_t i = 0; i < exact_blocked.size(); ++i)
			{
				const draw_match &match = exact_blocked[i];
				ImGui::PushID(static_cast<int>(i));
				ImGui::Text("Key=%016llX  index_count=%u  instances=%u", static_cast<unsigned long long>(draw_signature_key(match)), match.index_count, match.instance_count);
				ImGui::TextDisabled("RT=%u  Depth=%u  Topology=%u", match.render_target_format, match.depth_stencil_format, static_cast<unsigned int>(match.topology));
				ImGui::SameLine();
				if (ImGui::SmallButton("Remove"))
				{
					std::unique_lock lock(s_mutex);
					if (i < s_blocked_draws.size())
					{
						const draw_match removed_match = s_blocked_draws[i];
						s_blocked_draws.erase(s_blocked_draws.begin() + i);
						s_depth_mesh_targets.erase(
							std::remove_if(s_depth_mesh_targets.begin(), s_depth_mesh_targets.end(), [&removed_match](const depth_mesh_target &target) { return same_draw_match(target.match, removed_match); }),
							s_depth_mesh_targets.end());
						clear_partial_depth_buffers();
					}
				}
				ImGui::PopID();
			}
		}

		if (!blocked.empty() || !exact_blocked.empty())
		{
			if (ImGui::Button("Clear all"))
			{
				std::unique_lock lock(s_mutex);
				s_blocked_index_counts.clear();
				s_blocked_draws.clear();
				s_depth_mesh_targets.clear();
				clear_partial_depth_buffers();
			}
		}
	}
}

void register_addon_draw_filter()
{
	reshade::register_overlay(nullptr, draw_settings_overlay);
	reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
	reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_render_targets_and_depth_stencil);
	reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_render_pass);
	reshade::register_event<reshade::addon_event::end_render_pass>(on_end_render_pass);
	reshade::register_event<reshade::addon_event::bind_pipeline_states>(on_bind_pipeline_states);
	reshade::register_event<reshade::addon_event::bind_viewports>(on_bind_viewports);
	reshade::register_event<reshade::addon_event::bind_scissor_rects>(on_bind_scissor_rects);
	reshade::register_event<reshade::addon_event::begin_debug_event>(on_begin_debug_event);
	reshade::register_event<reshade::addon_event::end_debug_event>(on_end_debug_event);
	reshade::register_event<reshade::addon_event::insert_debug_marker>(on_insert_debug_marker);
	reshade::register_event<reshade::addon_event::present>(on_present);
	reshade::register_event<reshade::addon_event::finish_present>(on_finish_present);
	reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
}

void unregister_addon_draw_filter()
{
	disable_center_picker_after_draw_event();
	reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
	reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_render_targets_and_depth_stencil);
	reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_render_pass);
	reshade::unregister_event<reshade::addon_event::end_render_pass>(on_end_render_pass);
	reshade::unregister_event<reshade::addon_event::bind_pipeline_states>(on_bind_pipeline_states);
	reshade::unregister_event<reshade::addon_event::bind_viewports>(on_bind_viewports);
	reshade::unregister_event<reshade::addon_event::bind_scissor_rects>(on_bind_scissor_rects);
	reshade::unregister_event<reshade::addon_event::begin_debug_event>(on_begin_debug_event);
	reshade::unregister_event<reshade::addon_event::end_debug_event>(on_end_debug_event);
	reshade::unregister_event<reshade::addon_event::insert_debug_marker>(on_insert_debug_marker);
	reshade::unregister_event<reshade::addon_event::present>(on_present);
	reshade::unregister_event<reshade::addon_event::finish_present>(on_finish_present);
	reshade::unregister_event<reshade::addon_event::destroy_device>(on_destroy_device);
}

#ifndef BUILTIN_ADDON

extern "C" __declspec(dllexport) const char *NAME = "Draw Call Filter";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "Blocks indexed draw calls by exact index count for offline rendering diagnostics.";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		if (!reshade::register_addon(hModule))
			return FALSE;
		register_addon_draw_filter();
		break;
	case DLL_PROCESS_DETACH:
		unregister_addon_draw_filter();
		reshade::unregister_addon(hModule);
		break;
	}

	return TRUE;
}

#endif
