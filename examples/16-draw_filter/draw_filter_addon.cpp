/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: BSD-3-Clause OR MIT
 */

#include <imgui.h>
#include <reshade.hpp>
#include <d3d11.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
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
	std::atomic_bool s_enabled = true;
	std::atomic_uint64_t s_skipped_draws = 0;
	int s_input_index_count = 0;
	std::atomic_int s_pick_limit = 8;
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
		pipeline current_pipeline = {};
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
	};

	struct marker_state
	{
		std::vector<std::string> event_stack;
		std::string last_marker;
	};

	struct probe_state
	{
		pick_candidate candidate = {};
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

	struct filter_suppression
	{
		ID3D11DeviceContext *context = nullptr;
		ID3D11DepthStencilState *old_depth_stencil = nullptr;
		ID3D11DepthStencilState *suppressed_depth_stencil = nullptr;
		UINT old_stencil_reference = 0;
		ID3D11BlendState *old_blend = nullptr;
		ID3D11BlendState *suppressed_blend = nullptr;
		FLOAT old_blend_factor[4] = {};
		UINT old_sample_mask = 0;
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
	std::mutex s_filter_suppression_mutex;
	std::unordered_map<command_list *, filter_suppression> s_active_filter_suppressions;
	std::mutex s_marker_mutex;
	std::unordered_map<command_list *, marker_state> s_marker_states;
	std::set<resource> s_swapchain_back_buffers;
	std::unordered_map<command_list *, draw_state> s_command_states;
	std::atomic_bool s_after_draw_event_registered = false;
	std::atomic_uint s_special_draw_trace_count = 0;
	thread_local bool s_in_center_probe = false;

	void add_draw_match(const draw_match &match);

	const char *pick_status_text(const pick_status status)
	{
		switch (status)
		{
		case pick_status::armed: return "Waiting for next frame";
		case pick_status::scanning: return "Scanning screen center";
		case pick_status::waiting: return "Waiting for GPU result";
		case pick_status::ready: return "Results ready";
		case pick_status::no_result: return "No indexed draw covered the center";
		case pick_status::unsupported: return "Center picking requires D3D11";
		case pick_status::disabled: return "Disabled for D3D11 stability";
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

	void on_bind_render_targets_and_depth_stencil(command_list *cmd_list, uint32_t count, const resource_view *rtvs, resource_view)
	{
		std::lock_guard lock(s_state_mutex);
		draw_state &state = s_command_states[cmd_list];
		state.render_target = count != 0 && rtvs != nullptr ? cmd_list->get_device()->get_resource_from_view(rtvs[0]) : resource {};
	}

	void on_bind_pipeline(command_list *cmd_list, pipeline_stage, pipeline pipeline)
	{
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

	void add_draw_match(const draw_match &match)
	{
		if (!match.valid)
			return;

		std::unique_lock lock(s_mutex);
		if (std::find_if(s_blocked_draws.cbegin(), s_blocked_draws.cend(), [&match](const draw_match &item) { return same_geometry_match(item, match); }) == s_blocked_draws.cend())
			s_blocked_draws.push_back(match);
	}

	void add_draw_match(const pick_candidate &candidate)
	{
		add_draw_match(candidate.match);
	}

	void trace_special_draw(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
	{
		if (index_count != 288 && index_count != 6)
			return;
		if (s_special_draw_trace_count.fetch_add(1, std::memory_order_relaxed) >= 96)
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

		char trace[640] = {};
		std::snprintf(trace, sizeof(trace),
			"Draw Filter trace: index=%u inst=%u first=%u vertex=%d first_inst=%u ctx=%s topo=%u ib=%p ib_fmt=%u ib_off=%u vs=%p ps=%p rtv_fmt=%u dsv_fmt=%u viewport=%.0fx%.0f",
			index_count, instance_count, first_index, vertex_offset, first_instance, context_type,
			static_cast<unsigned int>(topology), static_cast<void *>(index_buffer), static_cast<unsigned int>(index_format), index_offset,
			static_cast<void *>(vertex_shader), static_cast<void *>(pixel_shader), static_cast<unsigned int>(render_target_desc.Format),
			static_cast<unsigned int>(depth_stencil_desc.Format), viewport.Width, viewport.Height);
		reshade::log::message(reshade::log::level::info, trace);

		if (render_target != nullptr) render_target->Release();
		if (depth_stencil != nullptr) depth_stencil->Release();
		if (vertex_shader != nullptr) vertex_shader->Release();
		if (pixel_shader != nullptr) pixel_shader->Release();
		if (index_buffer != nullptr) index_buffer->Release();
	}

	void cleanup_probe_state(probe_state &probe, bool release_query_object)
	{
		if (probe.context != nullptr)
		{
			if (probe.query_started && probe.candidate.query != nullptr)
				probe.context->End(probe.candidate.query);
			probe.query_started = false;
			probe.context->OMSetBlendState(probe.old_blend, probe.old_blend_factor, probe.old_sample_mask);
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

	bool begin_center_probe(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance, const draw_state &state, probe_state &probe)
	{
		if constexpr (!s_enable_native_center_probe)
			return false;
		if (cmd_list->get_device()->get_api() != device_api::d3d11)
			return false;

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(cmd_list->get_native()));
		if (context == nullptr)
			return false;
		// The picker temporarily changes rasterizer/output-merger state and
		// replays the draw in the after_draw callback. That is only safe on the
		// immediate context. Never mutate a deferred command list, since doing
		// so can corrupt the application's recorded command stream.
		if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
			return false;

		ID3D11Device *device = nullptr;
		context->GetDevice(&device);
		if (device == nullptr)
			return false;

		D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
		UINT viewport_count = ARRAYSIZE(viewports);
		context->RSGetViewports(&viewport_count, viewports);
		const uint32_t target_width = s_pick_target_width.load(std::memory_order_relaxed);
		const uint32_t target_height = s_pick_target_height.load(std::memory_order_relaxed);
		if (viewport_count == 0 || viewports[0].Width < 1.0f || viewports[0].Height < 1.0f || target_width == 0 || target_height == 0 ||
			static_cast<uint32_t>(viewports[0].Width + 0.5f) != target_width || static_cast<uint32_t>(viewports[0].Height + 0.5f) != target_height)
		{
			device->Release();
			return false;
		}

		ID3D11RenderTargetView *current_render_target = nullptr;
		ID3D11DepthStencilView *current_depth_stencil_view = nullptr;
		context->OMGetRenderTargets(1, &current_render_target, &current_depth_stencil_view);
		ID3D11Resource *current_resource = nullptr;
		ID3D11Texture2D *current_texture = nullptr;
		D3D11_TEXTURE2D_DESC current_desc = {};
		if (current_render_target != nullptr)
		{
			current_render_target->GetResource(&current_resource);
			if (current_resource != nullptr)
				current_resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&current_texture));
			if (current_texture != nullptr)
				current_texture->GetDesc(&current_desc);
		}
		// The scene is often rendered into an HDR/intermediate target before
		// UE4 composites it into the swap-chain back buffer. Only matching the
		// final back-buffer format therefore finds the full-screen triangle.
		// Require a depth target and matching dimensions instead, which keeps
		// scene geometry while excluding the usual post-processing pass.
		const bool target_matches = current_depth_stencil_view != nullptr && current_texture != nullptr && current_desc.Width == target_width && current_desc.Height == target_height;
		if (current_texture != nullptr) current_texture->Release();
		if (current_resource != nullptr) current_resource->Release();
		if (current_render_target != nullptr) current_render_target->Release();
		if (current_depth_stencil_view != nullptr) current_depth_stencil_view->Release();
		if (!target_matches)
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

		D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		context->IAGetPrimitiveTopology(&topology);
		probe.candidate.topology = topology;
		capture_draw_match(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, probe.candidate.match);

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

		const D3D11_RECT center_rect = { center_x, center_y, center_x + 1, center_y + 1 };
		context->RSSetState(probe.probe_rasterizer);
		context->RSSetScissorRects(1, &center_rect);
		context->OMSetDepthStencilState(probe.probe_depth_stencil, probe.old_stencil_reference);
		context->OMSetBlendState(probe.probe_blend, probe.old_blend_factor, probe.old_sample_mask);
		context->Begin(probe.candidate.query);
		probe.query_started = true;
		return true;
	}

	void replay_probe_draw(command_list *cmd_list, const pick_candidate &candidate)
	{
		if (cmd_list == nullptr || candidate.index_count == 0 || candidate.instance_count == 0)
			return;

		// The probe draw deliberately has color/depth writes disabled. Replay the
		// exact indexed draw after restoring the application's state so picking
		// does not remove or darken geometry in the captured frame.
		s_in_center_probe = true;
		cmd_list->draw_indexed(
			candidate.index_count,
			candidate.instance_count,
			candidate.first_index,
			candidate.vertex_offset,
			candidate.first_instance);
		s_in_center_probe = false;
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

		pick_candidate candidate = probe.candidate;
		cleanup_probe_state(probe, false);
		replay_probe_draw(cmd_list, candidate);
		std::lock_guard lock(s_pick_mutex);
		if (s_pick_active && !s_pick_finished && s_pending_pick.size() < 4096)
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

		if (queue == nullptr || queue->get_device()->get_api() != device_api::d3d11)
		{
			s_pick_state = pick_status::unsupported;
			s_pick_active = false;
			s_pick_finished = false;
			return;
		}

		ID3D11DeviceContext *context = reinterpret_cast<ID3D11DeviceContext *>(static_cast<uintptr_t>(queue->get_native()));
		if (context == nullptr)
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

		if (s_pick_wait_frames == 0)
			context->Flush();

		bool all_ready = true;
		for (pick_candidate &candidate : s_pending_pick)
		{
			UINT64 samples = 0;
			const HRESULT result = context->GetData(candidate.query, &samples, sizeof(samples), D3D11_ASYNC_GETDATA_DONOTFLUSH);
			if (result == S_FALSE)
			{
				all_ready = false;
				continue;
			}
			candidate.samples_passed = SUCCEEDED(result) ? samples : 0;
		}

		if (!all_ready && s_pick_wait_frames++ < 4)
			return;

		std::vector<pick_candidate> visible;
		for (pick_candidate &candidate : s_pending_pick)
			if (candidate.samples_passed != 0)
				visible.push_back(candidate);

		std::sort(visible.begin(), visible.end(), [](const pick_candidate &lhs, const pick_candidate &rhs) { return lhs.draw_order > rhs.draw_order; });

		const uint32_t result_limit = static_cast<uint32_t>(std::clamp(s_pick_limit.load(std::memory_order_relaxed), 1, 32));
		s_pick_results.clear();
		for (pick_candidate &candidate : visible)
		{
			candidate.query = nullptr;
			s_pick_results.push_back(candidate);
			if (s_pick_results.size() >= result_limit)
				break;
		}

		const size_t queried_count = s_pending_pick.size();
		for (pick_candidate &candidate : s_pending_pick)
			release_query(candidate.query);
		s_pending_pick.clear();
		char pick_log[192] = {};
		std::snprintf(pick_log, sizeof(pick_log), "Draw Filter center pick: %llu queried, %llu visible, %llu returned.", static_cast<unsigned long long>(queried_count), static_cast<unsigned long long>(visible.size()), static_cast<unsigned long long>(s_pick_results.size()));
		reshade::log::message(reshade::log::level::info, pick_log);
		s_pick_state = s_pick_results.empty() ? pick_status::no_result : pick_status::ready;
		s_pick_active = false;
		s_pick_finished = false;
		s_pick_wait_frames = 0;
	}

	bool on_draw_indexed(command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
	{
		if (s_in_center_probe)
			return false;

		trace_special_draw(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance);

		bool blocked = false;
		if (s_enabled.load(std::memory_order_relaxed))
		{
			std::shared_lock lock(s_mutex);
			blocked = std::find(s_blocked_index_counts.cbegin(), s_blocked_index_counts.cend(), index_count) != s_blocked_index_counts.cend();
			if (!blocked && !s_blocked_draws.empty())
			{
				draw_match current_match;
				if (capture_draw_match(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, current_match))
					blocked = std::find_if(s_blocked_draws.cbegin(), s_blocked_draws.cend(), [&current_match](const draw_match &item) { return same_geometry_match(item, current_match); }) != s_blocked_draws.cend();
			}
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
			if (begin_center_probe(cmd_list, index_count, instance_count, first_index, vertex_offset, first_instance, state, probe))
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
			if (s_pick_requested.exchange(false, std::memory_order_relaxed))
			{
				s_pick_armed = true;
				s_pick_state = pick_status::armed;
			}

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
		try_finish_pick(queue);
	}

	void on_destroy_device(device *)
	{
		cleanup_active_probes();

		std::lock_guard pick_lock(s_pick_mutex);
		for (pick_candidate &candidate : s_pending_pick)
			release_query(candidate.query);
		s_pending_pick.clear();
		s_pick_results.clear();
		s_pick_armed = false;
		s_pick_active = false;
		s_pick_finished = false;

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
			pick_in_progress = s_pick_active || s_pick_armed;
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

		const bool submitted = ImGui::InputInt("Index count", &s_input_index_count, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);
		ImGui::SameLine();
		const bool add_clicked = ImGui::Button("Add");
		if ((submitted || add_clicked) && s_input_index_count > 0)
		{
			add_index_count(static_cast<uint32_t>(s_input_index_count));
			s_input_index_count = 0;
		}

		ImGui::Text("Skipped draws this session: %llu", static_cast<unsigned long long>(s_skipped_draws.load(std::memory_order_relaxed)));
		int pick_limit = s_pick_limit.load(std::memory_order_relaxed);
		if (ImGui::InputInt("Max center results", &pick_limit, 1, 4))
			s_pick_limit.store(std::clamp(pick_limit, 1, 32), std::memory_order_relaxed);
		ImGui::SetItemTooltip("Maximum number of indexed draw records returned by the one-shot center picker.");

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
					if (!s_pick_active && !s_pick_armed)
					{
						s_pick_results.clear();
						s_pending_pick.clear();
						s_pick_requested.store(false, std::memory_order_relaxed);
						s_pick_armed = true;
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
					ImGui::Text("#%llu  index_count=%u  instances=%u", static_cast<unsigned long long>(candidate.draw_order), candidate.index_count, candidate.instance_count);
					ImGui::TextDisabled("%s  |  Est. primitives: %u", topology_text(candidate.topology), estimate_primitive_count(candidate.topology, candidate.index_count));
					ImGui::TextDisabled("Event: %s", candidate.event_name.empty() ? "No GPU event marker" : candidate.event_name.c_str());
					ImGui::SameLine();
					if (ImGui::SmallButton("Add to filter"))
						add_draw_match(candidate);
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
				ImGui::Text("index_count=%u  instances=%u", match.index_count, match.instance_count);
				ImGui::TextDisabled("RT=%u  Depth=%u  Topology=%u", match.render_target_format, match.depth_stencil_format, static_cast<unsigned int>(match.topology));
				ImGui::SameLine();
				if (ImGui::SmallButton("Remove"))
				{
					std::unique_lock lock(s_mutex);
					if (i < s_blocked_draws.size())
						s_blocked_draws.erase(s_blocked_draws.begin() + i);
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
			}
		}
	}
}

void register_addon_draw_filter()
{
	reshade::register_overlay(nullptr, draw_settings_overlay);
	reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
	reshade::register_event<reshade::addon_event::begin_debug_event>(on_begin_debug_event);
	reshade::register_event<reshade::addon_event::end_debug_event>(on_end_debug_event);
	reshade::register_event<reshade::addon_event::insert_debug_marker>(on_insert_debug_marker);
	reshade::register_event<reshade::addon_event::present>(on_present);
	reshade::register_event<reshade::addon_event::finish_present>(on_finish_present);
}

void unregister_addon_draw_filter()
{
	disable_center_picker_after_draw_event();
	reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
	reshade::unregister_event<reshade::addon_event::begin_debug_event>(on_begin_debug_event);
	reshade::unregister_event<reshade::addon_event::end_debug_event>(on_end_debug_event);
	reshade::unregister_event<reshade::addon_event::insert_debug_marker>(on_insert_debug_marker);
	reshade::unregister_event<reshade::addon_event::present>(on_present);
	reshade::unregister_event<reshade::addon_event::finish_present>(on_finish_present);
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
