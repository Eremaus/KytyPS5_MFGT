#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
namespace Libs::Graphics {

namespace {

// GPU breadcrumbs (KYTY_GPU_BREADCRUMBS=1, needs VK_AMD_buffer_marker): before every recorded
// draw/dispatch, write "started N" at the top of the pipe and "finished N-1" at the bottom of the
// pipe into host-visible memory. After a device loss the two values identify the operation the GPU
// was executing.
struct BreadcrumbRecord {
	uint32_t op     = 0;
	uint64_t submit = 0;
	uint32_t arg0 = 0, arg1 = 0, arg2 = 0, arg3 = 0;
	uint64_t arg4 = 0;
};

struct Breadcrumbs {
	std::mutex                          mutex;
	bool                                initialized = false;
	bool                                enabled     = false;
	vk::Device                          device      = nullptr;
	vk::Buffer                          buffer      = nullptr;
	vk::DeviceMemory                    memory      = nullptr;
	volatile uint32_t*                  values      = nullptr;
	uint32_t                            sequence    = 0;
	std::array<BreadcrumbRecord, 8192>  ring {};
};

Breadcrumbs g_breadcrumbs;

bool InitBreadcrumbs(GraphicContext& graphics) {
	auto& b = g_breadcrumbs;
	if (b.initialized) return b.enabled;
	b.initialized = true;
	const char* env = std::getenv("KYTY_GPU_BREADCRUMBS");
	if (env == nullptr || env[0] != '1') return false;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD == nullptr) {
		std::printf("GPU breadcrumbs: VK_AMD_buffer_marker is not available\n");
		return false;
	}
	vk::BufferCreateInfo info {};
	info.size  = 16;
	info.usage = vk::BufferUsageFlagBits::eTransferDst;
	if (graphics.device.createBuffer(&info, nullptr, &b.buffer) != vk::Result::eSuccess) return false;
	vk::MemoryRequirements req {};
	graphics.device.getBufferMemoryRequirements(b.buffer, &req);
	const auto props = graphics.physical_device.getMemoryProperties();
	uint32_t   type  = UINT32_MAX;
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
		const auto flags = props.memoryTypes[i].propertyFlags;
		if ((req.memoryTypeBits & (1u << i)) != 0 &&
		    (flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
		    (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
			type = i;
			break;
		}
	}
	if (type == UINT32_MAX) return false;
	vk::MemoryAllocateInfo alloc {};
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = type;
	if (graphics.device.allocateMemory(&alloc, nullptr, &b.memory) != vk::Result::eSuccess) return false;
	graphics.device.bindBufferMemory(b.buffer, b.memory, 0);
	void* mapped = nullptr;
	if (graphics.device.mapMemory(b.memory, 0, 16, {}, &mapped) != vk::Result::eSuccess) return false;
	b.values = static_cast<volatile uint32_t*>(mapped);
	b.values[0] = b.values[1] = 0;
	b.device  = graphics.device;
	b.enabled = true;
	std::printf("GPU breadcrumbs: enabled\n");
	return true;
}

} // namespace

void ReportGpuBreadcrumbs() {
	auto& b = g_breadcrumbs;
	if (!b.enabled) return;
	std::scoped_lock lock(b.mutex);
	const uint32_t started  = b.values[0];
	const uint32_t finished = b.values[1];
	static const char* const names[] = {"DispatchDirect", "DrawIndex", "DrawIndexAuto", "EopWrite",
	                                    "EopInterrupt", "EopWriteBack", "EopFlip",
	                                    "EopWriteBackFlip", "EopOnlyFlip", "DispatchIndirect",
	                                    "Unknown"};
	std::printf("GPU breadcrumbs: recorded=%u started=%u finished=%u\n", b.sequence, started, finished);
	const uint32_t first = finished + 1;
	const uint32_t last  = std::min(b.sequence, std::max(started, first) + 3);
	for (uint32_t seq = first; seq <= last && seq - first < 24; ++seq) {
		const auto& r = b.ring[seq % b.ring.size()];
		std::printf("  %s #%u %s submit=%llu args=%u,%u,%u,%u,0x%016llx\n",
		            seq <= started ? "RUNNING " : "queued  ", seq,
		            r.op < std::size(names) ? names[r.op] : "?", static_cast<unsigned long long>(r.submit),
		            r.arg0, r.arg1, r.arg2, r.arg3, static_cast<unsigned long long>(r.arg4));
	}
	std::fflush(stdout);
}

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
	if (!g_breadcrumbs.initialized) {
		std::scoped_lock lock(g_breadcrumbs.mutex);
		InitBreadcrumbs(m_graphics);
	}
	if (g_breadcrumbs.enabled && !IsInvalid()) {
		auto&    b = g_breadcrumbs;
		uint32_t seq = 0;
		{
			std::scoped_lock lock(b.mutex);
			seq = ++b.sequence;
			b.ring[seq % b.ring.size()] = {op, submit_id, arg0, arg1, arg2, arg3, arg4};
		}
		auto fn = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD;
		VkCommandBuffer cmd = m_buffer;
		fn(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, b.buffer, 0, seq);
		fn(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.buffer, 4, seq - 1);
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
