#ifndef EMULATOR_INCLUDE_COMMON_PERF_STATS_H_
#define EMULATOR_INCLUDE_COMMON_PERF_STATS_H_

// Lightweight per-second performance meter, enabled with the KYTY_PERF environment variable.
// Prints one line per second to stdout with time spent in the main host-side cost centres.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Common::Perf {

enum class Counter : uint32_t {
	ShaderTranslate,
	PipelineCreate,
	GpuWait,
	BdaSync,
	BufferReadback,
	ReadFault,
	ReadMetadata,
	ReadScalar,
	ReadIndirectArgs,
	ReadInvalidate,
	BdaFullWalk,
	Count
};

struct State {
	std::atomic<uint64_t> ns[static_cast<uint32_t>(Counter::Count)] {};
	std::atomic<uint64_t> calls[static_cast<uint32_t>(Counter::Count)] {};
	std::atomic<uint64_t> flips {0};
	std::atomic<uint64_t> draws {0};
	std::atomic<uint64_t> dispatches {0};
	std::atomic<int64_t>  last_report_ns {0};
};

inline State& GetState() {
	static State state;
	return state;
}

inline bool Enabled() {
	static const bool enabled = std::getenv("KYTY_PERF") != nullptr;
	return enabled;
}

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

class Scope {
public:
	explicit Scope(Counter counter): m_counter(counter), m_start(Enabled() ? NowNs() : 0) {}
	~Scope() {
		if (m_start != 0) {
			auto& state = GetState();
			state.ns[static_cast<uint32_t>(m_counter)] += static_cast<uint64_t>(NowNs() - m_start);
			state.calls[static_cast<uint32_t>(m_counter)]++;
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Counter m_counter;
	int64_t m_start;
};

// Phase timing for the draw path: each call charges the time since the previous call (on the
// same thread) to the previous phase. Phase names must be string literals.
struct PhaseTable {
	static constexpr uint32_t Max = 16;
	std::atomic<const char*>  names[Max] {};
	std::atomic<uint64_t>     ns[Max] {};
};
inline PhaseTable& GetPhases() {
	static PhaseTable table;
	return table;
}
inline uint32_t PhaseIndex(const char* name) {
	auto& table = GetPhases();
	for (uint32_t i = 0; i < PhaseTable::Max; i++) {
		const char* current = table.names[i].load(std::memory_order_acquire);
		if (current == name) return i;
		if (current == nullptr) {
			const char* expected = nullptr;
			if (table.names[i].compare_exchange_strong(expected, name) || expected == name) {
				return i;
			}
		}
	}
	return PhaseTable::Max - 1;
}
inline void MarkPhase(const char* name) {
	if (!Enabled()) return;
	thread_local int64_t     last_ns    = 0;
	thread_local const char* last_phase = nullptr;
	const auto now = NowNs();
	if (last_phase != nullptr && last_ns != 0) {
		GetPhases().ns[PhaseIndex(last_phase)] += static_cast<uint64_t>(now - last_ns);
	}
	last_ns    = now;
	last_phase = name;
}

inline void Count(Counter counter) {
	if (Enabled()) GetState().calls[static_cast<uint32_t>(counter)]++;
}

inline void CountDraw() {
	if (Enabled()) GetState().draws++;
}
inline void CountDispatch() {
	if (Enabled()) GetState().dispatches++;
}

// Call once per presented frame; prints and resets the counters about once per second.
// Readback hotspots: who made the emulator wait for GPU data. Origins are guest instruction
// addresses for CPU page faults, or small tags for emulator-side reads.
enum ReadbackOrigin : uint64_t {
	OriginUnknown       = 0,
	OriginIndirectArgs  = 1,
	OriginIndirectCount = 2,
	OriginDispatchArgs  = 3,
	OriginTexture       = 4,
	OriginScalar        = 5,
	OriginInvalidate    = 6,
};

inline thread_local uint64_t t_readback_origin = OriginUnknown;

class OriginScope {
public:
	explicit OriginScope(uint64_t origin): m_saved(t_readback_origin) { t_readback_origin = origin; }
	~OriginScope() { t_readback_origin = m_saved; }
	OriginScope(const OriginScope&)            = delete;
	OriginScope& operator=(const OriginScope&) = delete;

private:
	uint64_t m_saved;
};

struct ReadbackHotspots {
	std::mutex                                                   mutex;
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> by_origin; // count, ns
	int64_t                                                      last_report_ns = 0;
};

inline ReadbackHotspots& GetHotspots() {
	static ReadbackHotspots hotspots;
	return hotspots;
}

inline void RecordReadback(uint64_t origin, uint64_t ns) {
	if (!Enabled()) return;
	auto&            h = GetHotspots();
	std::scoped_lock lock(h.mutex);
	auto&            entry = h.by_origin[origin];
	entry.first++;
	entry.second += ns;
}

inline const char* OriginName(uint64_t origin) {
	switch (origin) {
		case OriginUnknown: return "unknown";
		case OriginIndirectArgs: return "indirect-draw-args";
		case OriginIndirectCount: return "indirect-draw-count";
		case OriginDispatchArgs: return "indirect-dispatch-args";
		case OriginTexture: return "texture";
		case OriginScalar: return "shader-scalar-read";
		case OriginInvalidate: return "cpu-write-invalidate";
		default: return nullptr;
	}
}

inline void ReportHotspots(int64_t now) {
	auto&            h = GetHotspots();
	std::scoped_lock lock(h.mutex);
	if (h.last_report_ns == 0) {
		h.last_report_ns = now;
		return;
	}
	if (now - h.last_report_ns < 5000000000ll) return;
	const double secs = static_cast<double>(now - h.last_report_ns) / 1e9;
	h.last_report_ns  = now;
	std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> rows(h.by_origin.begin(),
	                                                                     h.by_origin.end());
	h.by_origin.clear();
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
	std::printf("PERF readback hotspots (last %.0fs, per second):", secs);
	for (size_t i = 0; i < rows.size() && i < 10; ++i) {
		const auto origin = rows[i].first;
		const auto count  = static_cast<double>(rows[i].second.first) / secs;
		const auto ms     = static_cast<double>(rows[i].second.second) / 1e6 / secs;
		if (const char* name = OriginName(origin); name != nullptr) {
			std::printf(" [%s %.0fx %.0fms]", name, count, ms);
		} else if (origin >= 0x900000000ull && origin < 0xa00000000ull) {
			std::printf(" [eboot+0x%llx %.0fx %.0fms]",
			            static_cast<unsigned long long>(origin - 0x900000000ull), count, ms);
		} else {
			std::printf(" [pc=0x%llx %.0fx %.0fms]", static_cast<unsigned long long>(origin), count, ms);
		}
	}
	std::printf("\n");
}

inline void OnFlip() {
	if (!Enabled()) {
		return;
	}
	auto&      state = GetState();
	const auto now   = NowNs();
	state.flips++;
	auto last = state.last_report_ns.load();
	if (last == 0) {
		state.last_report_ns = now;
		return;
	}
	if (now - last < 1000000000 || !state.last_report_ns.compare_exchange_strong(last, now)) {
		return;
	}
	const double secs = static_cast<double>(now - last) / 1e9;
	auto ms = [&](Counter c) {
		return static_cast<double>(state.ns[static_cast<uint32_t>(c)].exchange(0)) / 1e6 / secs;
	};
	auto n = [&](Counter c) { return state.calls[static_cast<uint32_t>(c)].exchange(0); };
	const auto flips = state.flips.exchange(0);
	const auto draws = state.draws.exchange(0);
	const auto disp  = state.dispatches.exchange(0);
	const double t_shader = ms(Counter::ShaderTranslate), t_pipe = ms(Counter::PipelineCreate),
	             t_wait = ms(Counter::GpuWait), t_bda = ms(Counter::BdaSync),
	             t_read = ms(Counter::BufferReadback);
	std::printf("PERF fps=%.1f draws/s=%.0f dispatches/s=%.0f | per second (ms): "
	            "shader_translate=%.0f(%llu) pipeline_create=%.0f(%llu) gpu_wait=%.0f(%llu) "
	            "bda_sync=%.0f(%llu) buffer_readback=%.0f(%llu)\n",
	            static_cast<double>(flips) / secs, static_cast<double>(draws) / secs,
	            static_cast<double>(disp) / secs, t_shader,
	            static_cast<unsigned long long>(n(Counter::ShaderTranslate)), t_pipe,
	            static_cast<unsigned long long>(n(Counter::PipelineCreate)), t_wait,
	            static_cast<unsigned long long>(n(Counter::GpuWait)), t_bda,
	            static_cast<unsigned long long>(n(Counter::BdaSync)), t_read,
	            static_cast<unsigned long long>(n(Counter::BufferReadback)));
	{
		auto& table = GetPhases();
		std::printf("PERF draw phases (ms/s):");
		for (uint32_t i = 0; i < PhaseTable::Max; i++) {
			const char* name = table.names[i].load();
			if (name == nullptr) break;
			std::printf(" %s=%.0f", name, static_cast<double>(table.ns[i].exchange(0)) / 1e6 / secs);
		}
		std::printf("\n");
	}
	std::printf("PERF readback sources (count/s): cpu_page_fault=%llu texture_metadata=%llu "
	            "shader_scalar_read=%llu indirect_args=%llu cpu_write_invalidate=%llu "
	            "bda_full_walks=%llu\n",
	            static_cast<unsigned long long>(n(Counter::ReadFault)),
	            static_cast<unsigned long long>(n(Counter::ReadMetadata)),
	            static_cast<unsigned long long>(n(Counter::ReadScalar)),
	            static_cast<unsigned long long>(n(Counter::ReadIndirectArgs)),
	            static_cast<unsigned long long>(n(Counter::ReadInvalidate)),
	            static_cast<unsigned long long>(n(Counter::BdaFullWalk)));
	ReportHotspots(now);
	std::fflush(stdout);
}

} // namespace Common::Perf

#endif
