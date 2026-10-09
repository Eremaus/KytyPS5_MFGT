#ifndef EMULATOR_INCLUDE_COMMON_PERF_STATS_H_
#define EMULATOR_INCLUDE_COMMON_PERF_STATS_H_

// Lightweight per-second performance meter, enabled with the KYTY_PERF environment variable.
// Prints one line per second to stdout with time spent in the main host-side cost centres.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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
	SrtRead,
	MemoHit,
	MemoMiss,
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
	{
		const double t_srt = ms(Counter::SrtRead);
		std::printf("PERF srt_read=%.0fms(%llu) materialize_memo hit=%llu miss=%llu\n", t_srt,
		            static_cast<unsigned long long>(n(Counter::SrtRead)),
		            static_cast<unsigned long long>(n(Counter::MemoHit)),
		            static_cast<unsigned long long>(n(Counter::MemoMiss)));
	}
	std::printf("PERF readback sources (count/s): cpu_page_fault=%llu texture_metadata=%llu "
	            "shader_scalar_read=%llu indirect_args=%llu cpu_write_invalidate=%llu\n",
	            static_cast<unsigned long long>(n(Counter::ReadFault)),
	            static_cast<unsigned long long>(n(Counter::ReadMetadata)),
	            static_cast<unsigned long long>(n(Counter::ReadScalar)),
	            static_cast<unsigned long long>(n(Counter::ReadIndirectArgs)),
	            static_cast<unsigned long long>(n(Counter::ReadInvalidate)));
	std::fflush(stdout);
}

} // namespace Common::Perf

#endif
