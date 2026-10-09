// Unified-memory accounting shared by the sidebar and its Node.js check.
// tools/check_dgx_monarch_segments.mjs loads this file through a data URL, so
// keep it free of imports, ComfyUI dependencies, and DOM access.

function nonnegativeNumber(value, fallback = 0) {
    return typeof value === "number" && Number.isFinite(value)
        ? Math.max(0, value)
        : fallback;
}

// One card per worker whose host plane has arrived: before setup, or when its
// host stats fail, a worker reports `host` as a plain hostname string, and
// drawing defaults for it would fake a populated card. The driver's own box
// gets a card too when no worker card names it, so a driver-only machine (one
// that also hosts an LLM, say) keeps its card once a remote mesh comes up.
export function selectMemoryHostCards(telemetry = {}) {
    const cards = [];
    const seenWorkerHosts = new Set();
    const workers = Array.isArray(telemetry?.workers) ? telemetry.workers : [];

    for (const worker of workers) {
        const stats = worker?.host;
        if (!stats || typeof stats !== "object" || Array.isArray(stats)
            || !stats.host || !stats.mem_gib) continue;
        seenWorkerHosts.add(stats.host);
        cards.push({
            name: stats.host,
            stats,
            meta: { vram: worker.vram, memory: worker.memory },
        });
    }

    const driver = telemetry?.driver_host;
    if (driver && typeof driver === "object" && !Array.isArray(driver)
        && driver.host && !seenWorkerHosts.has(driver.host)) {
        cards.push({ name: `${driver.host} (driver)`, stats: driver, meta: null });
    }
    return cards;
}

export function computeMemorySegments(stats = {}, meta = null) {
    const mem = stats?.mem_gib || {};
    const totalCandidate = nonnegativeNumber(mem.MemTotal);
    const total = totalCandidate > 0 ? totalCandidate : 121.6;
    const available = Math.min(total, nonnegativeNumber(mem.MemAvailable));
    const anonPages = nonnegativeNumber(mem.AnonPages);
    const shmem = nonnegativeNumber(mem.Shmem);
    const cache = nonnegativeNumber(mem.Cached) + nonnegativeNumber(mem.Buffers);
    const used = total - available;

    // GB10 CUDA allocations are driver-owned pages outside AnonPages. Attribute
    // them first, then cap every later category by the unclaimed used-memory
    // remainder so the five used categories cannot overlap or exceed `used`.
    const gpu = Math.min(nonnegativeNumber(stats?.gpu_proc_gib), used);
    const memoryMeta = meta?.memory || null;
    const slab = Math.min(
        nonnegativeNumber(memoryMeta?.slab_gib),
        shmem,
        used - gpu,
    );
    const pool = Math.min(
        nonnegativeNumber(stats?.pool_gib),
        anonPages,
        used - gpu - slab,
    );
    const anon = Math.min(
        Math.max(0, anonPages - pool),
        used - gpu - slab - pool,
    );
    const other = used - gpu - slab - pool - anon;

    // MemAvailable already includes reclaimable cache: split it into
    // reclaimable and free, and count neither as used. Older telemetry lacks
    // MemFree and falls back to Cached + Buffers - Shmem; both branches stay
    // within MemAvailable.
    const reclaimable = mem.MemFree != null
        ? Math.min(available, Math.max(0, available - nonnegativeNumber(mem.MemFree)))
        : Math.min(available, Math.max(0, cache - shmem));
    const free = available - reclaimable;

    return { total, used, gpu, slab, pool, anon, other, reclaimable, free };
}
