import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

// The repository has no package.json, by design. Import the browser ESM source
// through a data URL, so this check runs on stock Node with no dependencies.
const source = await readFile(
    new URL("../web/js/dgx_monarch_segments.js", import.meta.url),
);
const moduleUrl = `data:text/javascript;base64,${source.toString("base64")}`;
const { computeMemorySegments, selectMemoryHostCards } = await import(moduleUrl);

const tables = [
    { name: "missing telemetry", stats: {}, meta: null },
    {
        name: "normal MemFree telemetry",
        stats: {
            mem_gib: {
                MemTotal: 120,
                MemAvailable: 32,
                MemFree: 8,
                AnonPages: 48,
                Shmem: 12,
                Cached: 25,
                Buffers: 2,
            },
            gpu_proc_gib: 24,
            pool_gib: 10,
        },
        meta: { memory: { slab_gib: 8 } },
    },
    {
        name: "legacy cache fallback",
        stats: {
            mem_gib: {
                MemTotal: 120,
                MemAvailable: 40,
                AnonPages: 30,
                Shmem: 6,
                Cached: 28,
                Buffers: 1,
            },
            gpu_proc_gib: 20,
            pool_gib: 7,
        },
        meta: { memory: { slab_gib: 5 } },
    },
    {
        name: "overreported categories clamp to physical bounds",
        stats: {
            mem_gib: {
                MemTotal: 64,
                MemAvailable: 80,
                MemFree: -5,
                AnonPages: 999,
                Shmem: 999,
            },
            gpu_proc_gib: 999,
            pool_gib: 999,
        },
        meta: { memory: { slab_gib: 999 } },
    },
    {
        name: "invalid numeric telemetry stays finite and nonnegative",
        stats: {
            mem_gib: {
                MemTotal: Number.NaN,
                MemAvailable: Number.POSITIVE_INFINITY,
                MemFree: Number.NaN,
                Cached: -4,
            },
            gpu_proc_gib: -2,
        },
        meta: { memory: { slab_gib: -1 } },
    },
];

const usedKeys = ["gpu", "slab", "pool", "anon", "other"];
const coloredKeys = [...usedKeys, "reclaimable"];
assert.equal(usedKeys.length, 5);
assert.equal(coloredKeys.length, 6);

function close(actual, expected) {
    const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
    return Math.abs(actual - expected) <= Number.EPSILON * 16 * scale;
}

for (const table of tables) {
    const segments = computeMemorySegments(table.stats, table.meta);
    for (const [name, value] of Object.entries(segments)) {
        assert.ok(Number.isFinite(value), `${table.name}: ${name} must be finite`);
        assert.ok(value >= 0, `${table.name}: ${name} must be nonnegative`);
    }
    const usedSum = usedKeys.reduce((sum, key) => sum + segments[key], 0);
    assert.ok(close(usedSum, segments.used), `${table.name}: used categories must sum to used`);

    const coloredSum = coloredKeys.reduce((sum, key) => sum + segments[key], 0);
    assert.ok(
        close(coloredSum + segments.free, segments.total),
        `${table.name}: six colored categories plus free must sum to total`,
    );
}

const incompleteWorker = { host: "spark-a" };
const driverA = { host: "spark-a", mem_gib: { MemTotal: 120 } };
const minimalDriverA = { host: "spark-a" };
assert.deepEqual(
    selectMemoryHostCards({ workers: [incompleteWorker] }),
    [],
    "hostname-only worker telemetry must not create a populated card",
);
assert.deepEqual(
    selectMemoryHostCards({ workers: [incompleteWorker], driver_host: driverA }),
    [{ name: "spark-a (driver)", stats: driverA, meta: null }],
    "an incomplete worker must not hide complete same-host driver telemetry",
);

const workerStats = { host: "spark-b", mem_gib: { MemTotal: 120 } };
const workerVram = { allocated_gib: 4 };
const workerMemory = { slab_gib: 2 };
const completeWorker = {
    host: workerStats,
    vram: workerVram,
    memory: workerMemory,
};
const distinctCards = selectMemoryHostCards({
    workers: [completeWorker],
    driver_host: driverA,
});
assert.deepEqual(
    distinctCards,
    [
        {
            name: "spark-b",
            stats: workerStats,
            meta: { vram: workerVram, memory: workerMemory },
        },
        { name: "spark-a (driver)", stats: driverA, meta: null },
    ],
    "a distinct driver must follow complete worker cards",
);
assert.equal(distinctCards[0].stats, workerStats);
assert.equal(distinctCards[0].meta.vram, workerVram);
assert.equal(distinctCards[0].meta.memory, workerMemory);
assert.deepEqual(
    selectMemoryHostCards({
        workers: [completeWorker],
        driver_host: { host: "spark-b", mem_gib: { MemTotal: 120 } },
    }),
    [{
        name: "spark-b",
        stats: workerStats,
        meta: { vram: workerVram, memory: workerMemory },
    }],
    "a complete same-host worker must deduplicate the driver card",
);
assert.equal(
    selectMemoryHostCards({ workers: [completeWorker, completeWorker] }).length,
    2,
    "two valid workers on one host must keep both cards",
);
assert.deepEqual(
    selectMemoryHostCards({ workers: "malformed", driver_host: minimalDriverA }),
    [{ name: "spark-a (driver)", stats: minimalDriverA, meta: null }],
    "malformed workers and legacy driver telemetry must remain bounded and visible",
);

console.log(
    `checked ${tables.length} unified-memory accounting tables and host-card selection`,
);
