// Topology and RDMA rail presentation for the sidebar.
// Pure: no imports, no DOM and no fetch. The node checks in
// tests/test_readiness_panel_model.py load this file directly, and the panel
// passes its own clock in.

// Topology.describe() in src/dgx_monarch/topology.py, in its order.
const DEGREES = [["ulysses", "uly"], ["ring", "ring"], ["cfg", "cfg"], ["dp", "dp"]];

// The short names `dgxm top` gives these rails (RAIL_ALIASES in tui/data.py).
const RAIL_ALIASES = { rocep1s0f0: "f0", roceP2p1s0f0: "P2p" };

function isRecord(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function degree(value) {
    return Number.isInteger(value) && value >= 1 ? value : null;
}

function finite(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// A worker's topology dict as the driver labels it: "uly2", "cfg2",
// "uly2+fsdp", or "single" when every degree is 1. A worker before setup
// publishes {}, which is no topology, so it reads null and not "single".
export function topologyLabel(topology) {
    if (!isRecord(topology) || Object.keys(topology).length === 0) return null;
    const parts = [];
    for (const [key, prefix] of DEGREES) {
        const value = topology[key] === undefined ? 1 : degree(topology[key]);
        if (value === null) return null;
        if (value > 1) parts.push(`${prefix}${value}`);
    }
    if (topology.fsdp === true) parts.push("fsdp");
    return parts.length ? parts.join("+") : "single";
}

// One line for the topology every set-up worker reports, or null when none
// has set up. Workers that disagree are named, never averaged.
export function activeTopology(workers) {
    const labels = new Set();
    const worlds = new Set();
    for (const worker of Array.isArray(workers) ? workers : []) {
        if (!isRecord(worker)) continue;
        const label = topologyLabel(worker.topology);
        if (label === null) continue;
        labels.add(label);
        if (degree(worker.world) !== null) worlds.add(worker.world);
    }
    if (!labels.size) return null;
    const names = [...labels].sort();
    if (names.length > 1) return `Topology mixed: ${names.join(", ")}`;
    return `Topology ${names[0]}` + (worlds.size === 1 ? ` · world ${[...worlds][0]}` : "");
}

// Per-rail RX and TX GB/s between two readings, with the math of _rail_rates
// in src/dgx_monarch/tui/data.py: the byte delta over the elapsed seconds, the
// elapsed time floored at 1 ms, and a negative or non-finite rate read as 0.
// `stamp` is the reading's payload stamp (stats.t) and `readAt` the box's own
// stamp of when it read the counters (stats.rails_t), both in seconds. When
// this reading and the last one both carry a readAt, and it moved forward, the
// elapsed time is the interval between them, so a reading that waited in the
// driver's cache is timed by when it was read. Otherwise the elapsed time is
// the caller's monotonic clock (the panel passes performance.now(), in
// milliseconds), as for an older driver. Two departures, both for the panel: a
// rail's first sample reads null, which the panel prints as a dash, because 0
// would read as an idle rail. And a host whose reading carries the same stamp
// as the last one (a cached worker row while a render keeps the worker busy)
// keeps its last rates and its baseline, so the gap never reads as 0 followed
// by a burst.
export function createRailRateTracker() {
    const hosts = new Map();  // host name -> last reading; replaced per poll
    return {
        update(host, rails, stamp, nowMs, readAt) {
            const previous = hosts.get(host);
            const at = finite(stamp);
            const read = finite(readAt);
            if (previous && ((at !== null && at === previous.stamp)
                || (read !== null && read === previous.read))) return previous.rates;
            const span = previous && read !== null && previous.read !== null && read > previous.read
                ? read - previous.read : null;
            const counters = new Map();
            const rates = {};
            for (const [dev, row] of Object.entries(isRecord(rails) ? rails : {})) {
                if (!isRecord(row)) continue;
                const rx = finite(row.rx_bytes);
                const tx = finite(row.tx_bytes);
                if (rx === null || tx === null) continue;
                counters.set(dev, { rx, tx });
                const before = previous?.counters.get(dev);
                if (!before) {
                    rates[dev] = { rx_gbs: null, tx_gbs: null };
                    continue;
                }
                const dt = Math.max(1e-3, span === null ? (nowMs - previous.t) / 1000 : span);
                const rate = (delta) => {
                    const gbs = delta / dt / 1e9;
                    return Number.isFinite(gbs) ? Math.max(0, gbs) : 0;
                };
                rates[dev] = { rx_gbs: rate(rx - before.rx), tx_gbs: rate(tx - before.tx) };
            }
            hosts.set(host, { t: nowMs, stamp: at, read, counters, rates });
            return rates;
        },
    };
}

// The rates of each box for one poll, as a Map from host name to the rates
// update() returns. A box that runs two ranks sends two worker rows with the
// same host name, each stamped by its own process. Measured one after the
// other, the second row would be timed against the first, read a fraction of a
// second before, and print a rate far too high. So each box is read once per
// poll and, as in `dgxm top`, its last row is its reading.
export function hostRailRates(tracker, cards, nowMs) {
    const readings = new Map();
    for (const card of Array.isArray(cards) ? cards : []) {
        const stats = isRecord(card) ? card.stats : null;
        if (isRecord(stats) && stats.host) readings.set(stats.host, stats);
    }
    const rates = new Map();
    for (const [host, stats] of readings) {
        rates.set(host, tracker.update(host, stats.rails, stats.t, nowMs, stats.rails_t));
    }
    return rates;
}

function plainState(text) {
    // sysfs prefixes the state with its number: "1: DOWN" reads "DOWN".
    return String(text).replace(/^\s*\d+:\s*/, "");
}

// The rails one host card lists. A port shows when it carries traffic (it
// has a rate entry) or when it is cabled but down the way `dgxm doctor` reads
// it: phys_state LinkUp with a state other than ACTIVE. An ACTIVE port with no
// counters (an unused function of a cabled port) stays out, and so does a
// down port with no link: an uncabled port reads the same as a rail that
// never trained at boot, and doctor reports that one before a render.
export function railRows(rates, linkState) {
    const traffic = isRecord(rates) ? rates : {};
    const states = isRecord(linkState) ? linkState : {};
    const devices = new Set(Object.keys(traffic));
    for (const [dev, state] of Object.entries(states)) {
        if (isRecord(state) && typeof state.state === "string" && state.state !== "ACTIVE"
            && typeof state.phys_state === "string" && state.phys_state.includes("LinkUp")) {
            devices.add(dev);
        }
    }
    return [...devices].sort().map((dev) => {
        const state = isRecord(states[dev]) && typeof states[dev].state === "string"
            ? states[dev] : null;
        const rate = isRecord(traffic[dev]) ? traffic[dev] : {};
        return {
            name: RAIL_ALIASES[dev] || dev,
            rx_gbs: finite(rate.rx_gbs),
            tx_gbs: finite(rate.tx_gbs),
            // null when the payload has no link state (an older driver).
            active: state ? state.state === "ACTIVE" : null,
            state: state ? plainState(state.state) : null,
            phys: state && typeof state.phys_state === "string"
                ? plainState(state.phys_state) : null,
        };
    });
}

function gbs(value) {
    return value === null ? "-" : value.toFixed(2);
}

// The compact line a host card draws, as text parts; `alert` parts mark a
// rail that is not ACTIVE. An empty list draws nothing.
export function railLineParts(rows) {
    const parts = [];
    for (const row of Array.isArray(rows) ? rows : []) {
        parts.push({ text: `${parts.length ? " · " : "rails "}${row.name} ` +
            `rx ${gbs(row.rx_gbs)} tx ${gbs(row.tx_gbs)} GB/s`, alert: false });
        if (row.active === false) {
            parts.push({ text: ` ${row.state}${row.phys ? ` (${row.phys})` : ""}`, alert: true });
        }
    }
    return parts;
}

// Where the last render ran and its last step rate, from the facts the driver
// records at finish: "uly2 · auto row 11 · 1.92s/it". Empty when the record
// carries none, as an older driver's does.
export function lastRenderFacts(last) {
    if (!isRecord(last)) return "";
    const parts = [];
    if (typeof last.topology === "string" && last.topology) parts.push(last.topology);
    if (degree(last.auto_row) !== null) parts.push(`auto row ${last.auto_row}`);
    const rate = finite(last.sec_per_step);
    if (rate !== null) parts.push(`${rate}s/it`);
    return parts.join(" · ");
}
