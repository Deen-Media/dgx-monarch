<script lang="ts">
  import type { Telemetry } from '../contracts';
  import type { Presentation } from '../telemetryHistory';
  import { hostSummary } from '../hostHealth';
  let {
    telemetry,
    presentation,
    heading = '',
    now = null,
    live = true,
  }: {
    telemetry: Telemetry;
    presentation?: Presentation;
    /** A small heading inside the block, for a panel that has none of its own. */
    heading?: string;
    /** The companion's time at the newest snapshot, which ages retained readings. */
    now?: number | null;
    /** Whether Studio still reaches the companion; without it every reading is retained. */
    live?: boolean;
  } = $props();
  const summary = $derived(hostSummary(telemetry, presentation, { now, live }));
</script>

<section class="host-health" aria-label="Host readings">
  {#if heading}<h3>{heading}</h3>{/if}
  {#if summary.stale}<p class="stale-note" role="status">{summary.stale}</p>{/if}
  {#if summary.hosts.length}<div class="host-cards">
      {#each summary.hosts as host (host.host)}<div class="host-card" class:stale={!!summary.stale}>
          <div class="host-name">
            <strong title={host.host}>{host.host}</strong><small>{host.roles}</small>
          </div>
          <dl class="readings">
            {#each host.readings as reading (reading.id)}<div
                class="reading"
                class:wide={reading.id === 'host.memory.available'}
              >
                <dt>{reading.name}</dt>
                <dd title={reading.status?.title}>
                  {#if reading.status?.icon}<span
                      class="status-icon"
                      style:color={reading.status.hex}
                      aria-hidden="true">{reading.status.icon}</span
                    >{/if}{reading.text}
                </dd>
              </div>{/each}
          </dl>
          {#if host.age}<small class="host-age" class:stale={!!summary.stale}>{host.age}</small
            >{/if}
        </div>{/each}
    </div>{/if}
  {#if summary.note}<p class="host-note">{summary.note}</p>{/if}
  {#if summary.readiness.length}<dl class="readiness-rows" aria-label="Driver readiness">
      {#each summary.readiness as row (row.label)}<dt>{row.label}</dt>
        <dd>
          {#if row.mark.icon}<span class="status-icon" style:color={row.mark.hex} aria-hidden="true"
              >{row.mark.icon}</span
            >{/if}{row.state}{#if row.stale}<span class="stale-tag">{row.stale}</span>{/if}
        </dd>
        <dd class="detail">{row.detail}</dd>{/each}
    </dl>{/if}
</section>

<style>
  .host-health {
    display: grid;
    gap: 10px;
    margin: 14px 0;
    min-width: 0;
  }
  .host-health h3 {
    margin: 0;
  }
  .host-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 10px;
  }
  .host-card {
    display: grid;
    gap: 6px;
    min-width: 0;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #191d24;
  }
  .host-name {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .host-name strong {
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .host-name small,
  .host-age {
    font-size: 10px;
    color: var(--muted);
  }
  dl {
    margin: 0;
  }
  dd {
    margin: 0;
  }
  /* Each reading puts its name over its value, two to a row, so a 236 px panel keeps both legible. */
  .readings {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    align-items: end;
    gap: 7px 10px;
  }
  .reading {
    min-width: 0;
  }
  .reading.wide {
    grid-column: 1 / -1;
  }
  .reading dt {
    font-size: 10px;
    color: var(--muted);
  }
  .reading dd {
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .status-icon {
    margin-right: 4px;
  }
  .host-note {
    margin: 0;
    font-size: 11px;
    color: var(--muted);
  }
  /* Retained readings stay readable but dim, under a line that says why and how old they are. */
  .stale-note,
  .host-age.stale,
  .stale-tag {
    color: var(--status-degraded);
  }
  .stale-note {
    margin: 0;
    font-size: 11px;
  }
  .host-card.stale .readings {
    opacity: 0.6;
  }
  .stale-tag {
    display: block;
    font-size: 10px;
  }
  /* A layer's name and state share a line; the driver's sentence about it runs below. */
  .readiness-rows {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 2px 8px;
    font-size: 11px;
  }
  .readiness-rows dt {
    color: var(--muted);
  }
  .readiness-rows .detail {
    grid-column: 1 / -1;
    margin-bottom: 5px;
    font-size: 10px;
    color: var(--muted);
  }
</style>
