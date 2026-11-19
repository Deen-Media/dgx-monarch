<script lang="ts">
  import {
    clipFrames,
    clipRate,
    clipSeconds,
    setDuration,
    snapFrames,
    type DurationGroup,
  } from '../frameCounts';
  import { nodeName } from '../workflowGraph';
  import { numberField } from '../numberField';
  let {
    group,
    values,
    onchange,
  }: {
    group: DurationGroup;
    values: Record<string, unknown>;
    /** Receives the draft values with every member set to the new frame count. */
    onchange: (values: Record<string, unknown>) => void;
  } = $props();
  const frames = $derived(clipFrames(group, values));
  // The control shows a count only while every member holds the same one.
  const shared = $derived(
    frames.length && frames.every((count) => count !== null && count === frames[0])
      ? (frames[0] as number)
      : null,
  );
  const rate = $derived(clipRate(group, values));
  const seconds = $derived(shared !== null && rate.fps ? clipSeconds(shared, rate.fps) : null);
  const key = $derived(group.key);
  const names = $derived(
    group.members
      .map((member) => `the ${member.role} (node ${nodeName(member.node)})`)
      .join(' and '),
  );
  function write(count: number) {
    onchange(setDuration(group, snapFrames(group, count), values));
  }
</script>

<fieldset class="duration-field" aria-describedby={`duration-${key}-note`}>
  <legend
    >Duration <span class="field-type">{group.family} clip{group.clip ? ` ${group.clip}` : ''}</span
    ></legend
  >
  <div class="input-row">
    <label
      >Frames<input
        id={`duration-${key}-frames`}
        type="number"
        min={group.min}
        max={Number.isFinite(group.max) ? group.max : undefined}
        step={group.grid.step}
        use:numberField={{
          value: shared,
          commit: (count) => count !== null && write(count),
          integer: true,
          event: 'change',
        }}
      /></label
    ><label
      >Seconds<input
        id={`duration-${key}-seconds`}
        type="number"
        min="0"
        step="0.01"
        disabled={!rate.fps}
        use:numberField={{
          value: seconds,
          commit: (value) => value !== null && rate.fps && write(value * rate.fps),
          min: 0,
          event: 'change',
        }}
      /></label
    >
  </div>
  <small id={`duration-${key}-note`}
    >{shared !== null && seconds !== null
      ? `${shared} frames is ${seconds} s at ${rate.fps} fps.`
      : shared !== null
        ? `${shared} frames. ${rate.reason ?? ''}`
        : `The frame counts differ: ${frames.map((count) => count ?? 'none').join(' and ')}. Enter a duration to set them all.`}
    Duration writes {group.members.length > 1 ? 'one count to' : 'the count of'}
    {names}, rounded to the nearest count {group.family} renders exactly: {group.grid.base} plus a multiple
    of {group.grid.step}.</small
  >
</fieldset>

<style>
  .duration-field {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px 12px;
    margin: 0 0 21px;
  }
  legend {
    font-size: 12px;
    font-weight: 600;
    color: #c3c8d1;
    padding: 0 4px;
  }
  .input-row {
    margin: 6px 0 10px;
  }
</style>
