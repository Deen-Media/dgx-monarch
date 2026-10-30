<script lang="ts">
  import { tick } from 'svelte';
  import { fieldLabel, bindingType, bindingOptions, bindingError } from '../model';
  import type { Binding } from '../model';
  import { numberField } from '../numberField';
  import type { InputMeta } from '../frameCounts';
  import type { FieldPresentation } from '../fieldPresentation';
  import { growText } from '../textGrowth';
  let {
    binding,
    value,
    note = '',
    locked = '',
    meta,
    onchange,
    presentation,
    compact = false,
  }: {
    binding: Binding;
    value: unknown;
    /** Text the field shows in place of its value while the draft does not use it, or ''. */
    locked?: string;
    /** A line about how the draft uses this value, such as a seed chosen at review. */
    note?: string;
    /** The input's node, unit and relation to other inputs, such as a clip frame count. */
    meta?: InputMeta;
    onchange: (value: unknown) => void;
    /** The field's name and prompt role; without one the field shows its input name. */
    presentation?: FieldPresentation;
    /** The Inputs tab: no raw type such as STRING, and an empty negative prompt folded away. */
    compact?: boolean;
  } = $props();
  let type = $derived(bindingType(binding));
  let options = $derived(bindingOptions(binding));
  let error = $derived(bindingError(binding, value));
  let fieldId = $derived(`field-${binding.id}`);
  let label = $derived(presentation?.label ?? fieldLabel(binding));
  let tag = $derived(
    [compact ? '' : type.toLowerCase(), meta?.unit ?? ''].filter(Boolean).join(' · '),
  );
  // The main prompt starts taller and grows with its text; an empty negative prompt waits behind
  // a small button on Inputs until it is wanted.
  let prompt = $derived(presentation?.role === 'prompt');
  let unfolded = $state(false);
  let folded = $derived(
    compact && presentation?.role === 'negative' && !unfolded && !error && (value ?? '') === '',
  );
  let area = $state<HTMLTextAreaElement>();
  async function unfold() {
    unfolded = true;
    await tick();
    area?.focus();
  }
</script>

<div class="field" class:field-invalid={!!error}>
  {#if folded}<button
      type="button"
      class="field-opener"
      aria-expanded="false"
      aria-controls={fieldId}
      onclick={unfold}>Add {label.toLowerCase()}</button
    >{/if}
  <label
    for={fieldId}
    hidden={folded}
    title={presentation && presentation.source !== label ? presentation.source : undefined}
    >{label}
    {#if meta?.context}<span class="field-context">{meta.context}</span>{/if}{#if tag}<span
        class="field-type">{tag}</span
      >{/if}</label
  >
  {#if locked}
    <input id={fieldId} value={locked} disabled />
  {:else if binding.resolved === false}
    <div class="unresolved-field">{binding.reason ?? 'Unresolved binding'}</div>
  {:else if options.length}
    <select
      id={fieldId}
      value={String(value ?? '')}
      onchange={(e) => onchange(options.find((o) => String(o) === e.currentTarget.value))}
      aria-invalid={!!error}
    >
      {#if !options.includes(value)}<option value="">Choose an available value</option>{/if}
      {#each options as option}<option value={String(option)}>{String(option)}</option>{/each}
    </select>
  {:else if ['INT', 'INTEGER', 'FLOAT', 'NUMBER'].includes(type)}
    <input
      id={fieldId}
      type="number"
      min={binding.schema?.min ?? binding.min}
      max={binding.schema?.max ?? binding.max}
      step={binding.schema?.step ?? binding.step ?? (type === 'INT' ? 1 : 'any')}
      use:numberField={{ value, commit: onchange, nullable: true }}
      aria-invalid={!!error}
    />
  {:else if ['BOOLEAN', 'BOOL'].includes(type)}
    <input
      id={fieldId}
      type="checkbox"
      checked={Boolean(value)}
      onchange={(e) => onchange(e.currentTarget.checked)}
      aria-invalid={!!error}
    />
  {:else if ['STRING', 'TEXT'].includes(type)}
    <textarea
      id={fieldId}
      bind:this={area}
      hidden={folded}
      class:prompt-text={prompt}
      rows={prompt ? 7 : /prompt|text|positive|negative/i.test(fieldLabel(binding)) ? 4 : 2}
      value={String(value ?? '')}
      oninput={(e) => onchange(e.currentTarget.value)}
      use:growText={{ value, grow: prompt }}
      spellcheck={true}
      aria-invalid={!!error}></textarea>
  {:else}
    <input
      id={fieldId}
      value={typeof value === 'object' ? JSON.stringify(value) : String(value ?? '')}
      readonly
      aria-invalid={!!error}
    />
    <small>This input requires a supported typed adapter or connected workflow node.</small>
  {/if}
  {#if note}<small class="field-note">{note}</small>{/if}
  {#if meta?.note}<small class="field-note">{meta.note}</small>{/if}
  {#if binding.description && !folded}<small>{binding.description}</small>{/if}
  {#if error}<small class="error" role="status">{error}</small>{/if}
</div>
