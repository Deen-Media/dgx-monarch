<script lang="ts">
  import { fieldLabel, bindingType, bindingOptions, bindingError } from '../model';
  import type { Binding } from '../model';
  import { numberField } from '../numberField';
  let {
    binding,
    value,
    onchange,
  }: { binding: Binding; value: unknown; onchange: (value: unknown) => void } = $props();
  let type = $derived(bindingType(binding));
  let options = $derived(bindingOptions(binding));
  let error = $derived(bindingError(binding, value));
  let fieldId = $derived(`field-${binding.id}`);
</script>

<div class="field" class:field-invalid={!!error}>
  <label for={fieldId}
    >{fieldLabel(binding)} <span class="field-type">{type.toLowerCase()}</span></label
  >
  {#if binding.resolved === false}
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
      rows={/prompt|text|positive|negative/i.test(fieldLabel(binding)) ? 4 : 2}
      value={String(value ?? '')}
      oninput={(e) => onchange(e.currentTarget.value)}
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
  {#if binding.description}<small>{binding.description}</small>{/if}
  {#if error}<small class="error" role="status">{error}</small>{/if}
</div>
