<script lang="ts">
  import Field from './Field.svelte';
  import type { OperationParameter } from '../contracts';
  import { numberField } from '../numberField';
  interface Host {
    name: string;
    fabric_ip: string;
    gpus: number;
    ssh_user?: string;
    comfy_dir?: string;
  }
  let {
    field,
    value,
    onchange,
  }: { field: OperationParameter; value: unknown; onchange: (value: unknown) => void } = $props();
  const hosts = $derived(Array.isArray(value) ? (value as Host[]) : []);
  function updateHost(index: number, key: keyof Host, next: string | number) {
    onchange(hosts.map((host, i) => (i === index ? { ...host, [key]: next } : host)));
  }
</script>

{#if field.type === 'STRINGS'}<label
    >{field.name}<textarea
      rows="3"
      value={Array.isArray(value) ? value.join('\n') : ''}
      onchange={(event) =>
        onchange(
          event.currentTarget.value
            .split('\n')
            .map((item) => item.trim())
            .filter(Boolean),
        )}></textarea></label
  ><small>One value per line.</small>
{:else if field.type === 'HOSTS'}<fieldset>
    <legend>{field.name}</legend>{#each hosts as host, i}<div class="host-row">
        <h4>Host {i + 1}</h4>
        <label
          >Name<input
            value={host.name}
            oninput={(event) => updateHost(i, 'name', event.currentTarget.value)}
          /></label
        >
        <label
          >Fabric IP<input
            value={host.fabric_ip}
            oninput={(event) => updateHost(i, 'fabric_ip', event.currentTarget.value)}
          /></label
        >
        <label
          >GPUs<input
            type="number"
            min="1"
            step="1"
            use:numberField={{
              value: host.gpus,
              commit: (gpus) => gpus !== null && updateHost(i, 'gpus', gpus),
              integer: true,
              min: 1,
            }}
          /></label
        >
        <label
          >SSH user (optional)<input
            value={host.ssh_user ?? ''}
            oninput={(event) => updateHost(i, 'ssh_user', event.currentTarget.value)}
          /></label
        >
        <label
          >Comfy directory (optional)<input
            value={host.comfy_dir ?? ''}
            oninput={(event) => updateHost(i, 'comfy_dir', event.currentTarget.value)}
          /></label
        >
        <button onclick={() => onchange(hosts.filter((_, index) => index !== i))}
          >Remove host</button
        >
      </div>{/each}
    <button onclick={() => onchange([...hosts, { name: '', fabric_ip: '', gpus: 1 }])}
      >Add host</button
    >
  </fieldset>
{:else}<Field
    binding={{
      id: `operation-${field.name}`,
      label: field.name,
      type: field.type === 'CHOICE' ? 'COMBO' : field.type,
      required: field.required,
      // The companion refuses an empty required operation parameter.
      validation: { nonEmpty: field.required },
      options: field.choices,
      min: field.min,
      max: field.max,
      default: field.default,
    }}
    {value}
    {onchange}
  />{/if}

<style>
  fieldset {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin: 12px 0;
  }
  .host-row {
    display: grid;
    gap: 6px;
    padding-bottom: 16px;
  }
  h4 {
    margin: 3px 0;
  }
</style>
