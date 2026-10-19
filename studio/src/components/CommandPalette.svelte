<script lang="ts">
  import type { OperationCapability } from '../contracts';
  import { navigation } from '../navigation';
  let {
    operations,
    navigate,
    onoperation,
  }: {
    operations: OperationCapability[];
    navigate: (view: string) => void;
    onoperation: (id: string) => void;
  } = $props();
  let paletteDialog: HTMLDialogElement,
    paletteInput: HTMLInputElement,
    paletteQuery = $state('');
  let commands = $derived(
    navigation.filter((n) => n.label.toLowerCase().includes(paletteQuery.toLowerCase())),
  );
  let operationCommands = $derived(
    operations.filter((o) =>
      `${o.label} ${o.id}`.toLowerCase().includes(paletteQuery.toLowerCase()),
    ),
  );
  export function open() {
    paletteQuery = '';
    paletteDialog.showModal();
    requestAnimationFrame(() => paletteInput.focus());
  }
  export function close() {
    paletteDialog.close();
  }
</script>

<dialog bind:this={paletteDialog} class="command-dialog" aria-labelledby="palette-title">
  <h2 id="palette-title" class="sr-only">Command palette</h2>
  <div class="palette-search">
    <span aria-hidden="true">⌕</span><input
      bind:this={paletteInput}
      bind:value={paletteQuery}
      placeholder="Go to a view…"
      aria-label="Search commands"
    /><button onclick={() => paletteDialog.close()} aria-label="Close command palette">Esc</button>
  </div>
  <p class="eyebrow">NAVIGATION</p>
  {#each commands as command}<button class="command-row" onclick={() => navigate(command.id)}
      ><span>{command.icon} {command.label}</span><span>↵</span></button
    >{/each}{#if !commands.length && !operationCommands.length}<p class="muted">
      No matching commands.
    </p>{/if}{#if operationCommands.length}<p class="eyebrow">REVIEW AN OPERATION</p>
    {#each operationCommands as command}<button
        class="command-row"
        onclick={() => {
          onoperation(command.id);
        }}
        ><span
          >{command.label}<small
            >{command.available ? 'Review before applying' : command.reason}</small
          ></span
        ><span>↵</span></button
      >{/each}{/if}
  <div class="palette-hint">Tab to choose · Enter to open · Esc to close</div>
</dialog>
