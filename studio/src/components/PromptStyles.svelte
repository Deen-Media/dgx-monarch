<script lang="ts">
  import type { Binding, Draft } from '../model';
  import { fieldLabel, bindingType } from '../model';
  import type { ChangeDraft } from '../contracts';
  import { moveStyle, styleEndNote, styleName } from '../promptStyles';
  let {
    draft,
    bindings,
    changeDraft,
  }: { draft: Draft; bindings: Binding[]; changeDraft: ChangeDraft } = $props();
  let textInputs = $derived(
    bindings.filter((binding) => binding.resolved !== false && bindingType(binding) === 'STRING'),
  );
  /** One draft change per move, so Undo puts the style back where it was. */
  function move(index: number, offset: number) {
    const styles = moveStyle(draft.styles, index, offset);
    if (styles) changeDraft({ styles });
  }
</script>

{#if textInputs.length}<div class="surface">
    <h3>Ordered prompt styles</h3>
    <p class="muted">
      Styles apply from top to bottom, to the prompt and the negative prompt separately. {'{prompt}'}
      in a style stands for the text so far; a style without it adds its text at the end, after a comma.
    </p>
    <p class="muted">
      For example, "a red fox" with "oil painting" then "(({'{prompt}'}))" gives "((a red fox, oil
      painting))"; the other order gives "((a red fox)), oil painting".
    </p>
    <p class="muted">Make styles in Library under styles, then add one here with Append style.</p>
    <label for="style-positive">Prompt input</label><select
      id="style-positive"
      value={draft.style_bindings?.prompt ?? ''}
      onchange={(e) =>
        changeDraft({
          style_bindings: {
            ...draft.style_bindings,
            prompt: e.currentTarget.value,
          },
        })}
      ><option value="">Choose the prompt binding</option>{#each textInputs as field}<option
          value={field.id}>{fieldLabel(field)} · {field.node_id}</option
        >{/each}</select
    ><label for="style-negative">Negative prompt input · optional</label><select
      id="style-negative"
      value={draft.style_bindings?.negative_prompt ?? ''}
      onchange={(e) =>
        changeDraft({
          style_bindings: Object.fromEntries(
            Object.entries({
              ...draft.style_bindings,
              negative_prompt: e.currentTarget.value,
            }).filter(([, value]) => !!value),
          ),
        })}
      ><option value="">No negative input</option>{#each textInputs as field}<option
          value={field.id}>{fieldLabel(field)} · {field.node_id}</option
        >{/each}</select
    >{#each draft.styles as style, i}{@const note = styleEndNote(i, draft.styles.length)}
      <div class="style-card">
        <div class="stack-row">
          <span>{i + 1}. {styleName(style, i)}</span><button
            onclick={() => changeDraft({ styles: draft.styles.filter((_, index) => index !== i) })}
            aria-label={`Remove ${styleName(style, i)}`}>×</button
          >
        </div>
        <div class="workflow-actions">
          <button
            disabled={i === 0}
            aria-describedby={i === 0 ? `style-note-${i}` : undefined}
            onclick={() => move(i, -1)}>Move up</button
          ><button
            disabled={i === draft.styles.length - 1}
            aria-describedby={i === draft.styles.length - 1 ? `style-note-${i}` : undefined}
            onclick={() => move(i, 1)}>Move down</button
          >{#if note}<small id={`style-note-${i}`}>{note}</small>{/if}
        </div>
      </div>{/each}<small>Submission review shows the effective text.</small>
  </div>{/if}
