import type { StudioClient } from './api';
import type { Inspection, Workflow } from './contracts';
import { droppedInspection, type InspectionGuard } from './inspectionGuard.ts';
import { stampIssues } from './issueRefresh.ts';
import type { Binding, Draft } from './model';
import { freshGraphFields } from './model.ts';
import { graphNodes, type ObjectInfo } from './workflowGraph.ts';
import { fittedOperation, workflowTraits } from './workflowTraits.ts';

/** The node classes the companion fills from the LoRA stack by position (lora_graph.LORA_NODES). */
export const LORA_SLOT_CLASSES = new Set([
  'LoraLoader',
  'LoraLoaderModelOnly',
  'DGXMonarchLoraLoader',
]);

/**
 * The notice for a workflow loaded in place of the draft's, or ''. It names the style input and
 * the LoRA insertion boundary the draft held, which a newly loaded workflow starts without
 * (freshGraphFields), and says when a LoRA stack will replace the loaded workflow's own LoRAs.
 */
export function freshGraphNotice(
  held: Pick<Draft, 'style_bindings' | 'lora_insertion' | 'lora_stack'>,
  workflow: Workflow | null | undefined,
  info?: ObjectInfo,
): string {
  const style = Object.values(held.style_bindings ?? {}).some(Boolean);
  const boundary = !!held.lora_insertion;
  const stack = held.lora_stack.length > 0;
  const slots = graphNodes(workflow, info).some((node) => LORA_SLOT_CLASSES.has(node.type));
  // A workflow with LoRA loaders takes the stack into its own chain and refuses or ignores a
  // boundary (lora_graph.apply_loras), so the notice asks for a boundary only where one applies.
  const choose = [
    style ? 'the prompt input for styles' : '',
    boundary && !slots ? 'the LoRA insertion boundary' : '',
  ].filter(Boolean);
  const lines = style || boundary ? ['This workflow starts from its own values.'] : [];
  if (choose.length) lines.push(`Choose ${choose.join(' and ')} again.`);
  if (slots && (boundary || stack))
    lines.push(
      `This workflow has LoRAs of its own${boundary ? ', so the LoRA insertion boundary is cleared' : ''}.`,
    );
  // A non-empty stack, even one with every entry off, takes over the slots: entry by entry, and
  // strength 0 for each slot no enabled entry reaches.
  if (slots && stack)
    lines.push(
      'At Generate, the LoRA stack replaces them by position, and those it does not reach run at strength 0. Remove the stack on the LoRAs tab to keep them.',
    );
  return lines.join(' ');
}

/**
 * Each resolved binding's value: the draft's edit first (by ID, then by a legacy alias), then the
 * graph's own value or default. Values for inputs the bindings no longer list are dropped.
 */
export function bindingValues(bindings: Binding[], priorValues?: Draft['values']) {
  return Object.fromEntries(
    bindings
      .filter((binding) => binding.resolved !== false)
      .map((binding) => [
        binding.id,
        priorValues?.[binding.id] ??
          binding.aliases
            ?.map((alias) => priorValues?.[alias])
            .find((value) => value !== undefined) ??
          binding.value ??
          binding.schema?.default ??
          binding.default,
      ]),
  );
}

/**
 * Resolve editable bindings once; connected sockets stay owned by the graph. Prior draft values
 * travel with the workflow, so an edited dynamic choice lists its own child inputs. A newly loaded
 * workflow, which has no prior values, starts without the previous graph's style inputs and LoRA
 * insertion targets (freshGraphFields), and takes `fit`: the draft keeps its operation when the
 * graph supports it, and otherwise takes the graph's first supported one (fittedOperation).
 * `fit.asked`, when a load asks for an operation, as Set up Fleet does, is fitted in place of the
 * draft's, so the workflow and its operation are one draft change.
 * `fit` reads the draft's operation when the request starts and again when the answer arrives, so
 * a choice made while the request runs stays as it is.
 */
export async function inspectDraft(
  client: StudioClient,
  workflow: Workflow,
  name: string,
  priorValues?: Draft['values'],
  fit?: { operation: () => string; asked?: string; info?: ObjectInfo },
) {
  const started = fit?.operation();
  const result = await client.request<Inspection>(
    '/workflows/inspect',
    priorValues ? { workflow, values: priorValues } : { workflow },
  );
  const bindings = result.bindings ?? [];
  const original = result.workflow ?? workflow;
  const values = bindingValues(bindings, priorValues);
  // A graph the inspection could not read lists no inputs, so its operations stay unknown and only
  // an operation the load asks for applies.
  const current = fit?.operation();
  const operation =
    fit && current === started && !priorValues
      ? bindings.length
        ? fittedOperation(
            fit.asked || current!,
            workflowTraits(original, bindings, values, fit.info),
          )
        : fit.asked || undefined
      : undefined;
  return {
    bindings,
    connections: result.connections ?? [],
    errors: result.errors ?? [],
    patch: {
      ...(priorValues ? {} : freshGraphFields()),
      workflow: original,
      workflowName: name,
      values,
      ...(operation ? { operation } : {}),
    },
    rawWorkflow: JSON.stringify(original, null, 2),
    controlTab: bindings.some((binding) => binding.prominent) ? 'inputs' : 'advanced',
  };
}

/**
 * Inspect again after a dynamic choice changed, so the bindings list that choice's inputs, and
 * return them with the draft whose values follow them. Returns null when a newer inspection or
 * another workflow superseded it; an ordinary field edited meanwhile keeps its value. The errors
 * carry this inspection's ticket, so after such an edit Create sees them as stale and checks the
 * draft again. A changed connection refuses with the usual reason.
 */
export async function rebuildBindings(
  connection: { client: StudioClient },
  guard: InspectionGuard<unknown>,
  current: () => Draft,
  values: Draft['values'],
) {
  const { workflow, workflowName } = current(),
    client = connection.client;
  if (!workflow) return null;
  const ticket = guard.begin(client);
  const result = await inspectDraft(client, workflow, workflowName, values);
  const state = guard.inspected(ticket, connection.client);
  if (state === 'reconnected') throw new Error(droppedInspection(workflowName, state));
  const draft = current();
  if (state === 'newer' || draft.workflow !== workflow) return null;
  return {
    ...result,
    errors: stampIssues(result.errors, ticket),
    draft: { ...draft, values: bindingValues(result.bindings, draft.values) },
  };
}
