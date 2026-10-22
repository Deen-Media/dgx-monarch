import type { StudioClient } from './api';
import type { Inspection, Workflow } from './contracts';
import { droppedInspection, type InspectionGuard } from './inspectionGuard.ts';
import type { Binding, Draft } from './model';

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
 * travel with the workflow, so an edited dynamic choice lists its own child inputs.
 */
export async function inspectDraft(
  client: StudioClient,
  workflow: Workflow,
  name: string,
  priorValues?: Draft['values'],
) {
  const result = await client.request<Inspection>(
    '/workflows/inspect',
    priorValues ? { workflow, values: priorValues } : { workflow },
  );
  const bindings = result.bindings ?? [];
  const original = result.workflow ?? workflow;
  return {
    bindings,
    connections: result.connections ?? [],
    errors: result.errors ?? [],
    patch: { workflow: original, workflowName: name, values: bindingValues(bindings, priorValues) },
    rawWorkflow: JSON.stringify(original, null, 2),
    controlTab: bindings.some((binding) => binding.prominent) ? 'inputs' : 'advanced',
  };
}

/**
 * Inspect again after a dynamic choice changed, so the bindings list that choice's inputs, and
 * return them with the draft whose values follow them. Returns null when a newer inspection or
 * another workflow superseded it; an ordinary field edited meanwhile keeps its value. A changed
 * connection refuses with the usual reason.
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
  return { ...result, draft: { ...draft, values: bindingValues(result.bindings, draft.values) } };
}
