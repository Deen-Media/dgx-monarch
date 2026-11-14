import type { StudioClient } from './api';
import type { Inspection, Workflow } from './contracts';
import { droppedInspection, type InspectionGuard } from './inspectionGuard.ts';
import { stampIssues } from './issueRefresh.ts';
import type { Binding, Draft } from './model';
import type { ObjectInfo } from './workflowGraph.ts';
import { fittedOperation, workflowTraits } from './workflowTraits.ts';

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
 * workflow, which has no prior values, takes `fit`: the draft keeps its operation when the graph
 * supports it, and otherwise takes the graph's first supported one (fittedOperation). `fit` reads
 * the draft's operation when the request starts and again when the answer arrives, so a choice
 * made while the request runs stays as it is.
 */
export async function inspectDraft(
  client: StudioClient,
  workflow: Workflow,
  name: string,
  priorValues?: Draft['values'],
  fit?: { operation: () => string; info?: ObjectInfo },
) {
  const started = fit?.operation();
  const result = await client.request<Inspection>(
    '/workflows/inspect',
    priorValues ? { workflow, values: priorValues } : { workflow },
  );
  const bindings = result.bindings ?? [];
  const original = result.workflow ?? workflow;
  const values = bindingValues(bindings, priorValues);
  // A graph the inspection could not read lists no inputs, so its operations stay unknown.
  const current = fit?.operation();
  const operation =
    fit && current === started && !priorValues && bindings.length
      ? fittedOperation(current!, workflowTraits(original, bindings, values, fit.info))
      : undefined;
  return {
    bindings,
    connections: result.connections ?? [],
    errors: result.errors ?? [],
    patch: { workflow: original, workflowName: name, values, ...(operation ? { operation } : {}) },
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
