import type { StudioClient } from './api';
import type { Inspection, Workflow } from './contracts';
import type { Draft } from './model';

/** Resolve editable bindings once; connected sockets stay owned by the graph. */
export async function inspectDraft(
  client: StudioClient,
  workflow: Workflow,
  name: string,
  priorValues?: Draft['values'],
) {
  const result = await client.request<Inspection>('/workflows/inspect', { workflow });
  const bindings = result.bindings ?? [];
  const values = Object.fromEntries(
    bindings
      .filter((binding) => binding.resolved !== false)
      .map((binding) => [
        binding.id,
        priorValues?.[binding.id] ?? binding.value ?? binding.schema?.default ?? binding.default,
      ]),
  );
  const original = result.workflow ?? workflow;
  return {
    bindings,
    errors: result.errors ?? [],
    patch: { workflow: original, workflowName: name, values },
    rawWorkflow: JSON.stringify(original, null, 2),
    controlTab: bindings.some((binding) => binding.prominent) ? 'inputs' : 'advanced',
  };
}
