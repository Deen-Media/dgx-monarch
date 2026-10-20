import type { Binding, Draft } from './model';

/** Keep visual layout edits and the repair editor on one workflow representation. */
export function layoutWorkflow(workflow: NonNullable<Draft['workflow']>, bindings: Binding[]) {
  const updated = structuredClone(workflow);
  updated.extra = {
    ...updated.extra,
    dgxm_studio: {
      ...(updated.extra?.dgxm_studio ?? {}),
      bindings: {
        ...(updated.extra?.dgxm_studio?.bindings ?? {}),
        ...Object.fromEntries(
          bindings.map((binding) => [
            binding.id,
            {
              ...(updated.extra?.dgxm_studio?.bindings?.[binding.id] ?? {}),
              label: binding.label,
              prominent: binding.prominent,
              order: binding.order,
              group: binding.group,
            },
          ]),
        ),
      },
    },
  };
  return { workflow: updated, rawWorkflow: JSON.stringify(updated, null, 2) };
}

/** Revalidating the unchanged source must not discard edits in the typed controls. */
export function unchangedWorkflow(workflow: Draft['workflow'], edited: unknown): boolean {
  return JSON.stringify(workflow) === JSON.stringify(edited);
}
