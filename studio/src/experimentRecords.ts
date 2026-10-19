import { restoreDraft } from './model.ts';
import type { RecordData } from './contracts';

/** Restore only an editable definition. A saved outcome never becomes an executable job. */
export function experimentDefinition(data: RecordData) {
  const request = data.grid_request as Record<string, unknown> | undefined;
  if (!request || typeof request !== 'object')
    throw new Error('Experiment has no saved grid definition.');
  const axes = request.axes as Array<{ name: string; values: unknown[] }>;
  if (
    !Array.isArray(axes) ||
    axes.length < 1 ||
    axes.length > 2 ||
    axes.some((axis) => !axis || typeof axis.name !== 'string' || !Array.isArray(axis.values))
  )
    throw new Error('This editor requires one or two named grid axes.');
  if (!['fixed', 'increment'].includes(String(request.seed_policy ?? 'fixed')))
    throw new Error('Unsupported saved seed policy.');
  const excluded = request.excluded ?? [];
  if (
    !Array.isArray(excluded) ||
    excluded.some((value) => !Number.isSafeInteger(value) || value < 0)
  )
    throw new Error('Invalid saved grid exclusions.');
  return {
    draft: restoreDraft(request.draft),
    axisA: axes[0].name,
    valuesA: JSON.stringify(axes[0].values),
    axisB: axes[1]?.name ?? '',
    valuesB: JSON.stringify(axes[1]?.values ?? [1]),
    seedPolicy: String(request.seed_policy ?? 'fixed'),
    seedBinding: String(request.seed_binding ?? ''),
    baseSeed: typeof request.base_seed === 'number' ? request.base_seed : 0,
    excluded: excluded.map((value) => value + 1).join(','),
  };
}
