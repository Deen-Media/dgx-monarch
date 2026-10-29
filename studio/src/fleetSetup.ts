import type { WorkflowEntry } from './contracts';
import type { Draft } from './model';
import { workflowTitle } from './workflowChoice.ts';
import { graphNodes } from './workflowGraph.ts';

/** The packaged Fleet workflow's catalog id; its title is "Fleet: many prompts, one per GPU". */
export const FLEET_WORKFLOW = 'workflow.dgx-monarch-fleet';
/** The node that spreads prompt lines over the GPUs. The draft's Operation only labels a job. */
export const FLEET_SAMPLER = 'DGXMonarchFleetKSampler';

/**
 * Whether a Fleet sampler runs in a UI workflow or an API prompt, read as Create's Operation select
 * reads the graph: muted and bypassed nodes do not count, nor a subgraph no running instance uses.
 */
export function hasFleetSampler(workflow: unknown): boolean {
  return graphNodes(workflow as Record<string, unknown> | null).some(
    (node) => node.type === FLEET_SAMPLER,
  );
}

/**
 * What Set up Fleet does: keep a draft that already holds a Fleet sampler, load the packaged
 * Fleet workflow, or, when the catalog lacks it, only set Operation and say what is missing.
 */
export type FleetPlan =
  | { kind: 'keep' }
  | { kind: 'load'; entry: WorkflowEntry; replaces: boolean }
  | { kind: 'missing' };

export function fleetPlan(draft: Draft, workflows: readonly WorkflowEntry[]): FleetPlan {
  if (hasFleetSampler(draft.workflow)) return { kind: 'keep' };
  const entry = workflows.find((item) => item.id === FLEET_WORKFLOW);
  return entry ? { kind: 'load', entry, replaces: !!draft.workflow } : { kind: 'missing' };
}

const LABEL =
  'Operation Fleet only labels the job for Gallery and this page; the Fleet sampler node in the workflow is what spreads the prompts over the GPUs.';

/** What the button will do, shown before it is pressed. */
export function fleetPlanText(plan: FleetPlan): string {
  if (plan.kind === 'keep')
    return `Your Create draft already holds a Fleet sampler. Set up Fleet keeps it, sets Operation to Fleet and opens Create. ${LABEL}`;
  if (plan.kind === 'load')
    return `Set up Fleet loads the packaged workflow "${workflowTitle(plan.entry)}" into Create${plan.replaces ? ' in place of your current workflow' : ''}, sets Operation to Fleet and opens Create. ${LABEL}`;
  return `The packaged Fleet workflow is not in this companion's workflow list. Set up Fleet sets Operation to Fleet and opens Create, where you load or import a workflow with a ${FLEET_SAMPLER} node. ${LABEL}`;
}

/** The notice Create shows once the setup is done, naming the one thing to do next. */
export function fleetNextStep(plan: FleetPlan): string {
  if (plan.kind === 'missing')
    return `Operation is set to Fleet, but this draft has no Fleet sampler, so it would render as one ordinary job. Load or import a workflow with a ${FLEET_SAMPLER} node next.`;
  const undo =
    plan.kind === 'load' && plan.replaces ? ' Undo brings back the workflow you had before.' : '';
  return `Fleet is ready in Create. Next, put your prompts in the prompts input, one per line: each line becomes its own job, line 1 uses noise_seed, line 2 noise_seed + 1 and so on. Then generate as usual.${undo}`;
}
