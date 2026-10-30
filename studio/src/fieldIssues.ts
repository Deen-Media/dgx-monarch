import type { Binding } from './model.ts';
import { bindingError, fieldLabel } from './model.ts';

/** A field the draft cannot submit as it is, or that the workflow check names. */
export interface FieldIssue {
  /** The binding's id; Field gives the field's control the element id `field-${id}`. */
  id: string;
  label: string;
  /** The node that holds the input, after any subgraph instances that hold it, such as 3 or 12/5. */
  node?: string;
  /** The title the workflow gives the node, when it gives one. */
  title?: string;
  message: string;
  /** Listed on the Inputs tab as well as on All inputs. */
  prominent: boolean;
}
/** One issue in Create's summary, with the field it concerns when Studio can find it. */
export interface Issue {
  text: string;
  field?: FieldIssue;
}

type Json = Record<string, any>;
const isObject = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/** The serialized node that holds a binding: an API prompt entry, or a node of its graph. */
function nodeOf(workflow: Json | null | undefined, binding: Binding): Json | undefined {
  if (!isObject(workflow) || binding.node_id === undefined) return undefined;
  if (!Array.isArray(workflow.nodes)) {
    const node = workflow[binding.node_id];
    return isObject(node) ? node : undefined;
  }
  const graphs: Json[] = [workflow, ...(workflow.definitions?.subgraphs ?? [])].filter(isObject);
  const graph = graphs.find((item) => String(item.id) === binding.graph_id) ?? workflow;
  const nodes: unknown[] = Array.isArray(graph.nodes) ? graph.nodes : [];
  return nodes.find((node): node is Json => isObject(node) && String(node.id) === binding.node_id);
}

/** A field issue for a binding and a message, named for people. */
export function fieldIssue(binding: Binding, message: string, workflow?: Json | null): FieldIssue {
  const node = nodeOf(workflow, binding);
  const title = node?._meta?.title ?? node?.title;
  return {
    id: binding.id,
    label: fieldLabel(binding),
    ...(binding.node_id !== undefined
      ? { node: [...(binding.instance_path ?? []), binding.node_id].join('/') }
      : {}),
    ...(typeof title === 'string' && title ? { title } : {}),
    message,
    prominent: !!(binding.prominent || binding.display?.prominent),
  };
}

/** The fields the draft cannot submit as they are, Inputs fields first, each in its list order. */
export function fieldIssues(
  bindings: Binding[],
  values: Record<string, unknown>,
  workflow?: Json | null,
): FieldIssue[] {
  const rank = (binding: Binding) => [
    binding.prominent || binding.display?.prominent ? 0 : 1,
    Number(binding.order ?? 0),
  ];
  return bindings
    .filter((binding) => binding.resolved !== false)
    .map((binding) => ({ binding, message: bindingError(binding, values[binding.id]) }))
    .filter((row): row is { binding: Binding; message: string } => !!row.message)
    .sort((a, b) => {
      const [x, y] = [rank(a.binding), rank(b.binding)];
      return x[0] - y[0] || x[1] - y[1];
    })
    .map(({ binding, message }) => fieldIssue(binding, message, workflow));
}

const sentence = (text: string) => (/[.!?]$/.test(text) ? text : `${text}.`);

/** An issue in words: "unet_name on node 3: Choose an available value." */
export function issueText(issue: FieldIssue): string {
  const node = issue.node === undefined ? '' : ` on node ${issue.node}`;
  return `${issue.label}${node}${issue.title ? ` (${issue.title})` : ''}: ${sentence(issue.message)}`;
}

/**
 * The workflow check's issues. The companion names a value it refuses as `node.input: problem`,
 * by the node's id in the compiled prompt; such an issue names its field. Any other issue, such as
 * an App Mode input it cannot resolve, stays as the companion wrote it.
 */
export function inspectionIssues(
  errors: string[],
  bindings: Binding[],
  workflow?: Json | null,
): Issue[] {
  return errors.map((error) => {
    const matches = bindings
      .map((binding) => ({
        binding,
        prefix: `${binding.execution_node_id ?? binding.node_id}.${binding.input_name}: `,
      }))
      .filter(({ binding, prefix }) => binding.input_name && error.startsWith(prefix))
      .sort((a, b) => b.prefix.length - a.prefix.length);
    if (!matches.length) return { text: error };
    const field = fieldIssue(matches[0].binding, error.slice(matches[0].prefix.length), workflow);
    return { text: issueText(field), field };
  });
}

/** The summary's issues: each field issue, then each workflow issue about another field or none. */
export function summaryIssues(fields: FieldIssue[], inspection: Issue[]): Issue[] {
  const named = new Set(fields.map((field) => field.id));
  return [
    ...fields.map((field) => ({ text: issueText(field), field })),
    ...inspection.filter((issue) => !issue.field || !named.has(issue.field.id)),
  ];
}

/** The tab that shows a field: the one open when it lists the field, else Inputs or All inputs. */
export function fieldTab(field: { prominent: boolean }, current: string): string {
  if (current === 'advanced' || (current === 'inputs' && field.prominent)) return current;
  return field.prominent ? 'inputs' : 'advanced';
}

interface Revealable {
  disabled?: boolean;
  closest?: (selector: string) => { open: boolean } | null;
  scrollIntoView: (options: ScrollIntoViewOptions) => void;
  focus: (options?: FocusOptions) => void;
}
interface Page {
  getElementById(id: string): unknown;
}

/**
 * Bring an element into view and give it focus: a field's control, or a block such as the list of
 * workflow issues. A closed disclosure around it opens; a disabled control is shown, not focused.
 */
export function reveal(page: Page, id: string): boolean {
  const target = page.getElementById(id) as Revealable | null;
  if (!target) return false;
  const disclosure = target.closest?.('details');
  if (disclosure && !disclosure.open) disclosure.open = true;
  target.scrollIntoView({ block: 'center' });
  if (!target.disabled) target.focus({ preventScroll: true });
  return true;
}
