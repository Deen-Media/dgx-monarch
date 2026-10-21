// Test support: render real Svelte components on the server without a DOM or a new dependency.
// Node module hooks compile `.svelte` files and `.svelte.ts` rune modules with the installed
// compiler and resolve the extensionless relative imports that Vite resolves for the bundle.
import { register } from 'node:module';

const source = new URL('./', import.meta.url).href;
const hooks = `
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const { compile, compileModule } = await import(${JSON.stringify(import.meta.resolve('svelte/compiler'))});
const ts = (await import(${JSON.stringify(import.meta.resolve('typescript'))})).default;
const source = ${JSON.stringify(source)};
const exists = (url) => existsSync(fileURLToPath(url));
export async function resolve(specifier, context, next) {
  if (/^\\.\\.?\\//.test(specifier) && context.parentURL?.startsWith(source)) {
    const url = new URL(specifier, context.parentURL);
    if (!exists(url))
      for (const suffix of ['.ts', '.js'])
        if (exists(url.href + suffix)) return { url: url.href + suffix, shortCircuit: true };
  }
  return next(specifier, context);
}
export async function load(url, context, next) {
  if (!url.startsWith(source)) return next(url, context);
  const filename = fileURLToPath(url);
  if (url.endsWith('.svelte')) {
    const { js } = compile(await readFile(filename, 'utf8'), { filename, generate: 'server' });
    return { format: 'module', source: js.code, shortCircuit: true };
  }
  if (url.endsWith('.svelte.ts')) {
    const plain = ts.transpile(await readFile(filename, 'utf8'), {
      target: ts.ScriptTarget.ESNext,
      module: ts.ModuleKind.ESNext,
      verbatimModuleSyntax: true,
    });
    const { js } = compileModule(plain, { filename, generate: 'server' });
    return { format: 'module', source: js.code, shortCircuit: true };
  }
  if (url.endsWith('.css')) return { format: 'module', source: 'export default "";', shortCircuit: true };
  return next(url, context);
}
`;
register(`data:text/javascript,${encodeURIComponent(hooks)}`);

const { render } = await import('svelte/server');

/** Server-render a component under studio/src and return its HTML body. */
export async function renderComponent(path, props) {
  const component = (await import(new URL(path, import.meta.url).href)).default;
  return render(component, { props }).body;
}

/**
 * Transpiled source of the named functions declared in a component's script, so a test can run
 * the shipped handlers against fakes. Undeclared names fail loudly under "use strict".
 */
export async function componentFunctions(path, names) {
  const ts = (await import('typescript')).default;
  const { readFile } = await import('node:fs/promises');
  const component = await readFile(new URL(path, import.meta.url), 'utf8');
  const script = component.slice(component.indexOf('>') + 1, component.indexOf('</script>'));
  const ast = ts.createSourceFile('component.ts', script, ts.ScriptTarget.Latest, true);
  return names
    .map((name) => {
      const node = ast.statements.find(
        (statement) => ts.isFunctionDeclaration(statement) && statement.name?.text === name,
      );
      if (!node) throw new Error(`${path} declares no function ${name}`);
      return ts.transpile(node.getText(ast), { target: ts.ScriptTarget.ESNext });
    })
    .join('\n');
}

/** Visible text of a server-rendered body, with markup and hydration comments removed. */
export function visibleText(html) {
  return html
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
