import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const schemas = [
  ['tdl-protocol-v1.schema.json', 'TdlProtocolV1'],
  ['tdl-result-v1.schema.json', 'TdlResultV1'],
  ['tdl-report-v1.schema.json', 'TdlReportV1'],
  ['job-v1.schema.json', 'JobV1'],
];
const definitions = [];
const typeOf = (schema) => {
  if (Object.hasOwn(schema, 'const')) return JSON.stringify(schema.const);
  if (Array.isArray(schema.enum)) return schema.enum.map(JSON.stringify).join(' | ');
  if (schema.type === 'object') {
    const required = new Set(schema.required || []);
    const fields = Object.entries(schema.properties || {}).map(([key, value]) =>
      `  ${JSON.stringify(key)}${required.has(key) ? '' : '?'}: ${typeOf(value)};`).join('\n');
    return `{\n${fields}\n}`;
  }
  if (schema.type === 'array') return `Array<${typeOf(schema.items || {})}>`;
  return ({string:'string',integer:'number',number:'number',boolean:'boolean'})[schema.type] || 'unknown';
};
for (const [file, name] of schemas) {
  const schema = JSON.parse(await readFile(resolve(root, 'schemas', file), 'utf8'));
  definitions.push(`// Source: schemas/${file}\nexport const ${name}Schema = ${JSON.stringify(schema)};`);
  if (name === 'JobV1') definitions.push(`export type JobV1 = ${typeOf(schema)};`);
}
await writeFile(new URL('./src/schema-snapshots.ts', import.meta.url), `${definitions.join('\n\n')}\n`);
