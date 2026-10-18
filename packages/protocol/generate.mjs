import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const schemas = [
  ['tdl-protocol-v1.schema.json', 'TdlProtocolV1'],
  ['tdl-result-v1.schema.json', 'TdlResultV1'],
  ['tdl-report-v1.schema.json', 'TdlReportV1'],
];
const definitions = [];
for (const [file, name] of schemas) {
  const schema = JSON.parse(await readFile(resolve(root, 'schemas', file), 'utf8'));
  definitions.push(`// Source: schemas/${file}\nexport const ${name}Schema = ${JSON.stringify(schema)} as const;`);
}
await writeFile(new URL('./src/schema-snapshots.ts', import.meta.url), `${definitions.join('\n\n')}\n`);
