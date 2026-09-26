import { readFileSync, existsSync } from 'node:fs';
import { makeServer, type Personality } from './server.js';

// A tiny .env reader keeps runtime dependencies at zero. Run from server/.
if (existsSync('.env')) {
  for (const line of readFileSync('.env', 'utf8').split(/\r?\n/)) {
    const match = line.match(/^([A-Z_]+)=(.*)$/);
    if (match && process.env[match[1]] === undefined) {
      process.env[match[1]] = match[2].trim().replace(/^(['"])(.*)\1$/, '$2');
    }
  }
}

try {
  const profiles = JSON.parse(readFileSync(new URL('../../../shared/personalities.json', import.meta.url), 'utf8')) as Personality[];
  const server = makeServer({
    apiKey: process.env.OPENAI_API_KEY ?? '',
    accessToken: process.env.BIZBOT_ACCESS_TOKEN ?? '',
    model: process.env.OPENAI_REALTIME_MODEL ?? 'gpt-realtime', profiles
  });
  const port = Number(process.env.PORT ?? 8787);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be between 1 and 65535.');
  server.listen(port, process.env.HOST ?? '0.0.0.0', () => console.log(`BizBot session server listening on port ${port}.`));
  server.on('error', () => { console.error('Could not start the session server. Check HOST and PORT.'); process.exitCode = 1; });
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close());
} catch (error) {
  console.error(error instanceof Error ? error.message : 'Server configuration error.');
  process.exitCode = 1;
}
