import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { timingSafeEqual } from 'node:crypto';

export interface Personality {
  id: string;
  name: string;
  instructions: string;
  voice: string;
  greetingsEnabled: boolean;
  greetingPrompt: string;
  arrivalSeconds: number;
  departureSeconds: number;
  cooldownSeconds: number;
}

export interface Configuration {
  apiKey: string;
  accessToken: string;
  model: string;
  profiles: Personality[];
  fetch?: typeof globalThis.fetch;
}

export const tools = [
  {
    type: 'function', name: 'set_expression',
    description: 'Express BizBot’s own conversational tone with its eyes. This does not describe a person’s emotions.',
    parameters: {
      type: 'object', properties: { expression: { type: 'string', enum: ['neutral', 'happy', 'curious', 'thoughtful', 'surprised'] } },
      required: ['expression'], additionalProperties: false
    }
  },
  {
    type: 'function', name: 'capture_scene',
    description: 'Request a fresh front-camera image to answer a visual question. The image is added to the conversation if available.',
    parameters: { type: 'object', properties: {}, additionalProperties: false }
  }
];

function reply(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
}

function authorized(req: IncomingMessage, token: string): boolean {
  const actual = Buffer.from(req.headers.authorization ?? '');
  const expected = Buffer.from(`Bearer ${token}`);
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let length = 0;
  for await (const chunk of req) {
    length += chunk.length;
    if (length > 4096) throw new Error('body-too-large');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

export function makeServer(config: Configuration) {
  if (!config.apiKey || config.accessToken.length < 32) throw new Error('Set OPENAI_API_KEY and a BIZBOT_ACCESS_TOKEN of at least 32 characters.');
  const upstream = config.fetch ?? globalThis.fetch;
  // Single-iPad prototype: cap credential issuance globally, without storing client identifiers.
  let issued: number[] = [];
  const server = createServer(async (req, res) => {
    if (req.url === '/health' && req.method === 'GET') { reply(res, 200, { status: 'ok' }); return; }
    if (req.url !== '/session') { reply(res, 404, { error: 'not_found' }); return; }
    if (req.method !== 'POST') { reply(res, 405, { error: 'method_not_allowed' }); return; }
    if (!authorized(req, config.accessToken)) { reply(res, 401, { error: 'unauthorized' }); return; }
    let body: unknown;
    try { body = await readBody(req); }
    catch { reply(res, 400, { error: 'invalid_request' }); return; }
    const id = body && typeof body === 'object' && 'profileId' in body ? body.profileId : undefined;
    const profile = config.profiles.find(p => p.id === id);
    if (!profile) { reply(res, 400, { error: 'unknown_profile' }); return; }
    const now = Date.now();
    issued = issued.filter(t => now - t < 60_000);
    if (issued.length >= 10) { reply(res, 429, { error: 'try_again_later' }); return; }
    issued.push(now);
    try {
      const response = await upstream('https://api.openai.com/v1/realtime/client_secrets', {
        method: 'POST',
        headers: { Authorization: `Bearer ${config.apiKey}`, 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(12_000),
        body: JSON.stringify({
          expires_after: { anchor: 'created_at', seconds: 60 },
          session: {
            type: 'realtime', model: config.model,
            instructions: profile.instructions,
            output_modalities: ['audio'],
            max_output_tokens: 512,
            audio: {
              input: {
                format: { type: 'audio/pcm', rate: 24000 },
                noise_reduction: { type: 'far_field' },
                turn_detection: { type: 'semantic_vad', eagerness: 'medium', create_response: true, interrupt_response: true }
              },
              output: { format: { type: 'audio/pcm', rate: 24000 }, voice: profile.voice }
            },
            tools, tool_choice: 'auto'
          }
        })
      });
      if (!response.ok) {
        // Upstream bodies may contain sensitive information. Never log or return them.
        reply(res, response.status === 429 ? 429 : 502, { error: response.status === 429 ? 'provider_rate_limited' : 'provider_unavailable' });
        return;
      }
      const data = await response.json() as { value?: unknown; expires_at?: unknown };
      if (typeof data.value !== 'string' || typeof data.expires_at !== 'number' || data.expires_at <= Date.now() / 1000) {
        reply(res, 502, { error: 'invalid_provider_response' }); return;
      }
      reply(res, 200, { clientSecret: data.value, expiresAt: data.expires_at, model: config.model });
    } catch { reply(res, 502, { error: 'provider_unavailable' }); }
  });
  server.requestTimeout = 15_000;
  server.headersTimeout = 10_000;
  return server;
}
